#!/usr/bin/env python3
"""Ingest *entire* data/financial_model.xlsx into a SIMPLE, LLM-friendly SQLite.

Goals
-----
1) Capture ALL numeric/time-series/FY tables across *all* sheets.
2) Keep querying simple: one unified fact table.
3) Avoid the "crazy joins" problem by designing for single-table queries.

Output schema (SQLite)
----------------------
1) facts(metric, period, period_type, value, sheet, unit)
   - period_type in {'FY','DATE'}
   - FY periods are strings like 'FY27', 'FY26 Apr-Sept'
   - DATE periods are ISO 'YYYY-MM-DD'

2) glossary(term, definition)
3) sheet_registry(sheet, parsed_blocks, notes)
4) raw_cells(sheet, row, col, value_text)
   - fallback storage for sheets we can't parse as FY/DATE tables
   - only stores non-empty cells

Run
---
python ingest_excel_to_sqlite.py --excel data/financial_model.xlsx --sqlite data/financial_model.sqlite

"""

import argparse
import datetime as dt
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

from openpyxl import load_workbook
from openpyxl.utils.datetime import from_excel


# --- Period detection ---
FY_RE = re.compile(r"^FY\s*\d{2}(?:\s*.*)?$", re.IGNORECASE)


def norm_period_fy(x: str) -> str:
    """Normalize FY labels so users can ask FY27 (no spaces)."""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(?i)^fy\s+(\d{2})$", r"FY\1", s)
    s = re.sub(r"(?i)^fy\s+(\d{2})\s+", r"FY\1 ", s)
    return s


def is_excel_serial_date(v) -> bool:
    """Heuristic: Excel date serials in this workbook are ~ 44k-47k."""
    if isinstance(v, (int, float)):
        try:
            fv = float(v)
        except Exception:
            return False
        return 30000 <= fv <= 60000
    return False


def norm_period_date(v) -> Optional[str]:
    try:
        d = from_excel(v)
        if isinstance(d, dt.datetime):
            d = d.date()
        if isinstance(d, dt.date):
            return d.isoformat()
    except Exception:
        return None
    return None


# --- Value parsing ---

def to_float(v):
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if s in ("", "-", "—"):
                return None
            return float(s)
        return None
    except Exception:
        return None


def to_text(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    # keep numbers as text too for raw_cells
    return str(v)


# --- DB schema ---

def ensure_schema(conn: sqlite3.Connection):
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=OFF;

        DROP TABLE IF EXISTS facts;
        DROP TABLE IF EXISTS glossary;
        DROP TABLE IF EXISTS sheet_registry;
        DROP TABLE IF EXISTS raw_cells;

        CREATE TABLE facts(
            metric TEXT NOT NULL,
            period TEXT NOT NULL,
            period_type TEXT NOT NULL,  -- 'FY' or 'DATE'
            value REAL,
            sheet TEXT NOT NULL,
            unit TEXT,
            PRIMARY KEY(metric, period, period_type, sheet)
        );

        CREATE INDEX idx_facts_metric ON facts(metric);
        CREATE INDEX idx_facts_period ON facts(period);
        CREATE INDEX idx_facts_ptype  ON facts(period_type);
        CREATE INDEX idx_facts_sheet  ON facts(sheet);

        CREATE TABLE glossary(
            term TEXT PRIMARY KEY,
            definition TEXT
        );

        CREATE TABLE sheet_registry(
            sheet TEXT PRIMARY KEY,
            parsed_blocks INTEGER DEFAULT 0,
            notes TEXT
        );

        CREATE TABLE raw_cells(
            sheet TEXT NOT NULL,
            row INTEGER NOT NULL,
            col INTEGER NOT NULL,
            value_text TEXT,
            PRIMARY KEY(sheet, row, col)
        );
        """
    )
    conn.commit()


# --- Glossary ingest ---

def ingest_glossary(conn: sqlite3.Connection, wb):
    if "Glossary" not in wb.sheetnames:
        return
    ws = wb["Glossary"]
    out = []
    for r in range(1, ws.max_row + 1):
        vals = [ws.cell(r, c).value for c in range(1, min(ws.max_column, 12) + 1)]
        vals_clean = [v for v in vals if v is not None and str(v).strip() != ""]
        if len(vals_clean) < 2:
            continue
        term = str(vals_clean[-2]).strip()
        definition = str(vals_clean[-1]).strip()
        if term.lower() in ("glossary", "term"):
            continue
        out.append((term, definition))
    if out:
        conn.executemany("INSERT OR REPLACE INTO glossary(term, definition) VALUES (?,?)", out)
        conn.commit()


# --- Header detection ---

def find_header_blocks(ws, search_rows=160, search_cols=800, min_fy=4, min_dates=8):
    """Find parseable blocks in a sheet.

    Returns list of blocks:
      { 'header_row': int, 'period_type': 'FY'|'DATE', 'cols': [int], 'periods': [str], 'first_period_col': int }

    Supports rows that contain BOTH FY and DATE headers.
    """
    blocks = []
    for r in range(1, min(search_rows, ws.max_row) + 1):
        fy_cols, fy_periods = [], []
        date_cols, date_periods = [], []
        for c in range(1, min(search_cols, ws.max_column) + 1):
            v = ws.cell(r, c).value
            if isinstance(v, str) and FY_RE.match(v.strip()):
                fy_cols.append(c)
                fy_periods.append(norm_period_fy(v))
            elif is_excel_serial_date(v):
                iso = norm_period_date(v)
                if iso:
                    date_cols.append(c)
                    date_periods.append(iso)

        # Heuristic: accept blocks when enough columns
        if len(fy_cols) >= min_fy:
            blocks.append({
                'header_row': r,
                'period_type': 'FY',
                'cols': fy_cols,
                'periods': fy_periods,
                'first_period_col': min(fy_cols)
            })
        if len(date_cols) >= min_dates:
            blocks.append({
                'header_row': r,
                'period_type': 'DATE',
                'cols': date_cols,
                'periods': date_periods,
                'first_period_col': min(date_cols)
            })

        # If we found something early, stop scanning deeper to avoid picking up repeated headers too far down.
        if blocks:
            # Keep only blocks from this first header row (avoid later repeats)
            blocks = [b for b in blocks if b['header_row'] == r]
            break

    return blocks


def infer_unit_from_sheet(sheet_name: str, metric_path: str) -> Optional[str]:
    s = metric_path.lower()
    if '%' in metric_path:
        return '%'
    if "('000" in s or "000" in s and "inr" not in s:
        return '#'
    if "inr" in s and ("mn" in s or "million" in s):
        return 'INR Mn'
    if "cr" in s or "crore" in s:
        return 'INR Cr'
    return None


def build_metric_path(label_vals: List[Optional[str]]) -> Optional[str]:
    parts = []
    for v in label_vals:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none"):
            continue
        parts.append(s)
    if not parts:
        return None
    # Collapse duplicates like '2. Financials' repeated
    out = []
    for p in parts:
        if not out or out[-1] != p:
            out.append(p)
    return " | ".join(out)


# --- Main sheet ingester ---

def ingest_blocks_to_facts(conn: sqlite3.Connection, ws, sheet_name: str, blocks: List[Dict]):
    cur = conn.cursor()
    inserted = 0

    for block in blocks:
        header_row = block['header_row']
        cols = block['cols']
        periods = block['periods']
        period_type = block['period_type']
        first_period_col = block['first_period_col']

        # Label columns: up to 6 columns before the first period column
        label_cols = list(range(1, max(1, first_period_col)))
        label_cols = label_cols[:6]

        # Forward-fill label columns
        last_labels = [None] * len(label_cols)

        blank_run = 0
        for r in range(header_row + 1, ws.max_row + 1):
            row_vals = [ws.cell(r, c).value for c in label_cols]
            # detect end of table
            if all(v is None or str(v).strip() == "" for v in row_vals):
                blank_run += 1
                if blank_run > 25:
                    break
                continue
            blank_run = 0

            # update ffill labels
            for i, v in enumerate(row_vals):
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    row_vals[i] = last_labels[i]
                else:
                    last_labels[i] = v

            metric = build_metric_path(row_vals)
            if not metric:
                continue

            unit = infer_unit_from_sheet(sheet_name, metric)

            # Insert per period col
            for c, p in zip(cols, periods):
                fv = to_float(ws.cell(r, c).value)
                if fv is None:
                    continue
                cur.execute(
                    "INSERT OR REPLACE INTO facts(metric, period, period_type, value, sheet, unit) VALUES (?,?,?,?,?,?)",
                    (metric, p, period_type, float(fv), sheet_name, unit)
                )
                inserted += 1

    conn.commit()
    return inserted


def ingest_raw_cells(conn: sqlite3.Connection, ws, sheet_name: str, max_cells: int = 250000):
    """Fallback: store all non-empty cells as text.

    max_cells is a safety limit to avoid huge writes.
    """
    cur = conn.cursor()
    written = 0
    # estimate upper bound
    total = ws.max_row * ws.max_column
    if total > max_cells:
        return 0, f"raw_cells skipped (sheet too large: {ws.max_row}x{ws.max_column})"

    for r in range(1, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(r, c).value
            vt = to_text(v)
            if vt is None:
                continue
            cur.execute(
                "INSERT OR REPLACE INTO raw_cells(sheet, row, col, value_text) VALUES (?,?,?,?)",
                (sheet_name, r, c, vt)
            )
            written += 1
    conn.commit()
    return written, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--excel', required=True)
    ap.add_argument('--sqlite', required=True)
    args = ap.parse_args()

    wb = load_workbook(args.excel, data_only=True)
    conn = sqlite3.connect(args.sqlite)
    ensure_schema(conn)

    # Glossary first
    ingest_glossary(conn, wb)

    total_inserted = 0

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]

        # Always register sheet
        conn.execute("INSERT OR REPLACE INTO sheet_registry(sheet, parsed_blocks, notes) VALUES (?,?,?)", (sheet_name, 0, None))
        conn.commit()

        # Skip obviously empty/cover-like sheets? We'll still raw-ingest if no blocks.
        blocks = []
        if sheet_name != 'Glossary':
            blocks = find_header_blocks(ws)

        if blocks:
            ins = ingest_blocks_to_facts(conn, ws, sheet_name, blocks)
            total_inserted += ins
            conn.execute("UPDATE sheet_registry SET parsed_blocks=?, notes=? WHERE sheet=?",
                         (len(blocks), f"parsed {len(blocks)} block(s), inserted {ins} facts", sheet_name))
            conn.commit()
        else:
            # fallback raw
            written, note = ingest_raw_cells(conn, ws, sheet_name)
            msg = note or f"no FY/DATE header detected; stored {written} raw cells"
            conn.execute("UPDATE sheet_registry SET parsed_blocks=?, notes=? WHERE sheet=?", (0, msg, sheet_name))
            conn.commit()

    # metadata
    conn.execute("INSERT OR REPLACE INTO sheet_registry(sheet, parsed_blocks, notes) VALUES (?,?,?)",
                 ('__meta__', 0, f"source_excel={args.excel}"))
    conn.commit()

    conn.close()
    print(f"Done. Wrote SQLite: {args.sqlite}. Inserted {total_inserted} fact rows.")


if __name__ == '__main__':
    main()
