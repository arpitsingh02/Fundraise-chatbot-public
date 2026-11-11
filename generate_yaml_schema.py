#!/usr/bin/env python3
"""Generate an LLM-friendly YAML schema for the FULL financial-model SQLite database.

This script is meant to be used after you run:
  ingest_excel_to_sqlite.py

It inspects tables/columns/indexes + includes a compact NL->SQL playbook focused on the single
primary table: facts.

Run:
  python generate_yaml_schema.py --sqlite data/financial_model.sqlite --out data/financial_model_schema.yaml

"""

import argparse
import datetime as dt
import sqlite3
from pathlib import Path

import yaml


def get_tables(conn):
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()]


def table_info(conn, t):
    cols = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
    return [{
        'name': c[1],
        'type': c[2] or 'TEXT',
        'nullable': (c[3] == 0),
        'primary_key': bool(c[5]),
        'default': c[4],
    } for c in cols]


def index_list(conn, t):
    idx = conn.execute(f'PRAGMA index_list("{t}")').fetchall()
    out = []
    for i in idx:
        # seq, name, unique, origin, partial
        name = i[1]
        cols = [r[2] for r in conn.execute(f'PRAGMA index_info("{name}")').fetchall()]
        out.append({'name': name, 'unique': bool(i[2]), 'columns': cols})
    return out


def sample_rows(conn, t, limit=3):
    try:
        return conn.execute(f'SELECT * FROM "{t}" LIMIT {limit}').fetchall()
    except Exception:
        return []


def column_descriptions(table: str, col: str) -> str:
    if table == 'facts':
        m = {
            'metric': 'Metric path / label (often hierarchical, joined with " | "). Use LIKE for fuzzy matching.',
            'period': "Period label. If period_type='FY', values like 'FY27' or 'FY26 Apr-Sept'. If period_type='DATE', ISO 'YYYY-MM-DD'.",
            'period_type': "Either 'FY' (fiscal-year style) or 'DATE' (time-series dates).",
            'value': 'Numeric value (REAL).',
            'sheet': 'Excel sheet name where this metric came from.',
            'unit': "Optional unit hint (e.g., 'INR Mn', '%', '#'). May be NULL.",
        }
        return m.get(col, f'Column {col} in facts.')
    if table == 'glossary':
        return {'term': 'Business term.', 'definition': 'Definition/explanation.'}.get(col, f'Column {col} in glossary.')
    if table == 'sheet_registry':
        return {
            'sheet': 'Sheet name.',
            'parsed_blocks': 'Number of detected FY/DATE header blocks parsed into facts.',
            'notes': 'Parser notes for this sheet (what was ingested / fallback applied).'
        }.get(col, f'Column {col} in sheet_registry.')
    if table == 'raw_cells':
        return {
            'sheet': 'Sheet name.',
            'row': '1-based row index in the sheet.',
            'col': '1-based column index in the sheet.',
            'value_text': 'Raw cell value stored as text (fallback when we cannot parse as FY/DATE table).'
        }.get(col, f'Column {col} in raw_cells.')
    return f'Column {col} in {table}.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sqlite', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    conn = sqlite3.connect(args.sqlite)
    tables = get_tables(conn)

    table_docs = []
    for t in tables:
        cols = table_info(conn, t)
        for c in cols:
            c['description'] = column_descriptions(t, c['name'])
        cnt = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        table_docs.append({
            'name': t,
            'row_count': cnt,
            'description': (
                "Unified fact table for all FY and time-series numeric values." if t == 'facts'
                else "Glossary terms from workbook." if t == 'glossary'
                else "Per-sheet parsing status and notes." if t == 'sheet_registry'
                else "Fallback raw cell storage for sheets not parsed into facts." if t == 'raw_cells'
                else "Table."
            ),
            'columns': cols,
            'indexes': index_list(conn, t),
            'sample_rows': sample_rows(conn, t, 3),
        })

    # Helpful distinct values (keep small)
    distinct_sheets = []
    distinct_period_types = []
    try:
        distinct_sheets = [r[0] for r in conn.execute('SELECT DISTINCT sheet FROM facts LIMIT 30').fetchall()]
        distinct_period_types = [r[0] for r in conn.execute('SELECT DISTINCT period_type FROM facts').fetchall()]
    except Exception:
        pass

    schema = {
        'schema_name': 'financial_model_full_simple',
        'schema_version': '1.0',
        'generated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'db': {
            'dialect': 'sqlite',
            'file': args.sqlite,
        },
        'core_tables': {
            'primary_fact_table': 'facts',
            'glossary_table': 'glossary',
            'fallback_table': 'raw_cells',
            'sheet_status_table': 'sheet_registry',
        },
        'llm_sql_rules': [
            'Output ONLY SQL. No markdown fences. No explanation.',
            'Use ONLY tables/columns listed in this schema. Do not invent tables/columns.',
            "For FY questions (e.g., 'FY27'), query facts with period_type='FY' and period='FY27'.",
            "For time-series questions, query facts with period_type='DATE' and period as ISO 'YYYY-MM-DD'.",
            "Find metrics using LIKE on facts.metric (e.g., metric LIKE '%EBITDA%').",
            'Prefer single-table queries on facts; joins are rarely needed.',
            'Never use words like TABLE GUIDE / SHEET GUIDE in SQL identifiers.',
        ],
        'quick_reference': {
            'period_types': distinct_period_types,
            'example_sheets': distinct_sheets,
        },
        'tables': table_docs,
        'query_playbook': {
            'patterns': [
                {
                    'name': 'fy_metric_single_value',
                    'when': "User asks: 'EBITDA in FY27'",
                    'sql': "SELECT value, unit FROM facts WHERE period_type='FY' AND period='FY27' AND metric LIKE '%EBITDA%' LIMIT 20;"
                },
                {
                    'name': 'fy_metric_over_years',
                    'when': "User asks: 'Net Profit over FY22-FY30'",
                    'sql': "SELECT period, value FROM facts WHERE period_type='FY' AND metric LIKE '%Net Profit%' ORDER BY period;"
                },
                {
                    'name': 'time_series_metric',
                    'when': "User asks: 'Show monthly AUM trend'",
                    'sql': "SELECT period, value FROM facts WHERE period_type='DATE' AND metric LIKE '%AUM%' ORDER BY period LIMIT 200;"
                },
                {
                    'name': 'filter_by_sheet',
                    'when': "Need values from a specific sheet",
                    'sql': "SELECT period, value FROM facts WHERE sheet='Income Statement' AND metric LIKE '%Revenue%' AND period_type='FY' ORDER BY period;"
                },
                {
                    'name': 'glossary_lookup',
                    'when': "User asks 'What is Burn Rate?'",
                    'sql': "SELECT definition FROM glossary WHERE term LIKE '%Burn Rate%' LIMIT 5;"
                }
            ],
            'troubleshooting': [
                "If no rows return, broaden metric LIKE pattern (e.g., '%Earnings%' instead of '%Revenue%').",
                "If too many rows, add sheet filter or use LIMIT.",
                "If a sheet wasn't parsed into facts, look at sheet_registry.notes and use raw_cells as fallback."
            ]
        }
    }

    conn.close()

    Path(args.out).write_text(yaml.safe_dump(schema, sort_keys=False, allow_unicode=True, width=120), encoding='utf-8')
    print('Done. Wrote YAML:', args.out)


if __name__ == '__main__':
    main()
