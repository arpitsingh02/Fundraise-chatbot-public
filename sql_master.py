# sql_master.py
# Fixes over v8:
# 1) FAQ retrieval no longer depends on an LLM "snippet selector" (which could discard relevant content).
#    We pass top retrieved FAQ chunks directly into the final answer prompt.
# 2) Final prompt forbids "I am an AI model" style self-talk and forbids placeholders.
# 3) FAQ store robustness: tries the configured FAQ_STORE_DIR first, then also tries ./vector_stores/faq_store_alt.
# 4) SQL: still always attempted; if empty, system naturally falls back to FAQ.
#
# NOTE: Paths from v6 are preserved exactly: DB_PATH, SCHEMA_STORE_DIR, OV_MODEL_PATH, TOKENIZER_PATH.

import os
import re
import warnings
import sqlite3
from typing import Any, Dict, List, Tuple

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from transformers import pipeline, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

# -----------------------------
# PATHS (preserve from v6)
# -----------------------------
DB_PATH = "./data/financial_model.sqlite"                      # unchanged
SCHEMA_STORE_DIR = "./vector_stores/schema_store"          # unchanged
OV_MODEL_PATH = "./models/qwen"                               # unchanged
TOKENIZER_PATH = "./models/qwen_tokenizer"                    # unchanged

# FAQ store path in your v8 file (keep as-is) + a fallback to a common alternate folder.
FAQ_STORE_DIR = "./vector_stores/faq_store"
# FAQ_STORE_FALLBACK = "./vector_stores/faq_store"

# -----------------------------
# Prompts
# -----------------------------
SQL_PROMPT = """You are an expert SQLite SQL assistant. Generate ONE valid SQLite SELECT query that answers the user's question.
Schema context:
{context}

HARD RULES:
- Output ONLY raw SQL. No explanations. No markdown.
- Output exactly ONE statement and ALWAYS end it with a semicolon ';'.
- Only write SELECT queries (WITH ... SELECT allowed).
- Use ONLY tables and columns that appear in the schema context. Never invent tables/columns.
- Do NOT include any non-SQL tokens (e.g., 'TABLE GUIDE', 'SHEET GUIDE') in identifiers.
- NEVER use LIMIT 0. If you use LIMIT, it must be >= 1.

DATABASE CONTRACT:
- Primary numeric table: facts(metric, period, period_type, value, sheet, unit).
- glossary has ONLY (term, definition). Use glossary ONLY for definition questions.

PERIOD RULES:
- If question mentions FYxx, use period_type='FY' AND period='FYxx'.
- If question asks time series, use period_type='DATE' and ORDER BY period.

BOOLEAN LOGIC RULE:
- NEVER use OR without parentheses.

OUTPUT SHAPE:
- Prefer selecting metric, period, sheet, value, unit so results can be disambiguated.

User question:
{question}

SQL Query:
"""

FINAL_PROMPT = """You are an investment/BD analyst summarizing the company's fundraising materials.

You have:
- FACTS: normalized numeric facts from SQL (authoritative).
- FAQ: qualitative excerpts.

HARD RULES:
- Output a concise answer with maximum 5 sentences. No explanations about what you did
- You MAY paraphrase, but ONLY using information contained in FACTS and FAQ_EXCERPTS.
- If FACTS != NO_SQL_RESULTS, you MUST - include the verbatim/ paraphrased FACT line(s) in your answer
- Include ONE (top) short direct quote from FAQ_EXCERPTS to anchor the paraphrase.
- Never invent numbers or units.
- You may add at most 20 non-source words total, only as glue to connect quoted fragments.
- Do NOT add any new facts, numbers, or interpretations.
- Do NOT include disclaimers ("contact", "refer to", "verify", etc.).
- If you cannot find an answer in the sources, output exactly: "I couldn't find this in the FAQ."
- Never include placeholders like “[insert …]” or “<…>”.
- Never say “contact support”, “refer to statements”, or similar.

Question:
{question}

FACTS:
{facts}

FAQ:
{faq}

Answer (max 5 sentences):
"""

# -----------------------------
# LLM (deterministic best-effort)
# -----------------------------

def build_llm() -> HuggingFacePipeline:
    ov_model = OVModelForCausalLM.from_pretrained(OV_MODEL_PATH, device="GPU")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Best-effort determinism (some OpenVINO backends ignore temperature/top_p).
    try:
        ov_model.generation_config.do_sample = False
        ov_model.generation_config.num_beams = 1
    except Exception:
        pass

    gen = pipeline(
        task="text-generation",
        model=ov_model,
        tokenizer=tokenizer,
        max_new_tokens=450,
        do_sample=False,
        num_beams=1,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen)


# -----------------------------
# Chroma retrievers
# -----------------------------

def make_retriever(persist_dir: str, k: int, fetch_k: int) -> Any:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    print("[FAQ] store:", persist_dir, "count:", store._collection.count())
    return store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k})


def init_sql_chain(llm_used: HuggingFacePipeline) -> RetrievalQA:
    retriever = make_retriever(SCHEMA_STORE_DIR, k=4, fetch_k=20)
    pt = PromptTemplate(input_variables=["context", "question"], template=SQL_PROMPT.strip())
    return RetrievalQA.from_chain_type(
        llm_used,
        chain_type="stuff",
        chain_type_kwargs={"prompt": pt},
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
    )


def init_faq_retrievers() -> List[Any]:
    retrievers = []
    for d in [FAQ_STORE_DIR]:
        try:
            retrievers.append(make_retriever(d, k=2, fetch_k=30))
        except Exception:
            pass
    return retrievers


# -----------------------------
# SQL cleaning + guardrails
# -----------------------------

def clean_sql(llm_text: str) -> str:
    if not llm_text or not isinstance(llm_text, str):
        raise ValueError("Empty or invalid LLM output")

    text = llm_text.strip()
    text = re.sub(r"```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    text = re.sub(r"^\s*sql\s*query\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    m = re.search(r"(?is)\b(with|select)\b", text)
    if not m:
        raise ValueError("No SELECT/WITH found")

    sql = text[m.start():].strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip() + ";"
    else:
        if not sql.endswith(";"):
            sql += ";"

    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "attach", "pragma"]
    low = sql.lower()
    if any(re.search(rf"\b{kw}\b", low) for kw in forbidden):
        raise ValueError("Unsafe SQL detected")

    return sql


def enforce_or_parentheses(sql: str) -> str:
    low = sql.lower()
    if " where " not in low or " or " not in low:
        return sql

    m = re.search(r"(?is)\bwhere\b(.*?)(\border\s+by\b|\blimit\b|;)", sql)
    if not m:
        return sql

    where_body = m.group(1)
    # If already parenthesized somewhere, assume okay
    if "(" in where_body and ")" in where_body:
        return sql

    # Wrap whole WHERE body (safe default to avoid AND/OR precedence bugs)
    new_sql = re.sub(
        r"(?is)\bwhere\b.*?(\border\s+by\b|\blimit\b|;)",
        "WHERE (" + where_body.strip() + r") \1",
        sql,
        count=1,
    )
    return new_sql


def enforce_select_disambiguation(sql: str) -> str:
    """If query hits facts, rewrite SELECT list to include metric/period/sheet/value/unit."""
    low = sql.lower()
    if "from facts" not in low:
        return sql

    # Replace first SELECT ... FROM facts
    m = re.search(r"(?is)^\s*(with\b.*?\)\s*)?select\s+.*?\s+from\s+facts\b", sql)
    if not m:
        return sql

    prefix = m.group(1) or ""
    rewritten = re.sub(
        r"(?is)^\s*(with\b.*?\)\s*)?select\s+.*?\s+from\s+facts\b",
        prefix + "SELECT metric, period, sheet, value, unit\nFROM facts",
        sql,
        count=1,
    )
    return rewritten


# -----------------------------
# Execute SQL
# -----------------------------

def execute_sql(db_path: str, sql: str) -> Tuple[List[str], List[Tuple]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    conn.close()
    return cols, rows


# -----------------------------
# Fact extraction & relevance filtering
# -----------------------------

KEY_METRIC_TERMS = [
    "ebitda", "net profit", "profit", "pat", "revenue", "arr", "aum", "gmv", "cash", "opex", "cogs",
    "gross margin", "margin", "earnings", "capex", "burn", "runway"
]
FY_RE = re.compile(r"\bFY\d{2}\b", re.I)


def extract_keywords(question: str) -> List[str]:
    ql = question.lower()
    kws = []

    kws.extend([m.group(0).upper() for m in FY_RE.finditer(question)])
    for t in KEY_METRIC_TERMS:
        if t in ql:
            kws.append(t)

    out = []
    seen = set()
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def rows_to_facts(cols: List[str], rows: List[Tuple], limit: int = 25) -> List[str]:
    if not rows:
        return []

    uniq = []
    seen = set()
    for r in rows:
        if r not in seen:
            seen.add(r)
            uniq.append(r)

    facts = []
    for r in uniq[:limit]:
        d = {cols[i]: r[i] for i in range(min(len(cols), len(r)))} if cols else {}
        metric = d.get("metric")
        period = d.get("period")
        sheet = d.get("sheet")
        value = d.get("value")
        unit = d.get("unit")

        if value is None:
            continue

        label_parts = [p for p in [metric, period, sheet] if p]
        label = " | ".join([str(x) for x in label_parts]) if label_parts else "value"

        if unit is None:
            facts.append(f"{label}: {value} (unit not specified)")
        else:
            u = str(unit).strip()
            if u == "%":
                facts.append(f"{label}: {value}{u}")
            else:
                facts.append(f"{label}: {value} {u}")

    return facts


def filter_relevant_facts(facts: List[str], question: str) -> List[str]:
    if not facts:
        return []

    kws = extract_keywords(question)
    if not kws:
        return facts[:5]

    fy_in_q = [k for k in kws if k.startswith("FY")]
    metric_kws = [k for k in kws if not k.startswith("FY")]

    def ok(line: str) -> bool:
        ll = line.lower()
        # FY constraint
        if fy_in_q:
            if any(fy.lower() in ll for fy in fy_in_q):
                pass
            else:
                if re.search(r"\bfy\d{2}\b", ll):
                    return False
        # metric constraint
        if metric_kws:
            return any(mk in ll for mk in metric_kws)
        return True

    picked = [f for f in facts if ok(f)]
    return picked if picked else facts[:5]


# -----------------------------
# FAQ extraction (deterministic)
# -----------------------------

def faq_excerpts_from_docs(docs: List[Any], max_chars: int = 1800) -> str:
    if not docs:
        return ""

    # keep top 2-3, de-dup by exact content
    parts = []
    seen = set()
    for d in docs[:4]:
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        txt = re.sub(r"\s+", " ", txt)
        if txt in seen:
            continue
        seen.add(txt)

        meta = getattr(d, "metadata", {}) or {}
        q = meta.get("question")
        sec = meta.get("section")
        prefix = ""
        if sec:
            prefix += f"[{sec}] "
        if q:
            prefix += f"{q}: "
        parts.append((prefix + txt).strip())

    out = "\n\n".join(parts)
    return out[:max_chars]


# -----------------------------
# Main orchestration (always query both)
# -----------------------------
def _is_percent(metric: str, unit: str) -> bool:
    m = (metric or "").lower()
    u = (unit or "").strip()
    return u == "%" or "margin" in m or "rate" in m or "ratio" in m or "%" in m

def _normalize_percent(value, metric: str, unit: str):
    if value is None:
        return None, unit
    v = float(value)
    if _is_percent(metric, unit):
        # treat fractions as percent
        if 0 <= v <= 1.5:
            v *= 100.0
        return v, "%"
    return v, unit

def _to_inr_m(value, unit):
    if value is None:
        return None, "INR M"
    v = float(value)
    u = (unit or "").strip().lower()

    # common conversions; extend as needed
    if u in ["inr cr", "inr crore", "cr"]:
        return v * 10.0, "INR M"     # 1 Cr = 10 M
    if u in ["inr mn", "inr m", "mn", "m", "inr million"]:
        return v, "INR M"

    # if missing or unknown, enforce display in INR M (your preference)
    return v, "INR M"

def normalize_sql_rows(cols, rows, max_rows=30):
    """
    Returns a cleaned list of dict rows with normalized values/units.
    Assumes cols includes metric, period, sheet, value, unit (your prompt + disambiguation).
    """
    if not rows:
        return []

    # dedupe exact tuples
    uniq = []
    seen = set()
    for r in rows:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    uniq = uniq[:max_rows]

    cleaned = []
    for r in uniq:
        d = {cols[i]: r[i] for i in range(min(len(cols), len(r)))} if cols else {}
        metric = d.get("metric") or ""
        period = d.get("period") or ""
        sheet  = d.get("sheet") or ""
        value  = d.get("value")
        unit   = d.get("unit")

        # normalize percent rows
        v1, u1 = _normalize_percent(value, metric, unit)

        # normalize non-percent to INR M
        if not _is_percent(metric, u1):
            v2, u2 = _to_inr_m(v1, u1)
        else:
            v2, u2 = v1, u1

        cleaned.append({
            "metric": metric,
            "period": period,
            "sheet": sheet,
            "value": v2,
            "unit": u2
        })

    return cleaned

def facts_block(cleaned_rows, limit=12):
    if not cleaned_rows:
        return "NO_SQL_RESULTS"
    lines = []
    for d in cleaned_rows[:limit]:
        metric = d["metric"]
        period = d["period"]
        sheet  = d["sheet"]
        value  = d["value"]
        unit   = d["unit"]

        if value is None:
            continue

        label = " | ".join([x for x in [metric, period, sheet] if x])
        if unit == "%":
            lines.append(f"{label}: {value:.2f}%")
        else:
            lines.append(f"{label}: {value:.2f} {unit}")
    return "\n".join(lines) if lines else "NO_SQL_RESULTS"
def answer_query(user_q: str, llm_used: HuggingFacePipeline, qa_sql: RetrievalQA, faq_retrievers: List[Any]) -> Dict[str, Any]:
    # --- SQL path
    sql_generated = ""
    executed_sql = ""
    sql_cols: List[str] = []
    sql_rows: List[Tuple] = []

    try:
        sql_out = qa_sql.invoke({"query": user_q})
        sql_generated = (sql_out.get("result") or "").strip()
        sql = clean_sql(sql_generated)
        sql = enforce_or_parentheses(sql)
        sql = enforce_select_disambiguation(sql)
        executed_sql = sql
        sql_cols, sql_rows = execute_sql(DB_PATH, sql)
    except Exception as e:
        executed_sql = f"SQL_ERROR: {type(e).__name__}: {e}"
        sql_cols, sql_rows = [], []

    cleaned = normalize_sql_rows(sql_cols, sql_rows)
    facts_text = facts_block(cleaned)

    # all_facts = rows_to_facts(sql_cols, sql_rows)
    # rel_facts = filter_relevant_facts(all_facts, user_q)
    # # facts_text = "\n".join(rel_facts) if rel_facts else "NO_SQL_RESULTS"

    # --- FAQ path (query all available stores and merge)
    all_docs = []
    for r in faq_retrievers:

        try:
            # LangChain newer interface
            if hasattr(r, "invoke"):
                docs = r.invoke(user_q)
            else:
                docs = r.get_relevant_documents(user_q)
            all_docs.extend(docs or [])
        except Exception as e:
            print("[FAQ RETRIEVAL ERROR]", type(e).__name__, str(e))
    faq_excerpt = faq_excerpts_from_docs(all_docs)
    # print("FAQ documents pulled: ", faq_excerpt)

    # --- Final answer
    final_pt = PromptTemplate(input_variables=["question", "facts", "faq"], template=FINAL_PROMPT)
    final_prompt = final_pt.format(question=user_q, facts=facts_text, faq=faq_excerpt if faq_excerpt else "(none)")
    final_answer = llm_used.invoke(final_prompt).strip()

    # Guard against placeholders
    if re.search(r"\[insert .*?\]", final_answer, flags=re.I) or re.search(r"<.*?>", final_answer):
        # deterministic fallback
        if facts_text != "NO_SQL_RESULTS" and faq_excerpt:
            final_answer = f"{facts_text}\n\n{faq_excerpt}".strip()
        elif facts_text != "NO_SQL_RESULTS":
            final_answer = facts_text
        elif faq_excerpt:
            final_answer = faq_excerpt
        else:
            final_answer = "Sorry — I could not find an answer in the database or FAQ."

    return {
        "sql_generated": sql_generated,
        "executed_sql": executed_sql,
        "sql_rows": sql_rows,
        "facts_text": facts_text,
        "faq_excerpt": faq_excerpt,
        "final_answer": final_answer,
    }


def main():
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    warnings.filterwarnings("ignore")

    llm_used = build_llm()
    qa_sql = init_sql_chain(llm_used)
    faq_retrievers = init_faq_retrievers()

    if not faq_retrievers:
        print("(Warning) Could not load FAQ vector store from either:")
        print(" -", FAQ_STORE_DIR)
        # print(" -", FAQ_STORE_FALLBACK)
        print("Proceeding with SQL-only behavior.\n")

    while True:
        q = input("Enter the query for financial projections: ")
        if q.strip().lower() == "x":
            break

        out = answer_query(q, llm_used, qa_sql, faq_retrievers)

        # Debug prints
        print("\nExecuted SQL:\n", out["executed_sql"])
        print("SQL rows:", out["sql_rows"])
        print("\nFACTS:\n", out["facts_text"])
        if out["faq_excerpt"]:
            print("\nFAQ EXCERPTS (top hits):\n", out["faq_excerpt"])
        print("\nFinal answer:\n", out["final_answer"], "\n")


if __name__ == "__main__":
    main()
