"""chat_engine.py
Wrapper around sql_master.py to provide a stable interface for the web app.
- Loads the LLM and QA chains once at startup.
- Provides a thread-safe-ish (single-process) answer() method.

Assumptions:
- sql_master.py is in the same folder.
- Vector stores and sqlite db paths are as defined inside that script.
"""

import threading
from typing import Dict, Any

import sql_master as core

# Patch common FAQ fallback bug (v9 had fallback same as primary in some variants)
# If your FAQ store is actually somewhere else, set these before init().
try:
    if getattr(core, 'FAQ_STORE_FALLBACK', None) == getattr(core, 'FAQ_STORE_DIR', None):
        core.FAQ_STORE_FALLBACK = "./vector_stores/faq_store_alt"
except Exception:
    pass

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "llm": None,
    "qa_sql": None,
    "faq_retrievers": None,
    "ready": False,
}


def init() -> None:
    """Initialize model + chains once."""
    with _lock:
        if _state["ready"]:
            return
        llm = core.build_llm()
        qa_sql = core.init_sql_chain(llm)
        faq_retrievers = core.init_faq_retrievers()
        _state.update({"llm": llm, "qa_sql": qa_sql, "faq_retrievers": faq_retrievers, "ready": True})


def answer(message: str) -> Dict[str, Any]:
    """Run a single query through the underlying pipeline."""
    if not _state["ready"]:
        init()

    # Underlying answer_query returns dict with executed_sql, facts_text, faq_excerpt, final_answer
    with _lock:
        out = core.answer_query(message, _state["llm"], _state["qa_sql"], _state["faq_retrievers"])

    # Normalize keys for the UI
    return {
        "answer": out.get("final_answer", ""),
        "debug": {
            "executed_sql": out.get("executed_sql", ""),
            "facts": out.get("facts_text", ""),
            "faq_excerpt": out.get("faq_excerpt", ""),
            "sql_generated": out.get("sql_generated", ""),
            "sql_rows": out.get("sql_rows", []),
        }
    }
