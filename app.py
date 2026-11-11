"""app.py
Local Copilot-style chat UI server.
Run:
  python -m uvicorn app:app --reload --port 8000
Then open:
  http://localhost:8000

This server is fully local.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Body
import uuid
from typing import Dict, Any

import chat_engine

app = FastAPI(title="Fundraise Chat UI", version="0.1")

# Serve static assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory sessions (simple local)
SESSIONS: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
def _startup():
    chat_engine.init()


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/session")
def new_session():
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"messages": []}
    return {"session_id": sid}


@app.post("/api/chat")
def chat(payload: Dict[str, Any] = Body(...)):
    session_id = payload.get("session_id")
    message = (payload.get("message") or "").strip()
    if not message:
        return {"error": "Empty message"}

    if not session_id or session_id not in SESSIONS:
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {"messages": []}

    # store user message
    SESSIONS[session_id]["messages"].append({"role": "user", "content": message})

    # model answer
    out = chat_engine.answer(message)
    answer = out.get("answer", "")
    debug = out.get("debug", {})

    SESSIONS[session_id]["messages"].append({"role": "assistant", "content": answer, "debug": debug})

    return {"session_id": session_id, "answer": answer, "debug": debug}


@app.get("/api/history/{session_id}")
def history(session_id: str):
    if session_id not in SESSIONS:
        return {"session_id": session_id, "messages": []}
    return {"session_id": session_id, "messages": SESSIONS[session_id]["messages"]}


@app.post("/api/reset/{session_id}")
def reset(session_id: str):
    SESSIONS[session_id] = {"messages": []}
    return {"ok": True}
