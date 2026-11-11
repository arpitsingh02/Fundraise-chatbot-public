"""
faq_chroma_embedding_generation.py

Build a Chroma vector store from a Word FAQ document (DOCX) with Q/A-aware chunking.

Why this works well for FAQs:
- Each question becomes an "atomic" document: Q + (Short Answer) + (Detailed Answer)
- Avoids mixing multiple questions in one chunk (reduces semantic confusion)
- Deduplicates repeated questions
- Keeps answers intact unless they exceed token budget, then splits safely
- Adds metadata: section, intent, qid, answer_type, source

Matches the structure found in company_faq.docx (Q ... Short Answer ... Detailed Answer ...).
"""

import os
import re
import argparse
import hashlib
from typing import List, Optional, Dict, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader

from transformers import AutoTokenizer
try:
    # You used this path in other scripts; keep compatibility
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore


# -----------------------------
# Defaults (tune as needed)
# -----------------------------
DEFAULT_PERSIST_DIR = "./vector_stores/faq_store"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# IMPORTANT: use a local tokenizer path (you already used "./models/qwen_tokenizer" elsewhere)
DEFAULT_TOKENIZER_PATH = "./models/qwen_tokenizer"

# Token-aware chunking: keep Q/A together; split only if very long
CHUNK_SIZE_TOKENS = 900
CHUNK_OVERLAP_TOKENS = 120

# Store variants (combined is enough for most cases; short/detailed variants help targeted retrieval)
EMIT_SHORT_VARIANTS = True
EMIT_DETAILED_VARIANTS = True

# -----------------------------
# Cleaning helpers
# -----------------------------
def normalize_ws(text: str) -> str:
    text = text.replace("\u00A0", " ")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def strip_markdown_bold(text: str) -> str:
    # Your DOCX content includes **Q ...** patterns (bold markers may appear after extraction)
    return text.replace("**", "").strip()

def stable_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def normalize_question_key(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def detect_intent(question: str) -> str:
    q = question.lower().strip()
    if q.startswith("why"):
        return "why"
    if q.startswith("how"):
        return "how"
    if q.startswith("what"):
        return "what"
    if q.startswith("when"):
        return "when"
    if q.startswith("who"):
        return "who"
    if q.startswith("where"):
        return "where"
    return "other"


# -----------------------------
# Parsing FAQ structure
# -----------------------------
# Q lines in your doc look like:
# "Q What does the company do?" or "**Q: ...**" etc.
Q_LINE = re.compile(r"(?i)^\s*\*{0,2}\s*Q\s*:?\s*(.+?)\s*\*{0,2}\s*$")

SHORT_HDR = re.compile(r"(?i)\bshort answer\b\s*:?\s*")
DETAILED_HDR = re.compile(r"(?i)\bdetailed answer\b\s*:?\s*")

# Section headers (your doc includes uppercase-ish separators like "TEAM & FUNDING")
SECTION_HDR = re.compile(r"^[A-Z][A-Z &/+\-]{6,}$")


def split_short_detailed(answer_text: str) -> Tuple[str, str]:
    """
    Extract (short, detailed) from an answer blob.
    If markers are missing, returns ("", full_text) as detailed.
    """
    txt = normalize_ws(strip_markdown_bold(answer_text))
    m_short = SHORT_HDR.search(txt)
    m_det = DETAILED_HDR.search(txt)

    if m_short and m_det:
        if m_short.start() < m_det.start():
            short = normalize_ws(txt[m_short.end():m_det.start()])
            detailed = normalize_ws(txt[m_det.end():])
            return short, detailed
        else:
            detailed = normalize_ws(txt[m_det.end():m_short.start()])
            short = normalize_ws(txt[m_short.end():])
            return short, detailed

    if m_short and not m_det:
        short = normalize_ws(txt[m_short.end():])
        return short, ""

    if m_det and not m_short:
        detailed = normalize_ws(txt[m_det.end():])
        return "", detailed

    # Neither marker exists: treat as detailed
    return "", txt


def parse_faq(text: str, source_name: str) -> List[Document]:
    """
    Convert the full DOCX text into Q/A atomic Documents with metadata.
    Also captures section headers and assigns section to each Q.
    Deduplicates repeated questions.
    """
    lines = [strip_markdown_bold(l) for l in text.splitlines()]
    lines = [l.rstrip() for l in lines]

    current_section = "GENERAL"
    current_q: Optional[str] = None
    buf: List[str] = []

    blocks: List[Tuple[str, str, str]] = []  # (section, question, answer_text)

    def flush():
        nonlocal current_q, buf
        if current_q is not None:
            ans = normalize_ws("\n".join(buf))
            blocks.append((current_section, current_q, ans))
        current_q = None
        buf = []

    for line in lines:
        l = line.strip()
        if not l:
            continue

        # Detect section header
        if SECTION_HDR.match(l) and "Q " not in l.upper():
            # new section begins
            flush()
            current_section = l
            continue

        m = Q_LINE.match(l)
        if m:
            flush()
            current_q = normalize_ws(m.group(1))
            continue

        if current_q is not None:
            buf.append(l)

    flush()

    out: List[Document] = []
    seen_q = set()

    for section, question, answer_blob in blocks:
        q_key = normalize_question_key(question)
        if q_key in seen_q:
            continue
        seen_q.add(q_key)

        short, detailed = split_short_detailed(answer_blob)
        intent = detect_intent(question)
        qid = stable_hash(q_key)

        # Combined doc: best default retrieval
        parts = [f"Q: {question}"]
        if short:
            parts.append(f"Short Answer: {short}")
        if detailed:
            parts.append(f"Detailed Answer: {detailed}")
        if not short and not detailed:
            # edge-case: empty answer
            parts.append("Answer: (not provided in document)")
        combined = "\n".join(parts).strip()

        base_meta = {
            "doc_type": "faq",
            "source": source_name,
            "section": section,
            "question": question,
            "qid": qid,
            "intent": intent,
        }

        out.append(Document(page_content=combined, metadata={**base_meta, "answer_type": "combined"}))

        # Optional variants
        if EMIT_SHORT_VARIANTS and short:
            out.append(Document(
                page_content=f"Q: {question}\nShort Answer: {short}",
                metadata={**base_meta, "answer_type": "short"},
            ))
        if EMIT_DETAILED_VARIANTS and detailed:
            out.append(Document(
                page_content=f"Q: {question}\nDetailed Answer: {detailed}",
                metadata={**base_meta, "answer_type": "detailed"},
            ))

    return out


# -----------------------------
# Token-aware splitting (only if needed)
# -----------------------------
def split_token_aware(docs: List[Document], tokenizer_path: str) -> List[Document]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    out: List[Document] = []
    for d in docs:
        tok_len = len(tokenizer.encode(d.page_content))
        if tok_len <= CHUNK_SIZE_TOKENS:
            out.append(d)
        else:
            parts = splitter.split_text(d.page_content)
            for i, p in enumerate(parts):
                md = dict(d.metadata)
                md["chunk_index"] = i
                md["chunk_total"] = len(parts)
                out.append(Document(page_content=p, metadata=md))
    return out


# -----------------------------
# Build Chroma store
# -----------------------------
def build_store(docs: List[Document], persist_dir: str, embed_model: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    store = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
    try:
        store.persist()
    except Exception:
        pass
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True, help="Path to FAQ DOCX file")
    ap.add_argument("--persist", default=DEFAULT_PERSIST_DIR, help="Chroma persist directory")
    ap.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL, help="Embedding model name")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_PATH, help="HF tokenizer path (local)")
    args = ap.parse_args()

    docx_path = args.docx
    source_name = os.path.basename(docx_path)

    loader = Docx2txtLoader(docx_path)
    raw_docs = loader.load()
    if not raw_docs:
        raise RuntimeError(f"No content loaded from: {docx_path}")

    full_text = normalize_ws(raw_docs[0].page_content)
    parsed = parse_faq(full_text, source_name=source_name)
    chunked = split_token_aware(parsed, tokenizer_path=args.tokenizer)

    print(f"Loaded: {source_name}")
    print(f"Parsed FAQ docs (pre-split): {len(parsed)}")
    print(f"Final chunks to embed: {len(chunked)}")
    print(f"Persist dir: {args.persist}")
    print(f"Embedding model: {args.embed_model}")

    build_store(chunked, persist_dir=args.persist, embed_model=args.embed_model)
    print("Done.")


if __name__ == "__main__":
    main()