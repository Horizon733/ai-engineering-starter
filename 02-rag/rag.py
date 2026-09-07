"""The six steps, one function each.

LOAD -> CHUNK -> EMBED+STORE -> FIND -> ANSWER
Two of them decide everything: how you cut it up, and what comes back.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Settings  # noqa: E402

SYSTEM_PROMPT = """You answer questions using only the context below, taken from one PDF.

Use nothing else — not your own knowledge, not a guess that sounds right.
Every claim ends with the page it came from, written as (p. 12).
If the context does not contain the answer, say exactly: "That isn't in this document."
Answer in Markdown, and keep it short."""


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int


# 1. LOAD — pages to text, page numbers kept, because that is the citation later.
def load_pdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


# 2. CHUNK — split on paragraphs, overlap a little, never lose the page.
def chunk_pages(pages: list[tuple[int, str]], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []

    for page_number, text in pages:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buffer = ""

        for paragraph in paragraphs:
            if len(buffer) + len(paragraph) + 2 <= chunk_size:
                buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
                continue
            tail = ""
            if buffer:
                chunks.append(Chunk(buffer, page_number))
                tail = buffer[-overlap:] if overlap else ""
                buffer = ""

            if len(paragraph) > chunk_size:
                # One paragraph longer than a whole chunk: cut it straight, and accept the loss.
                step = max(1, chunk_size - overlap)
                for start in range(0, len(paragraph), step):
                    piece = paragraph[start : start + chunk_size]
                    if piece.strip():
                        chunks.append(Chunk(piece, page_number))
            else:
                buffer = f"{tail}\n\n{paragraph}" if tail else paragraph

        if buffer:
            chunks.append(Chunk(buffer, page_number))

    return chunks


def collection_name(path: Path) -> str:
    digest = hashlib.sha1(path.name.encode()).hexdigest()[:8]
    return f"pdf_{digest}"


# 3. EMBED + STORE — Chroma turns each chunk into numbers and remembers them.
def build_store(settings: Settings, name: str, chunks: list[Chunk]):
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    # chroma returns names on newer versions, collection objects on older ones
    existing = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    if name in existing:
        client.delete_collection(name)  # so a re-index starts clean
    collection = client.create_collection(name)

    collection.add(
        ids=[f"{name}-{i}" for i in range(len(chunks))],
        documents=[c.text for c in chunks],
        metadatas=[{"page": c.page} for c in chunks],
    )
    return collection


def open_store(settings: Settings, name: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    return client.get_collection(name)


# 4. FIND — your question goes in, the nearest pieces come back. Look here first when it's wrong.
def find(collection, question: str, top_k: int) -> list[Chunk]:
    result = collection.query(query_texts=[question], n_results=top_k)
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    return [Chunk(doc, int(meta["page"])) for doc, meta in zip(documents, metadatas)]


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[page {c.page}]\n{c.text}" for c in chunks)


# 5. ANSWER — one instruction does more work here than any framework.
def answer(settings: Settings, question: str, chunks: list[Chunk]) -> str:
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is missing. Copy .env.example to .env and fill it in.")

    from groq import Groq

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n---\n{format_context(chunks)}\n---\n\nQuestion: {question}",
        },
    ]

    response = Groq(api_key=settings.groq_api_key).chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content or ""
