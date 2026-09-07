"""Section 02 — RAG: ask a PDF questions, get the page back with the answer.

Run from the ai-engineering folder:
    streamlit run 02-rag/app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from rag import (  # noqa: E402
    SYSTEM_PROMPT,
    answer,
    build_store,
    chunk_pages,
    collection_name,
    find,
    load_pdf,
    open_store,
)

st.set_page_config(page_title="02 · RAG over a PDF", page_icon="📄", layout="wide")

settings = get_settings()

st.title("Ask a PDF questions")
st.caption("Same model as section 01. The only difference is what it was holding when you asked.")

if not settings.groq_api_key:
    st.error("GROQ_API_KEY is not set. Copy `.env.example` to `.env`, add your key, and rerun.")

with st.sidebar:
    st.subheader("The two dials that matter")
    chunk_size = st.slider("Chunk size", 200, 2000, settings.chunk_size, 100, help="Too small and meaning gets sliced in half.")
    overlap = st.slider("Overlap", 0, 400, settings.chunk_overlap, 50, help="A little, so an answer on a boundary survives.")
    top_k = st.slider("Pages fetched", 1, 20, settings.top_k, 1, help="Too few and it misses. Too many and the answer drowns.")
    st.divider()
    st.caption(f"Model: `{settings.groq_model}`  ·  Store: `{settings.chroma_path}`")

uploaded = st.file_uploader("Your PDF", type="pdf")

if uploaded is None:
    st.info("Upload a PDF to index it. Indexing happens once; questions are cheap after that.")
    st.stop()

name = collection_name(Path(uploaded.name))

if st.button("Index this PDF", type="primary") or st.session_state.get("indexed") != name:
    with st.spinner("Reading, cutting up, and turning into numbers…"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            pdf_path = Path(tmp.name)

        pages = load_pdf(pdf_path)
        pdf_path.unlink(missing_ok=True)

        if not pages:
            st.error(
                "No text came out of this PDF — it is probably a scan of a photograph. "
                "That is step one, and no amount of prompting fixes it."
            )
            st.stop()

        chunks = chunk_pages(pages, chunk_size, overlap)
        build_store(settings, name, chunks)

    st.session_state["indexed"] = name
    st.session_state["stats"] = (len(pages), len(chunks))

pages_count, chunks_count = st.session_state.get("stats", (0, 0))
a, b, c = st.columns(3)
a.metric("Pages with text", pages_count)
b.metric("Chunks stored", chunks_count)
c.metric("Fetched per question", top_k)

question = st.text_input("Your question", placeholder="What does this document say about…")

if question:
    collection = open_store(settings, name)

    with st.spinner("Finding the near pieces…"):
        found = find(collection, question, top_k)

    with st.expander(f"What came back — read this before blaming the model ({len(found)} chunks)"):
        for i, chunk in enumerate(found, start=1):
            st.markdown(f"**{i} · page {chunk.page}**")
            st.text(chunk.text[:1200])
            st.divider()

    if not settings.groq_api_key:
        st.warning("Retrieval works without a key. The answer needs one.")
        st.stop()

    with st.spinner("Writing the answer…"):
        st.markdown(answer(settings, question, found))

    st.caption("Pages used: " + ", ".join(str(p) for p in sorted({c.page for c in found})))

with st.expander("The one instruction doing most of the work"):
    st.code(SYSTEM_PROMPT, language="text")
    st.markdown(
        "**Break it on purpose:** ask something the PDF never mentions — does it admit that? "
        "Then drop chunk size to 200, or fetch twenty pages, and watch which step actually degrades."
    )
