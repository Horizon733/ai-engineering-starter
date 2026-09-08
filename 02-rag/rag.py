import os
import tempfile

import chromadb
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma = chromadb.PersistentClient(path=CHROMA_PATH)

SYSTEM_PROMPT = """You answer questions using only the context below, taken from one PDF.

Use nothing else — not your own knowledge, not a guess that sounds right.
Every claim ends with the page it came from, written as (p. 12).
If the context does not contain the answer, say exactly: "That isn't in this document."
Answer in Markdown, and keep it short."""


# ---------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------

def load_pdf(path):
    pages = []

    for number, page in enumerate(PdfReader(path).pages, start=1):
        text = (page.extract_text() or "").strip()

        if text:
            pages.append((number, text))

    return pages


# ---------------------------------------------------------
# 2. CHUNK
# ---------------------------------------------------------

def chunk_pages(pages, chunk_size=1000, overlap=200):
    chunks = []

    step = chunk_size - overlap

    for page_number, text in pages:
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_size].strip()

            if piece:
                chunks.append((page_number, piece))

    return chunks


# ---------------------------------------------------------
# 3. EMBED + STORE
# ---------------------------------------------------------

def build_store(name, chunks):
    try:
        chroma.delete_collection(name)
    except Exception:
        pass

    collection = chroma.create_collection(name)

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=[text for _, text in chunks],
        metadatas=[{"page": page} for page, _ in chunks],
    )

    return collection


# ---------------------------------------------------------
# 4. FIND
# ---------------------------------------------------------

def find(name, question, top_k=4):
    collection = chroma.get_collection(name)

    result = collection.query(
        query_texts=[question],
        n_results=top_k,
    )

    return list(
        zip(
            [m["page"] for m in result["metadatas"][0]],
            result["documents"][0],
        )
    )


# ---------------------------------------------------------
# 5. ANSWER
# ---------------------------------------------------------

def answer(question, chunks):
    context = "\n\n".join(
        f"[page {page}]\n{text}"
        for page, text in chunks
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n"
                    f"---\n"
                    f"{context}\n"
                    f"---\n\n"
                    f"Question: {question}"
                ),
            },
        ],
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="Simple RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Simple PDF RAG")

st.caption(
    "LOAD → CHUNK → EMBED + STORE → FIND → ANSWER"
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ RAG Settings")

    chunk_size = st.slider(
        "Chunk size",
        min_value=300,
        max_value=3000,
        value=1000,
        step=100,
    )

    overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=800,
        value=200,
        step=50,
    )

    top_k = st.slider(
        "Retrieved chunks (Top K)",
        min_value=1,
        max_value=10,
        value=4,
    )

    st.divider()

    st.caption(f"Model: `{MODEL}`")
    st.caption(f"Chroma: `{CHROMA_PATH}`")


# ---------------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------------

st.subheader("1️⃣ Upload your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
)

if uploaded_file:

    st.success(
        f"Loaded: **{uploaded_file.name}**"
    )

    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as tmp:

        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name

    # Collection name based on session
    collection_name = "pdf_rag"

    if st.button(
        "🔨 Index PDF",
        type="primary",
        use_container_width=True,
    ):

        with st.status(
            "Building RAG index...",
            expanded=True,
        ) as status:

            # LOAD
            st.write("📄 Loading PDF...")

            pages = load_pdf(pdf_path)

            st.write(
                f"Found **{len(pages)} pages**."
            )

            # CHUNK
            st.write("✂️ Creating chunks...")

            chunks = chunk_pages(
                pages,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            st.write(
                f"Created **{len(chunks)} chunks**."
            )

            # EMBED + STORE
            st.write("🧠 Embedding + storing...")

            build_store(
                collection_name,
                chunks,
            )

            # Save state
            st.session_state["indexed"] = True
            st.session_state["pdf_name"] = uploaded_file.name
            st.session_state["pages"] = pages
            st.session_state["chunks"] = chunks
            st.session_state["collection_name"] = collection_name

            status.update(
                label="RAG index ready!",
                state="complete",
            )


# ---------------------------------------------------------
# INDEX INFO
# ---------------------------------------------------------

if st.session_state.get("indexed"):

    st.divider()

    st.subheader("📊 Index")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Pages",
            len(st.session_state["pages"]),
        )

    with col2:
        st.metric(
            "Chunks",
            len(st.session_state["chunks"]),
        )

    with col3:
        st.metric(
            "Chunk size",
            chunk_size,
        )

    # -----------------------------------------------------
    # CHUNK INSPECTOR
    # -----------------------------------------------------

    with st.expander("🔍 Inspect chunks"):

        chunks = st.session_state["chunks"]

        for i, (page, text) in enumerate(chunks[:20]):

            st.markdown(
                f"**Chunk {i + 1} — Page {page}**"
            )

            st.code(
                text,
                language="text",
            )

            st.divider()


# ---------------------------------------------------------
# QUESTION ANSWERING
# ---------------------------------------------------------

if st.session_state.get("indexed"):

    st.divider()

    st.subheader("2️⃣ Ask questions")

    question = st.text_input(
        "Ask something about your PDF",
        placeholder="What is this document about?",
    )

    if st.button(
        "🚀 Ask",
        type="primary",
        disabled=not question,
    ):

        with st.spinner("Searching the document..."):

            # FIND
            retrieved_chunks = find(
                st.session_state["collection_name"],
                question,
                top_k=top_k,
            )

        # -------------------------------------------------
        # RETRIEVAL RESULTS
        # -------------------------------------------------

        st.subheader("🔎 Retrieved Context")

        for i, (page, text) in enumerate(
            retrieved_chunks,
            start=1,
        ):

            with st.expander(
                f"Chunk {i} — Page {page}",
                expanded=False,
            ):

                st.write(text)

        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------

        with st.spinner("Generating answer..."):

            response = answer(
                question,
                retrieved_chunks,
            )

        st.subheader("💬 Answer")

        st.markdown(response)


# ---------------------------------------------------------
# EMPTY STATE
# ---------------------------------------------------------

else:

    st.info(
        "Upload a PDF and click **Index PDF** to start."
    )

    st.markdown(
        """
### How this RAG works

```text
PDF
 │
 ▼
LOAD
 │
 ▼
CHUNK
 │
 ▼
EMBED + STORE
 │
 ▼
FIND
 │
 ▼
ANSWER
````

The important knobs are:

* **Chunk size** → how much text goes into each chunk
* **Overlap** → how much neighboring chunks share
* **Top K** → how many chunks are retrieved
  """
  )
