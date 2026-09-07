# AI Engineering — the two builds

Two small Streamlit apps that follow the workshop deck (`python-to-production-ai.excalidraw`).
Same model in both. The second one just hands it the right page first.

| Section | What it is | Run it |
| --- | --- | --- |
| [01-llm-basics/](01-llm-basics/) | One LLM call, taken apart: system prompt, material, question, two dials — plus a weak-vs-sharp prompt bake-off | `streamlit run 01-llm-basics/app.py` |
| [02-rag/](02-rag/) | Upload a PDF, ask it anything, get the answer back with the page number | `streamlit run 02-rag/app.py` |

## Setup

```bash
cd ai-engineering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then paste your GROQ_API_KEY
```

Get a key at [console.groq.com](https://console.groq.com). Nothing else needs an account —
Chroma runs on your laptop and embeds locally, so the PDF never leaves your machine.

## 01 — Basic LLM calls with prompting

`streamlit run 01-llm-basics/app.py`

- **Playground** — pick a prompt from the library, paste material, ask a question. It shows the exact
  messages going over the wire, streams the answer, and counts the tokens each way.
- **Weak vs sharp** — the same material and question through `"Summarize this."` and through a prompt
  with a role, one job, a shape and a way out. Run it three times: the sharp one keeps its shape.

The prompts live in [prompts.py](01-llm-basics/prompts.py) — in the repo, like code, which is the point.
The call itself is [llm.py](01-llm-basics/llm.py), about forty lines.

## 02 — RAG

`streamlit run 02-rag/app.py`

[rag.py](02-rag/rag.py) is the six steps, one function each:

```
LOAD → CHUNK → EMBED + STORE → FIND → ANSWER
```

Chunking keeps the page number, so every answer can be checked. The app shows the retrieved chunks
above the answer on purpose — nine times out of ten a bad answer is a finding problem, not a model
problem, and you can see that here before you touch the wording.

**When it breaks, in this order:**

1. Look at one chunk. Is it readable text, or is your PDF a scan? (The app says so if no text came out.)
2. Look at what came back. Is the answer even in those pieces? Usually it isn't — move the sliders.
3. Only now, the wording.

**Break it deliberately:** ask something the PDF never mentions; drop the chunk size to 200; fetch
twenty pages instead of four. Each failure has a name in production.

## Notes

- Model: `GROQ_MODEL`, default `qwen/qwen3-32b`. Any Groq chat model works.
- Embeddings: Chroma's built-in local model. First index downloads it once, then works offline.
- The vector store is a folder (`CHROMA_PATH`, default `./chroma_data`). Delete it to start over.
