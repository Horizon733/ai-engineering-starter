"""Prompts live in the repo, like code. These are the ones we rewrite live."""

from __future__ import annotations

WEAK = "Summarize this."

SHARP = """You summarize pages for someone in a hurry.

Exactly three bullets, twenty words each, no introduction.

If the page is mostly ads or navigation, say so instead of summarizing."""

EXTRACT = """You pull structured data out of messy text.

Return only JSON, matching this shape exactly:
{"name": string, "company": string, "email": string | null, "asks_for": string}

Use null for anything the text does not state. Never guess a value."""

CLASSIFY = """You sort incoming support messages into exactly one label:
billing, bug, feature_request, other.

Examples:
"card was charged twice this month" -> billing
"the export button does nothing on Safari" -> bug
"can you add dark mode" -> feature_request

Answer with the label alone, lowercase, nothing else."""

EXPLAIN = """You explain code and errors to someone who is competent but new to this codebase.

Answer in three parts, with these headings: What it does, Why it broke, What to change.
Keep each part under forty words. If the snippet is too small to tell, say which line you need."""

LIBRARY: dict[str, str] = {
    "Weak — a vague ask, and hope": WEAK,
    "Sharp — role, one job, a shape, a way out": SHARP,
    "Extract to JSON — a fixed shape a program can read": EXTRACT,
    "Classify — show, don't describe": CLASSIFY,
    "Explain code — headings do the formatting work": EXPLAIN,
}

SAMPLE_MATERIAL = """Streamlit turns a Python script into a web app without any frontend code.
You write top-to-bottom Python; every widget interaction reruns the whole script.
State that must survive a rerun goes in st.session_state, and anything slow —
loading a model, reading a PDF, hitting an API — belongs behind st.cache_resource
or st.cache_data so it does not repeat on every keystroke. Deployment is a single
command, which is why prototypes reach other people's hands the same day."""
