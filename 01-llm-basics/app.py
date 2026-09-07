"""Section 01 — Basic LLM calls with prompting.

Run from the ai-engineering folder:
    streamlit run 01-llm-basics/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from llm import build_messages, complete, estimate_tokens, stream  # noqa: E402
from prompts import LIBRARY, SAMPLE_MATERIAL, SHARP, WEAK  # noqa: E402

st.set_page_config(page_title="01 · LLM calls & prompting", page_icon="💬", layout="wide")

settings = get_settings()

st.title("Basic LLM calls, and the prompt that decides the answer")
st.caption("One call. A system prompt, your material, your question, two dials.")

if not settings.groq_api_key:
    st.error("GROQ_API_KEY is not set. Copy `.env.example` to `.env`, add your key, and rerun.")

with st.sidebar:
    st.subheader("The dials")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.1, help="Zero for facts. Higher for variety.")
    max_tokens = st.slider("Max tokens out", 128, 4096, 800, 64, help="Give the answer room, then stop paying.")
    use_stream = st.toggle("Stream the answer", value=True)
    st.divider()
    st.caption(f"Model: `{settings.groq_model}`")
    st.caption("Sending is long and cheap. Writing is short and several times more expensive.")

playground, comparison = st.tabs(["Playground", "Weak vs sharp"])

with playground:
    preset = st.selectbox("Start from a prompt", list(LIBRARY), index=1)
    system_prompt = st.text_area("WHO IT IS — the system prompt", LIBRARY[preset], height=180, key=preset)

    left, right = st.columns(2)
    material = left.text_area("THE DATA — material, marked as material", SAMPLE_MATERIAL, height=200)
    question = right.text_area("THE QUESTION — asked last", "Summarize the material.", height=200)

    sent = estimate_tokens(system_prompt + material + question)
    st.caption(f"Roughly {sent} tokens going up, before the answer comes back.")

    if st.button("Send it", type="primary", disabled=not settings.groq_api_key):
        messages = build_messages(system_prompt, material, question)

        with st.expander("What actually goes over the wire"):
            st.json(messages)

        if use_stream:
            st.write_stream(stream(settings, messages, temperature, max_tokens))
        else:
            with st.spinner("Waiting on the model — every call is mostly waiting"):
                answer, usage = complete(settings, messages, temperature, max_tokens)
            st.markdown(answer)
            a, b, c = st.columns(3)
            a.metric("Prompt tokens", usage["prompt_tokens"])
            b.metric("Completion tokens", usage["completion_tokens"])
            c.metric("Total", usage["total_tokens"])

with comparison:
    st.write("Same model, same material, different instructions. Run both and read the two columns.")
    text = st.text_area("Material", SAMPLE_MATERIAL, height=160, key="compare_material")
    ask = st.text_input("Question", "Summarize the material.")

    if st.button("Run both", disabled=not settings.groq_api_key):
        weak_col, sharp_col = st.columns(2)

        with weak_col:
            st.subheader("Weak")
            st.code(WEAK, language="text")
            with st.spinner("Running…"):
                answer, usage = complete(settings, build_messages(WEAK, text, ask), temperature, max_tokens)
            st.markdown(answer)
            st.caption(f"{usage['total_tokens']} tokens")

        with sharp_col:
            st.subheader("Sharp")
            st.code(SHARP, language="text")
            with st.spinner("Running…"):
                answer, usage = complete(settings, build_messages(SHARP, text, ask), temperature, max_tokens)
            st.markdown(answer)
            st.caption(f"{usage['total_tokens']} tokens")

        st.info("Run it three more times. The sharp one keeps its shape; the weak one drifts every time.")
