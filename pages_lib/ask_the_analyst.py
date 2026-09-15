import streamlit as st
import pandas as pd
from utils.styling import apply_custom_style
from utils.db import get_all_cases, RISK_COLORS
from utils.genai import ask_analyst_groq

apply_custom_style()

st.title("💬 Ask the Analyst")
st.caption("Interactive AI analyst assistant to query model performance, active batch metrics, or case log status.")

# ---------- Context Builder ----------
def build_context() -> str:
    lines = [
        "MODEL SPECIFICATIONS & BENCHMARKS (from training):",
        "- Architecture: XGBoost (Tuned) & Random Forest Benchmark.",
        "- Decision Threshold: 0.50 (High risk: >= 0.80, Medium-High: >= 0.50, Low-Medium: >= 0.30, Normal: < 0.30).",
        "- Top Decisive Features: amt_log, category, city_pop, trans_depth."
    ]

    # Check for session predictions batch
    if "last_predictions" in st.session_state and isinstance(st.session_state["last_predictions"], pd.DataFrame):
        results = st.session_state["last_predictions"]
        n_total = len(results)
        n_flagged = int(results["fraud_probability"].ge(0.5).sum()) if "fraud_probability" in results.columns else 0
        lines.append(f"\nCURRENT SESSION BATCH: {n_total} transaction(s) scored, {n_flagged} flagged as suspicious.")
    else:
        lines.append("\nCURRENT SESSION BATCH: No new batch scored yet in this session.")

    # Check case logs in database
    cases = get_all_cases()
    if cases.empty:
        lines.append("\nCASE DATABASE: No cases logged yet.")
    else:
        status_counts = cases["status"].value_counts().to_dict()
        risk_counts = cases["risk_level"].value_counts().to_dict()
        lines.append(
            f"\nCASE DATABASE SUMMARY:\n"
            f"- Total Logged: {len(cases)} cases.\n"
            f"- Distribution by Risk Tier: {risk_counts}\n"
            f"- Distribution by Resolution Status: {status_counts}"
        )

    return "\n".join(lines)

with st.expander("🔍 What real-time system context is the AI analyst using?"):
    st.code(build_context(), language="markdown")

st.divider()

# ---------- Chat History ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- Quick Suggestion Buttons ----------
suggestions = [
    "How many cases are pending review?",
    "What are the risk score threshold tiers?",
    "Summarise the case database distribution.",
]

cols = st.columns(len(suggestions))
for i, s in enumerate(suggestions):
    if cols[i].button(s, use_container_width=True):
        st.session_state.pending_question = s

question = st.chat_input("Ask about model performance, thresholds, or the case queue...")
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing operational context..."):
            context = build_context()
            system_msg = {
                "role": "system",
                "content": (
                    "You are a helpful AML fraud analytics assistant. "
                    "Answer accurately and concisely using ONLY the real-time operational context provided below. "
                    "If the context does not contain the answer, state that clearly.\n\n"
                    f"OPERATIONAL CONTEXT:\n{context}"
                ),
            }
            messages = [system_msg] + st.session_state.chat_history
            answer = ask_analyst_groq(messages=messages)
        st.write(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

if st.session_state.chat_history:
    if st.button("🗑️ Clear Conversation", key="clear_analyst_chat"):
        st.session_state.chat_history = []
        st.rerun()