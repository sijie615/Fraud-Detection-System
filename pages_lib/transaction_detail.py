# pages_lib/transaction_detail.py
import streamlit as st
import pandas as pd
from utils.styling import apply_custom_style
from utils.db import RISK_COLORS, get_latest_dataset, get_risk_level
from utils.genai import ask_analyst_groq

apply_custom_style()

st.title("📄 Transaction Detail")
st.caption("Full detail view for a single scored transaction.")

# -----------------------------------------------------------------------------
# Read the latest active dataset across all pages
# -----------------------------------------------------------------------------
results = get_latest_dataset()

if results.empty:
    if "prediction_results" in st.session_state and isinstance(st.session_state["prediction_results"], pd.DataFrame):
        results = st.session_state["prediction_results"].copy()

if results.empty:
    st.warning("No predictions yet. Go to **Prediction** first.")
    st.stop()

# Ensure standard columns exist
if "amt" not in results.columns and "amount" in results.columns:
    results["amt"] = results["amount"]

if "fraud_probability" not in results.columns and "is_fraud" in results.columns:
    results["fraud_probability"] = results["is_fraud"].astype(float)

if "risk_level" not in results.columns:
    if "fraud_probability" in results.columns:
        results["risk_level"] = results["fraud_probability"].apply(get_risk_level)
    else:
        results["risk_level"] = "Green"

# Identify or assign a display ID
id_col = None
for col_candidate in ["trans_num", "trans_id", "transaction_id", "id", "Transaction_ID"]:
    if col_candidate in results.columns:
        id_col = col_candidate
        break

if id_col is None:
    results["trans_num"] = [f"TX-{i:05d}" for i in range(len(results))]
    id_col = "trans_num"

# ========== SELECT TRANSACTION FROM LIST ==========
st.subheader("🔍 Select a Transaction")

display_options = results.index.tolist()[:500]  # Limit to first 500 rows for responsive UI performance
selected_idx = st.selectbox(
    "Choose a transaction from the list:",
    options=display_options,
    format_func=lambda i: (
        f"Row {i} | ID: {results.loc[i, id_col]} | "
        f"Amt: ${results.loc[i, 'amt']:.2f} | "
        f"Risk: {results.loc[i, 'risk_level']}"
        if "amt" in results.columns and "risk_level" in results.columns
        else f"Row {i} | ID: {results.loc[i, id_col]}"
    )
)

row = results.loc[selected_idx]

# ========== DISPLAY DETAIL ==========
if row is not None:
    color = RISK_COLORS.get(row.get("risk_level", "Green"), "#4caf50")
    curr_prob = float(row.get("fraud_probability", 0.0))
    curr_risk = str(row.get("risk_level", "Green"))
    curr_amt = float(row.get("amt", row.get("transaction_amount", 0.0)))
    curr_cat = str(row.get("category", "General"))

    st.markdown(
        f"""
        <div style='background:{color}22; border-left:6px solid {color}; 
                    padding:16px; border-radius:8px; font-size:1.15rem; margin-bottom:20px;'>
            <b>Risk Level:</b> {curr_risk} &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Fraud Probability:</b> {curr_prob:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Amount:</b> ${curr_amt:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Transaction Details")
    
    # Original two-column layout
    col1, col2 = st.columns(2)
    items = list(row.items())
    mid = len(items) // 2

    with col1:
        for key, value in items[:mid]:
            st.markdown(f"**{key}**")
            st.write(value)
            st.markdown("---")

    with col2:
        for key, value in items[mid:]:
            st.markdown(f"**{key}**")
            st.write(value)
            st.markdown("---")

    # ==============================================================================
    # 🤖 AI Assistant: Transaction Explanation Chatbot (At the bottom)
    # ==============================================================================
    st.markdown("---")
    st.subheader(f"💬 Ask AI Analyst About Transaction #{row.get(id_col, selected_idx)}")

    tx_chat_key = f"detail_chat_{selected_idx}"
    if tx_chat_key not in st.session_state:
        st.session_state[tx_chat_key] = []

    c_btn, _ = st.columns([1.8, 3.2])
    with c_btn:
        if st.button("💡 Explain This Transaction", key=f"btn_exp_detail_{selected_idx}", use_container_width=True):
            with st.spinner("AI evaluating transaction profile..."):
                prompt = f"""
                You are a senior AML forensic auditor.
                Explain the risk profile for this transaction:
                - ID: {row.get(id_col, selected_idx)}
                - Amount: ${curr_amt:,.2f}
                - Category: {curr_cat}
                - Fraud Probability: {curr_prob:.4f} (Risk Tier: {curr_risk})
                - Context Attributes: {dict(list(row.items())[:8])}

                Provide a 2-part summary:
                1. Assessment: Why this transaction earned its {curr_risk} rating.
                2. Recommended Action: What the investigator should inspect first.
                Keep it authoritative, concise, and under 80 words.
                """
                ai_resp = ask_analyst_groq([
                    {"role": "system", "content": "You are a professional financial crimes intelligence officer."},
                    {"role": "user", "content": prompt}
                ])
                st.session_state[tx_chat_key].append({"role": "assistant", "content": ai_resp})
                st.rerun()

    chat_box = st.container(height=260)
    with chat_box:
        if not st.session_state[tx_chat_key]:
            st.caption("No questions asked yet for this transaction. Type your question below or click '💡 Explain This Transaction'.")
        for m in st.session_state[tx_chat_key]:
            with st.chat_message(m["role"]):
                st.write(m["content"])

    user_q = st.chat_input(f"Ask why this transaction is {curr_risk}, check its amount anomaly, etc...", key=f"chat_inp_detail_{selected_idx}")
    if user_q:
        st.session_state[tx_chat_key].append({"role": "user", "content": user_q})
        with chat_box:
            with st.chat_message("user"):
                st.write(user_q)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing transaction details..."):
                    sys_prompt = (
                        f"You are an AML specialist assisting an investigator with transaction #{row.get(id_col, selected_idx)}. "
                        f"Summary: Amount=${curr_amt:,.2f}, Category={curr_cat}, Risk={curr_risk}, Probability={curr_prob:.4f}. "
                        f"Full Attributes: {dict(items[:10])}. "
                        "Answer questions accurately, professionally, and concisely."
                    )
                    ans = ask_analyst_groq([{"role": "system", "content": sys_prompt}] + st.session_state[tx_chat_key])
                st.write(ans)
                st.session_state[tx_chat_key].append({"role": "assistant", "content": ans})

    if st.session_state[tx_chat_key]:
        if st.button("🗑️ Clear Chat History", key=f"clr_detail_chat_{selected_idx}"):
            st.session_state[tx_chat_key] = []
            st.rerun()