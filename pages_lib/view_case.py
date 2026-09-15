import json
import pandas as pd
import streamlit as st
from utils.db import get_all_cases
from utils.styling import apply_custom_style

apply_custom_style()

st.title("🔎 View Results")
st.caption("View prediction results and persistent Fraud Analyst verification decisions.")

cases = get_all_cases()

# Keep this Admin page focused on results and insight/status. Recommendation is
# intentionally available only in the Fraud Analyst workflow.
if cases.empty:
    df = st.session_state.get("prediction_results")
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No prediction results or stored review cases are available yet.")
        st.stop()
    st.info("No review cases have been stored yet. Run Prediction to create cases for analyst review.")
    st.dataframe(df, use_container_width=True, height=500)
    st.stop()

cases["review_state"] = cases.get("analyst_decision", pd.Series(index=cases.index, dtype=str)).fillna("")
cases.loc[cases["review_state"] == "", "review_state"] = "Pending Analyst Review"

pending = int((cases["review_state"] == "Pending Analyst Review").sum())
suspicious = int((cases["review_state"] == "Suspicious").sum())
not_fraud = int((cases["review_state"] == "Not a Fraud Case").sum())

st.subheader("📊 Case Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stored Cases", f"{len(cases):,}")
c2.metric("Pending Review", f"{pending:,}")
c3.metric("Suspicious", f"{suspicious:,}")
c4.metric("Not a Fraud Case", f"{not_fraud:,}")

st.markdown("---")
search = st.text_input("Search case ID, transaction ID or category", "").strip()
status_filter = st.selectbox(
    "Analyst Decision",
    ["All", "Pending Analyst Review", "Suspicious", "Not a Fraud Case"],
)

filtered = cases.copy()
if search:
    mask = pd.Series(False, index=filtered.index)
    for col in ["case_id", "transaction_id", "category"]:
        if col in filtered.columns:
            mask |= filtered[col].astype(str).str.contains(search, case=False, na=False)
    filtered = filtered[mask]
if status_filter != "All":
    filtered = filtered[filtered["review_state"] == status_filter]

st.caption(f"Showing {len(filtered):,} stored case(s).")
cols = [
    "case_id", "transaction_id", "timestamp", "amt", "category",
    "fraud_probability", "risk_level", "review_state", "analyst_name", "reviewed_at"
]
cols = [c for c in cols if c in filtered.columns]
st.dataframe(filtered[cols], use_container_width=True, height=500)

if filtered.empty:
    st.stop()

st.markdown("---")
st.subheader("🧠 Case Insight")
selected_case = st.selectbox("Select a stored case", filtered["case_id"].astype(int).tolist())
case = cases[cases["case_id"] == selected_case].iloc[0].to_dict()

c1, c2, c3 = st.columns(3)
c1.metric("Fraud Probability", f"{float(case.get('fraud_probability', 0) or 0):.4f}")
c2.metric("Risk Level", str(case.get("risk_level", "Unknown")))
c3.metric("Analyst Decision", str(case.get("review_state", "Pending Analyst Review")))

shap_text = str(case.get("shap_analysis") or "").strip()
if shap_text:
    st.markdown("**SHAP Analysis**")
    for part in shap_text.split("; "):
        if part.strip():
            st.markdown(f"- {part}")
else:
    st.info("No SHAP analysis is stored for this case.")

try:
    transaction = json.loads(case.get("transaction_json") or "{}")
except Exception:
    transaction = {}

if transaction:
    with st.expander("📄 Transaction Details"):
        preferred = [
            "trans_date_trans_time", "merchant", "category", "amt", "gender",
            "city", "state", "city_pop", "lat", "long", "merch_lat", "merch_long"
        ]
        detail = {k: transaction.get(k) for k in preferred if k in transaction}
        st.dataframe(pd.DataFrame([detail or transaction]), use_container_width=True, hide_index=True)
