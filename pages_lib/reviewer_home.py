import streamlit as st
import pandas as pd
from utils.styling import apply_custom_style
from utils.db import get_cases_by_status, get_all_cases

apply_custom_style()

st.title("🏠 Reviewer Homepage")
st.caption("Cases awaiting your verification.")

username = st.session_state.get("username", "reviewer")

pending_cases = get_cases_by_status("Pending Review", reviewer=username)
verified_cases = get_cases_by_status("Verified", reviewer=username)

# Fallback check if assigned to generic reviewer queue
if pending_cases.empty:
    all_cases = get_all_cases()
    if not all_cases.empty and "status" in all_cases.columns:
        pending_cases = all_cases[
            (all_cases["status"].str.lower() == "pending review") &
            (all_cases["risk_level"].isin(["Yellow", "Green", "Moderate Risk", "Low Risk", "Normal"]))
        ]
        verified_cases = all_cases[all_cases["status"].str.lower() == "verified"]

pending_count = len(pending_cases)
verified_count = len(verified_cases)

c1, c2 = st.columns(2)
c1.metric("Cases Awaiting Verification", pending_count)
c2.metric("Cases You've Verified", verified_count)

st.markdown("---")

if pending_count > 0:
    st.subheader(f"📋 Normal / Low-Risk Cases Awaiting Verification ({pending_count})")
    
    display_cols = [
        "case_id", "transaction_id", "timestamp", "amt", "category",
        "fraud_probability", "risk_level", "notes", "status"
    ]
    cols = [c for c in display_cols if c in pending_cases.columns]
    
    st.dataframe(pending_cases[cols].head(20), use_container_width=True)
    
    if st.button("🔍 Open Case Management", type="primary"):
        st.switch_page("pages_lib/case_management.py")
else:
    st.info("No pending cases. Sampled normal transactions from Prediction will appear here.")