# pages_lib/admin_home.py
import streamlit as st
import pandas as pd
from utils.styling import apply_custom_style
from utils.db import get_all_cases, get_connection

apply_custom_style()

st.title("🏠 Admin Homepage")
st.caption("System overview — upload data, run predictions, and monitor overall fraud risk.")

# =============================================================================
# Load current dataset
# =============================================================================
df = st.session_state.get("prediction_results")
if not isinstance(df, pd.DataFrame) or df.empty:
    df = st.session_state.get("uploaded_df")

if not isinstance(df, pd.DataFrame) or df.empty:
    st.info("No dataset is currently loaded. Go to **Upload Data** to begin.")
    st.markdown("---")
    st.subheader("Workflow")
    st.markdown(
        "**1. Upload Data → 2. Run Prediction → 3. Review Analytics → "
        "4. Fraud Analyst verifies review cases**"
    )
    st.stop()

# =============================================================================
# Risk counts
# =============================================================================
risk = df.get("risk_level", pd.Series(["Green"] * len(df), index=df.index))
red_cases = int(risk.isin(["Red", "Critical Risk"]).sum())
orange_cases = int(risk.isin(["Orange", "High Risk"]).sum())
yellow_cases = int(risk.isin(["Yellow", "Moderate Risk", "Low-Medium"]).sum())
green_cases = int(risk.isin(["Green", "Low Risk", "Normal"]).sum())

flag_series = (
    df["predicted_is_suspicious"] if "predicted_is_suspicious" in df.columns
    else (df["is_suspicious"] if "is_suspicious" in df.columns else pd.Series(0, index=df.index))
)
flagged = int(pd.to_numeric(flag_series, errors="coerce").fillna(0).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Transactions", f"{len(df):,}")
c2.metric("🔴 Red", f"{red_cases:,}")
c3.metric("🟠 Orange", f"{orange_cases:,}")
c4.metric("🟡 Yellow", f"{yellow_cases:,}")
c5.metric("🟢 Green", f"{green_cases:,}")

st.markdown("---")

# =============================================================================
# Risk Tier Guide
# =============================================================================
st.subheader("📌 Risk Tier Classification Guide")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div style="padding:12px; border-left:4px solid #ef4444; background:#fef2f2; border-radius:6px;">
    <b style="color:#b91c1c;">🔴 Red (High Risk)</b><br>
    <small><b>Score: 0.80 – 1.00</b></small><br>
    <span style="font-size:13px;color:#374151;">Highest-risk model output requiring Fraud Analyst attention.</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div style="padding:12px; border-left:4px solid #f97316; background:#fff7ed; border-radius:6px;">
    <b style="color:#c2410c;">🟠 Orange (Med-High)</b><br>
    <small><b>Score: 0.50 – 0.79</b></small><br>
    <span style="font-size:13px;color:#374151;">Elevated model risk and included in analyst review when flagged.</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div style="padding:12px; border-left:4px solid #eab308; background:#fefce8; border-radius:6px;">
    <b style="color:#a16207;">🟡 Yellow (Low-Med)</b><br>
    <small><b>Score: 0.30 – 0.49</b></small><br>
    <span style="font-size:13px;color:#374151;">Borderline model risk that can be monitored through the dashboard.</span>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div style="padding:12px; border-left:4px solid #22c55e; background:#f0fdf4; border-radius:6px;">
    <b style="color:#15803d;">🟢 Green (Normal)</b><br>
    <small><b>Score: 0.00 – 0.29</b></small><br>
    <span style="font-size:13px;color:#374151;">Lower model risk; normally not sent to analyst review.</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# Current Prediction Summary
# =============================================================================
st.subheader("📊 Current Prediction Summary")

s1, s2, s3 = st.columns(3)
s1.metric("Review Cases", f"{flagged:,}")

if "fraud_probability" in df.columns:
    s2.metric("Average Fraud Probability", f"{df['fraud_probability'].mean():.4f}")
else:
    s2.metric("Average Fraud Probability", "N/A")

cases = get_all_cases()
reviewed_count = 0
if not cases.empty and "analyst_decision" in cases.columns:
    reviewed_count = int(cases["analyst_decision"].fillna("").astype(str).ne("").sum())
s3.metric("Analyst Reviews Completed", f"{reviewed_count:,}")

st.subheader("Recent / Highest-Risk Results")
display_cols = [
    "trans_num", "transaction_id", "trans_date_trans_time", "amt", "category",
    "fraud_probability", "risk_level", "alert", "predicted_is_suspicious"
]
existing = [c for c in display_cols if c in df.columns]
show_df = df.sort_values("fraud_probability", ascending=False).head(25) if "fraud_probability" in df.columns else df.head(25)
st.dataframe(show_df[existing] if existing else show_df, use_container_width=True)

st.markdown("---")

# =============================================================================
# Manage / Delete Cases (placed at the bottom)
# =============================================================================
st.subheader("🗑️ Manage / Delete Cases")

cases = get_all_cases()

if cases.empty:
    st.info("No cases in the database.")
else:
    st.write(f"Total cases in database: **{len(cases)}**")

    display_cols = [c for c in [
        "case_id", "transaction_id", "amt", "risk_level",
        "fraud_probability", "status", "analyst_decision"
    ] if c in cases.columns]

    st.dataframe(cases[display_cols], use_container_width=True, height=300)

    st.markdown("### Delete Options")
    col1, col2, col3 = st.columns(3)

    with col1:
        case_id_to_delete = st.number_input("Delete by Case ID", min_value=1, step=1, key="del_case_id")
        if st.button("🗑️ Delete This Case", type="primary", key="btn_del_one"):
            conn = get_connection()
            conn.execute("DELETE FROM fraud_cases WHERE case_id = ?", (int(case_id_to_delete),))
            conn.commit()
            conn.close()
            st.success(f"Case #{case_id_to_delete} deleted.")
            st.rerun()

    with col2:
        if st.button("🗑️ Delete All Pending Review", key="btn_del_pending"):
            conn = get_connection()
            result = conn.execute("DELETE FROM fraud_cases WHERE status = 'Pending Review'")
            conn.commit()
            deleted = result.rowcount
            conn.close()
            st.success(f"Deleted {deleted} pending cases.")
            st.rerun()

    with col3:
        if st.button("⚠️ Delete ALL Cases", type="secondary", key="btn_del_all"):
            conn = get_connection()
            conn.execute("DELETE FROM fraud_cases")
            conn.commit()
            conn.close()
            st.success("All cases have been deleted.")
            st.rerun()