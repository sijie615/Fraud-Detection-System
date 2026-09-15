import streamlit as st
import pandas as pd
from utils.styling import apply_custom_style
from utils.db import get_connection, RISK_COLORS

apply_custom_style()

st.title("📋 Case Logs & Resolved History")

username = st.session_state.get("username", "investigator")
role = st.session_state.get("role", "Investigator")

st.caption(f"Logged in as **{username}** ({role}). Tracking historical audits, False Positives, and False Negatives.")

# -----------------------------------------------------------------------------
# Query Handled Logs (Role-Separated)
# -----------------------------------------------------------------------------
conn = get_connection()
c = conn.cursor()

if role == "Reviewer":
    # Cases handled by the Reviewer
    c.execute("""
        SELECT case_id, transaction_id, amt, category, fraud_probability, risk_level, status, notes, reviewer, timestamp 
        FROM fraud_cases 
        WHERE status != 'Pending Review'
          AND (reviewer = ? OR reviewer = 'reviewer' OR status LIKE '%Normal%' OR status LIKE '%False Negative%')
        ORDER BY case_id DESC
    """, (str(username),))
else:
    # Cases handled by the Investigator
    c.execute("""
        SELECT case_id, transaction_id, amt, category, fraud_probability, risk_level, status, notes, investigator, timestamp 
        FROM fraud_cases 
        WHERE status != 'Pending Review'
          AND (investigator = ? OR investigator = 'investigator' OR status LIKE '%Reported%' OR status LIKE '%False Positive%')
        ORDER BY case_id DESC
    """, (str(username),))

logs_df = pd.DataFrame([dict(ix) for ix in c.fetchall()])
conn.close()

if logs_df.empty:
    st.info(f"No resolved case records found yet for **{username}**. Complete case reviews in Case Management to view them here.")
    st.stop()

# Derive Standard Classification Badges
def classify_outcome(row):
    status_str = str(row.get("status", "")).lower()
    notes_str = str(row.get("notes", "")).lower()
    
    if "false positive" in status_str or "false positive" in notes_str:
        return "⚠️ False Positive (FP)"
    elif "false negative" in status_str or "false negative" in notes_str:
        return "🚨 False Negative (FN)"
    elif "true positive" in status_str or "reported" in status_str:
        return "🎯 Escalated Fraud (TP)"
    else:
        return "✅ Verified Normal (TN)"

logs_df["outcome_type"] = logs_df.apply(classify_outcome, axis=1)

# Metric KPI Header
k1, k2, k3, k4 = st.columns(4)
total_cnt = len(logs_df)
fp_cnt = len(logs_df[logs_df["outcome_type"].str.contains("False Positive")])
fn_cnt = len(logs_df[logs_df["outcome_type"].str.contains("False Negative")])
tp_cnt = len(logs_df[logs_df["outcome_type"].str.contains("Escalated Fraud")])

k1.metric("Total Handled", total_cnt)
k2.metric("False Positives (Cleared)", fp_cnt)
k3.metric("False Negatives (Flagged)", fn_cnt)
k4.metric("Escalated / Confirmed", tp_cnt)

st.markdown("---")

# -----------------------------------------------------------------------------
# Tabs to View Specific Categories
# -----------------------------------------------------------------------------
tab_all, tab_fp, tab_fn = st.tabs([
    f"All Handled Records ({total_cnt})", 
    f"⚠️ False Positives ({fp_cnt})", 
    f"🚨 False Negatives ({fn_cnt})"
])

def color_risk_level(val):
    c_val = RISK_COLORS.get(val, "#ffffff")
    return f"background-color: {c_val}33; font-weight: bold;"

with tab_all:
    st.dataframe(
        logs_df.style.map(color_risk_level, subset=["risk_level"]).format({
            "amt": "${:,.2f}",
            "fraud_probability": "{:.4f}"
        }),
        use_container_width=True,
        height=360
    )

with tab_fp:
    fp_df = logs_df[logs_df["outcome_type"].str.contains("False Positive")]
    if fp_df.empty:
        st.info("No False Positive cases cleared yet.")
    else:
        st.caption("Transactions flagged by the ML model as suspicious, but cleared by human audit as legitimate customer spend.")
        st.dataframe(
            fp_df.style.map(color_risk_level, subset=["risk_level"]).format({
                "amt": "${:,.2f}",
                "fraud_probability": "{:.4f}"
            }),
            use_container_width=True
        )

with tab_fn:
    fn_df = logs_df[logs_df["outcome_type"].str.contains("False Negative")]
    if fn_df.empty:
        st.info("No False Negative cases detected yet.")
    else:
        st.caption("Transactions predicted as Normal / Low-Risk by the model, but discovered as actual fraud by Reviewer quality checks.")
        st.dataframe(
            fn_df.style.map(color_risk_level, subset=["risk_level"]).format({
                "amt": "${:,.2f}",
                "fraud_probability": "{:.4f}"
            }),
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# Detailed Case Inspector
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔎 Inspect Case Audit Record")
selected_cid = st.selectbox("Select case ID to inspect:", options=logs_df["case_id"].tolist())
inspected = logs_df[logs_df["case_id"] == selected_cid].iloc[0]

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Case ID:** #{inspected['case_id']}")
    st.markdown(f"**Transaction ID:** `{inspected['transaction_id']}`")
    st.markdown(f"**Amount:** ${inspected['amt']:,.2f}")
    st.markdown(f"**Category:** {inspected['category']}")
    st.markdown(f"**Classification:** `{inspected['outcome_type']}`")
with c2:
    st.markdown(f"**Final Status:** `{inspected['status']}`")
    st.markdown(f"**Model Risk Level:** {inspected['risk_level']}")
    st.markdown(f"**Fraud Probability:** {inspected['fraud_probability']:.4f}")
    st.markdown(f"**Audit Notes / Reason:**")
    st.info(inspected['notes'])