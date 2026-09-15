# pages_lib/fraud_analyst.py
import json
import os
import numpy as np
import pandas as pd
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

from utils.db import get_all_cases, update_case_analysis, update_analyst_review
from utils.genai import get_groq_recommendation, ask_analyst_groq
from utils.preprocessing import load_artifacts, prepare_model_input
from utils.styling import apply_custom_style

apply_custom_style()

st.title("🕵️ Fraud Analyst Review")
st.caption("Review cases using two separate queues: High-Risk (False Positive focus) and Low-Risk (False Negative focus).")

# =============================================================================
# Load cases
# =============================================================================
cases = get_all_cases()

if cases.empty:
    st.info("No review cases are currently stored. Ask the Admin to upload a dataset and run Prediction.")
    st.stop()

# Ensure required columns
if "analyst_decision" not in cases.columns:
    cases["analyst_decision"] = ""
if "status" not in cases.columns:
    cases["status"] = "Pending Review"
if "risk_level" not in cases.columns:
    cases["risk_level"] = "Green"

cases["review_state"] = cases["analyst_decision"].fillna("").replace("", "Pending Analyst Review")

# =============================================================================
# Summary Metrics
# =============================================================================
pending = int((cases["review_state"] == "Pending Analyst Review").sum())
suspicious = int((cases["review_state"] == "Suspicious").sum())
not_fraud = int((cases["review_state"] == "Not a Fraud Case").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Review Cases", f"{len(cases):,}")
c2.metric("Pending Review", f"{pending:,}")
c3.metric("Marked Suspicious", f"{suspicious:,}")
c4.metric("Marked Not Fraud", f"{not_fraud:,}")

st.markdown("---")

# =============================================================================
# Helper function to render case review UI
# =============================================================================
def render_case_review(case, queue_type="high"):
    """Renders the full case review panel (used by both queues)."""
    case_id = case.get("case_id")
    prob_val = float(case.get("fraud_probability", 0) or 0)
    amt_val = float(case.get("amt", 0) or 0)
    risk_val = str(case.get("risk_level", "Unknown"))
    category = str(case.get("category", "General"))
    transaction_id = str(case.get("transaction_id", "N/A"))

    # ---- Basic metrics ----
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Transaction", transaction_id)
    d2.metric("Amount", f"RM {amt_val:,.2f}")
    d3.metric("Fraud Probability", f"{prob_val:.4f}")
    d4.metric("Risk Level", risk_val)

    # ---- SHAP & LIME side by side ----
    col_shap, col_lime = st.columns(2, gap="large")

    with col_shap:
        st.markdown("#### 🧠 SHAP Analysis")
        shap_text = str(case.get("shap_analysis") or "").strip()
        if shap_text:
            for part in shap_text.split("; "):
                if part.strip():
                    st.markdown(f"- {part}")
        else:
            st.info("SHAP analysis is not available for this case.")

    with col_lime:
        st.markdown("#### 🧪 LIME Explanation")
        try:
            model, preprocessor = load_artifacts()

            # Try to get background data for LIME
            X_all = st.session_state.get("prediction_X", None)

            if X_all is None:
                bg_path = "models/lime_training_background.csv"
                if os.path.exists(bg_path):
                    df_bg = pd.read_csv(bg_path)
                    X_all, _ = prepare_model_input(df_bg, preprocessor)
                else:
                    # Fallback using stored transaction
                    transaction = {}
                    try:
                        transaction = json.loads(case.get("transaction_json") or "{}")
                    except Exception:
                        pass
                    df_fallback = pd.DataFrame([transaction]) if transaction else pd.DataFrame()
                    if not df_fallback.empty:
                        X_all, _ = prepare_model_input(df_fallback, preprocessor)
                    else:
                        raise ValueError("No background data available for LIME")

            # Select the row to explain
            pred_row_id = case.get("prediction_row_id", 0)
            try:
                pred_row_id = int(pred_row_id)
            except Exception:
                pred_row_id = 0

            if pred_row_id < len(X_all):
                row_vals = X_all.iloc[pred_row_id]
            else:
                row_vals = X_all.iloc[0]

            lime_explainer = LimeTabularExplainer(
                training_data=np.array(X_all),
                feature_names=list(X_all.columns),
                class_names=["Legitimate", "Fraud"],
                mode="classification"
            )

            exp = lime_explainer.explain_instance(
                data_row=row_vals.to_numpy(),
                predict_fn=model.predict_proba,
                num_features=5
            )

            for feat_desc, weight in exp.as_list():
                st.markdown(f"- `{feat_desc}` : **{weight:+.4f}**")

        except Exception as e:
            st.info(f"LIME explanation could not be rendered: {e}")

    st.markdown("---")

    # =====================================================
    # 1. Insight & Reference Assessment
    # =====================================================
    st.markdown("### 💡 Insight & Reference Assessment")

    if risk_val in ["Red", "Orange", "Critical Risk", "High Risk"] or prob_val >= 0.90:
        reference_badge = "🚨 **Reference Indicator: SUSPICIOUS**"
        reference_desc = (
            "The model high-risk probability score and feature weights strongly suggest "
            "an anomalous pattern typical of fraudulent behavior. Thorough validation of "
            "cardholder activity is advised."
        )
        badge_bg = "#fef2f2"
        badge_border = "#ef4444"
    else:
        reference_badge = "✅ **Reference Indicator: LIKELY NOT A FRAUD CASE**"
        reference_desc = (
            "Although this case is in the review queue, current metrics suggest it may be "
            "legitimate. Review carefully to catch possible False Negatives."
        )
        badge_bg = "#f0fdf4"
        badge_border = "#22c55e"

    st.markdown(f"""
    <div style="background-color: {badge_bg}; border-left: 4px solid {badge_border};
                border-radius: 6px; padding: 14px; margin-bottom: 12px;">
        {reference_badge}<br>
        <p style="font-size: 0.92rem; margin-top: 4px; margin-bottom: 0; color: #334155;">
        {reference_desc}</p>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # 2. Recommendation For Next Step
    # =====================================================
    st.markdown("### 🧭 Recommendation For Next Step")

    recommendation = str(case.get("recommendation") or "").strip()
    if recommendation:
        st.markdown(
            f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
            f"padding:16px;line-height:1.6;'>{recommendation}</div>",
            unsafe_allow_html=True,
        )
    else:
        if st.button("Generate Recommendation", key=f"rec_{queue_type}_{case_id}"):
            with st.spinner("Generating recommendation..."):
                recommendation = get_groq_recommendation(
                    transaction_id=transaction_id,
                    amount=amt_val,
                    category=category,
                    fraud_score=prob_val,
                    risk_level=risk_val,
                    top_feature=case.get("top_feature", "SHAP contributors"),
                )
                update_case_analysis(
                    str(case.get("case_key", case_id)),
                    recommendation=recommendation
                )
                st.rerun()

    # =====================================================
    # 3. Decision Assistant & Verification Guide
    # =====================================================
    st.markdown("### ⚖️ Decision Assistant & Verification Guide")

    with st.expander("🔍 Click to view analytical checklist for decision support", expanded=True):
        col_chk1, col_chk2 = st.columns(2)
        with col_chk1:
            st.markdown("**🚨 Evidence Supporting Fraud (Suspicious):**")
            st.markdown(f"- High anomaly model score probability (`{prob_val:.4f}`).")
            st.markdown(f"- Transaction category profile (`{category}`) exhibits high historical risk.")
            st.markdown("- SHAP / LIME features display heavy positive risk contributions.")
        with col_chk2:
            st.markdown("**🛡️ Evidence Suggesting Legitimate (Not Fraud):**")
            st.markdown(f"- Transaction value (RM {amt_val:,.2f}) falls within standard cardholder behavior limits.")
            st.markdown("- Geolocation and distance metrics align with historical patterns.")
            st.markdown("- Absence of multi-channel velocity triggers.")

        if prob_val >= 0.90:
            st.markdown(
                "👉 **Suggested Verdict:** <span style='color:red; font-weight:bold;'>Suspicious</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "👉 **Suggested Verdict:** <span style='color:green; font-weight:bold;'>Not a Fraud Case</span>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # =====================================================
    # 4. Analyst Decision
    # =====================================================
    st.markdown("### ✅ Analyst Decision")

    if queue_type == "low":
        st.caption("This is a Low-Risk case. If you believe it is actually fraud, mark it as **Suspicious** (False Negative).")

    decision = st.radio(
        "Final verification decision",
        ["Suspicious", "Not a Fraud Case"],
        horizontal=True,
        key=f"decision_{queue_type}_{case_id}"
    )

    notes = st.text_area(
        "Analyst Notes / Reason",
        value=str(case.get("analyst_notes") or case.get("notes") or ""),
        placeholder="Explain the evidence or reasoning supporting your decision.",
        height=120,
        key=f"notes_{queue_type}_{case_id}"
    )

    if st.button("💾 Save Case Decision", type="primary", use_container_width=True,
                 key=f"save_{queue_type}_{case_id}"):
        analyst_name = st.session_state.get("username", "Fraud Analyst")
        update_analyst_review(
            str(case.get("case_key", case_id)),
            decision,
            notes.strip(),
            analyst_name
        )
        st.success(f"Case #{case_id} saved as **{decision}**.")
        st.rerun()

# =============================================================================
# TWO QUEUES
# =============================================================================
tab1, tab2 = st.tabs(["🔴 High-Risk Queue", "🟡 Low-Risk Review Queue"])

# -----------------------------------------------------------------------------
# TAB 1: High-Risk Queue
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("High-Risk Queue")
    st.info("These cases were predicted as **Suspicious** by the model. Your goal is to confirm fraud or clear them as **False Positive**.")

    # Filter by Review Status
    status_filter = st.selectbox(
        "Filter by Review Status",
        ["Pending Analyst Review", "Suspicious", "Not a Fraud Case", "All"],
        key="high_status_filter"
    )

    high_risk_base = cases[
        cases["risk_level"].isin(["Red", "Orange", "Critical Risk", "High Risk"])
    ].copy()

    if status_filter != "All":
        high_risk = high_risk_base[high_risk_base["review_state"] == status_filter].copy()
    else:
        high_risk = high_risk_base.copy()

    if high_risk.empty:
        st.success(f"No high-risk cases with status: **{status_filter}**")
    else:
        st.write(f"**{len(high_risk)}** high-risk cases (Status: {status_filter})")

        queue_cols = [c for c in [
            "case_id", "transaction_id", "amt", "category",
            "fraud_probability", "risk_level", "review_state"
        ] if c in high_risk.columns]

        st.dataframe(high_risk[queue_cols], use_container_width=True, height=260)

        selected_high = st.selectbox(
            "Select a High-Risk case to review",
            high_risk["case_id"].astype(int).tolist(),
            format_func=lambda x: (
                f"Case #{x} | "
                f"{high_risk.loc[high_risk['case_id']==x, 'transaction_id'].iloc[0]} | "
                f"Prob: {float(high_risk.loc[high_risk['case_id']==x, 'fraud_probability'].iloc[0]):.4f} | "
                f"{high_risk.loc[high_risk['case_id']==x, 'review_state'].iloc[0]}"
            ),
            key="high_risk_select"
        )

        case = high_risk[high_risk["case_id"] == selected_high].iloc[0].to_dict()
        render_case_review(case, queue_type="high")

# -----------------------------------------------------------------------------
# TAB 2: Low-Risk Review Queue
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Low-Risk Review Queue")
    st.warning("These cases were predicted as **Normal** by the model. Review them carefully to catch possible **False Negatives**.")

    status_filter_low = st.selectbox(
        "Filter by Review Status",
        ["Pending Analyst Review", "Suspicious", "Not a Fraud Case", "All"],
        key="low_status_filter"
    )

    low_risk_base = cases[
        cases["risk_level"].isin(["Yellow", "Green", "Low Risk", "Normal", "Moderate Risk"])
    ].copy()

    if status_filter_low != "All":
        low_risk = low_risk_base[low_risk_base["review_state"] == status_filter_low].copy()
    else:
        low_risk = low_risk_base.copy()

    if low_risk.empty:
        st.success(f"No low-risk cases with status: **{status_filter_low}**")
    else:
        st.write(f"**{len(low_risk)}** low-risk cases (Status: {status_filter_low})")

        queue_cols = [c for c in [
            "case_id", "transaction_id", "amt", "category",
            "fraud_probability", "risk_level", "review_state"
        ] if c in low_risk.columns]

        st.dataframe(low_risk[queue_cols], use_container_width=True, height=260)

        selected_low = st.selectbox(
            "Select a Low-Risk case to review",
            low_risk["case_id"].astype(int).tolist(),
            format_func=lambda x: (
                f"Case #{x} | "
                f"{low_risk.loc[low_risk['case_id']==x, 'transaction_id'].iloc[0]} | "
                f"Prob: {float(low_risk.loc[low_risk['case_id']==x, 'fraud_probability'].iloc[0]):.4f} | "
                f"{low_risk.loc[low_risk['case_id']==x, 'review_state'].iloc[0]}"
            ),
            key="low_risk_select"
        )

        case = low_risk[low_risk["case_id"] == selected_low].iloc[0].to_dict()
        render_case_review(case, queue_type="low")