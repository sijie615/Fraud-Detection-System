# pages_lib/model_performance.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)
from utils.styling import apply_custom_style
from utils.db import get_all_cases

apply_custom_style()

st.title("📈 Model Performance Monitoring")
st.caption("Live performance based on your latest predicted data + case management records.")

# =============================================================================
# Helper: Get the latest prediction results
# =============================================================================
def get_live_prediction_df():
    for key in ["prediction_results", "scored_df", "active_analytics_df", "uploaded_df"]:
        if key in st.session_state and isinstance(st.session_state[key], pd.DataFrame):
            df = st.session_state[key]
            if not df.empty:
                return df.copy()
    return pd.DataFrame()

results = get_live_prediction_df()

# =============================================================================
# 1. Time Period Selector
# =============================================================================
st.subheader("📅 Select Time Period")
time_period = st.selectbox(
    "Choose time period for analysis:",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
    index=1
)

# =============================================================================
# 2. Key Metrics  (now works on large datasets too)
# =============================================================================
st.subheader("🎯 Key Metrics")

# You can raise this further if your machine has enough RAM
MAX_FULL_ROWS = 500_000          # full calculation up to this size
MAX_SAMPLE_ROWS = 200_000        # if larger → stratified sample of this size

has_live_labels = False
y_true = y_pred = None
live_precision = live_recall = live_f1 = live_accuracy = None
used_sample = False
n_used = 0

if not results.empty:
    # Detect label + prediction columns
    if "is_fraud" in results.columns and "flagged_as_fraud" in results.columns:
        y_true_full = results["is_fraud"].astype(int)
        y_pred_full = results["flagged_as_fraud"].astype(int)
        has_live_labels = True
    elif "is_fraud" in results.columns and "fraud_probability" in results.columns:
        y_true_full = results["is_fraud"].astype(int)
        y_pred_full = (results["fraud_probability"] >= 0.9626).astype(int)
        has_live_labels = True

    if has_live_labels:
        n_total = len(results)

        if n_total <= MAX_FULL_ROWS:
            # Full dataset – fast enough
            y_true = y_true_full
            y_pred = y_pred_full
            n_used = n_total
            used_sample = False
        else:
            # Very large → stratified sample (keeps the rare fraud class)
            from sklearn.model_selection import train_test_split
            # We only need the two series, so we sample the indices
            idx = np.arange(n_total)
            # Stratify on the true label so the sample keeps the original fraud rate
            _, sample_idx = train_test_split(
                idx,
                test_size=min(MAX_SAMPLE_ROWS / n_total, 0.5),
                stratify=y_true_full,
                random_state=42
            )
            y_true = y_true_full.iloc[sample_idx]
            y_pred = y_pred_full.iloc[sample_idx]
            n_used = len(sample_idx)
            used_sample = True

        live_precision = precision_score(y_true, y_pred, zero_division=0)
        live_recall    = recall_score(y_true, y_pred, zero_division=0)
        live_f1        = f1_score(y_true, y_pred, zero_division=0)
        live_accuracy  = accuracy_score(y_true, y_pred)

# Official training results (fallback)
OFFICIAL_PRECISION = 0.9597
OFFICIAL_RECALL    = 0.8728
OFFICIAL_F1        = 0.9142
OFFICIAL_ACCURACY  = 0.9991

precision = live_precision if has_live_labels else OFFICIAL_PRECISION
recall    = live_recall    if has_live_labels else OFFICIAL_RECALL
f1        = live_f1        if has_live_labels else OFFICIAL_F1
accuracy  = live_accuracy  if has_live_labels else OFFICIAL_ACCURACY

# Status banner
if has_live_labels:
    if used_sample:
        st.warning(
            f"⚠️ **LIVE METRICS (STRATIFIED SAMPLE)** — "
            f"Original data has {len(results):,} rows. "
            f"Metrics calculated on a stratified sample of {n_used:,} rows "
            f"to keep the app responsive."
        )
    else:
        st.success(f"✅ **LIVE METRICS** — Calculated from all {n_used:,} predicted transactions")
else:
    st.info(
        "ℹ️ **OFFICIAL TRAINING METRICS** — "
        "Upload a labelled dataset (`is_fraud` column) and run Prediction to see live metrics."
    )

# Metric cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Recall",    f"{recall*100:.1f}%")
col2.metric("Precision", f"{precision*100:.1f}%")
col3.metric("F1-Score",  f"{f1*100:.1f}%")
col4.metric("Accuracy",  f"{accuracy*100:.1f}%")

# Performance Alert
if recall < 0.80 or precision < 0.80 or f1 < 0.80:
    st.error("⚠️ **Model Performance Alert**: One or more key metrics are below the acceptable threshold.")
else:
    st.success("✅ All key metrics are within acceptable range.")

st.markdown("---")

# =============================================================================
# 3. Case Status Breakdown + Visual Confusion Matrix
# =============================================================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 Case Status Breakdown")
    try:
        cases = get_all_cases()
        if not cases.empty and "status" in cases.columns:
            status_counts = cases["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            st.dataframe(status_counts, use_container_width=True, hide_index=True)
            reviewed = len(cases[cases["status"] != "Pending Review"])
            st.caption(f"Total Reviewed Cases: **{reviewed}**")
        else:
            st.info("No cases in database yet.")
    except Exception as e:
        st.warning(f"Could not load cases: {e}")

with col_right:
    st.subheader("🧩 Confusion Matrix")
    if has_live_labels:
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Normal", "Suspicious"],
            y=["Normal", "Suspicious"],
            colorscale="Blues",
            showscale=False,
            text=[[str(cm[i][j]) for j in range(2)] for i in range(2)],
            texttemplate="%{text}",
            textfont={"size": 20},
        ))
        fig_cm.update_layout(
            title=dict(text="Confusion Matrix", x=0.5),
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=360,
            margin=dict(l=70, r=40, t=60, b=50)
        )
        fig_cm.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_cm, use_container_width=True)
        if used_sample:
            st.caption("Confusion matrix is based on the stratified sample.")
    else:
        st.info("Live confusion matrix will appear after you run prediction on a labelled dataset.")

st.markdown("---")

# =============================================================================
# 4. Detailed Classification Report
# =============================================================================
with st.expander("📄 Detailed Classification Report", expanded=True):
    if has_live_labels:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.reset_index().rename(columns={"index": ""})

        for col in ["precision", "recall", "f1-score"]:
            if col in report_df.columns:
                report_df[col] = report_df[col].apply(
                    lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
                )
        if "support" in report_df.columns:
            report_df["support"] = report_df["support"].apply(
                lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else x
            )
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        if used_sample:
            st.caption("Report is based on the stratified sample of the large dataset.")
    else:
        official_report = {
            "": ["0 (Legitimate)", "1 (Fraud)", "accuracy", "macro avg", "weighted avg"],
            "precision": ["0.9993", "0.9597", "", "0.9795", "0.9990"],
            "recall":    ["0.9998", "0.8728", "", "0.9363", "0.9991"],
            "f1-score":  ["0.9995", "0.9142", "0.9991", "0.9568", "0.9990"],
            "support":   ["257834", "1501", "259335", "259335", "259335"]
        }
        st.dataframe(pd.DataFrame(official_report), use_container_width=True, hide_index=True)
        st.caption("Official training set classification report (Tuned XGBoost)")

st.markdown("---")
st.caption(
    "Live metrics are calculated on the full dataset up to 500 k rows. "
    "For larger datasets a stratified sample (≈200 k rows) is used so the page stays responsive."
)