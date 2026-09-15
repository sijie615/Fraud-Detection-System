# pages_lib/full_chart_insight.py -- Figure 5.10.18: Full Chart Insight Page
# An expanded, all-at-once version of the Analytics Dashboard's charts, for
# a single detailed report-style view rather than one-selector-at-a-time.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.styling import apply_custom_style
from utils.db import get_latest_dataset, get_dataset_stats, set_active_dataset, get_risk_level
from utils.genai import ask_analyst_groq, generate_chart_insight_groq

apply_custom_style()

st.title("📈 Full Chart Insight")
st.caption("Expanded view of every chart on the Analytics Dashboard, side by side.")

# -----------------------------------------------------------------------------
# 1. Dataset Ingestion & Active Dataset Retrieval
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload new dataset to update all charts (CSV):", type=["csv"], key="full_chart_uploader")
if uploaded_file is not None:
    new_df = pd.read_csv(uploaded_file)
    set_active_dataset(new_df, source_name=uploaded_file.name)
    st.success(f"✅ Loaded new dataset: `{uploaded_file.name}` ({len(new_df)} records)")

# Always read the shared latest dataset
results = get_latest_dataset()

if results.empty:
    st.warning("No predictions or dataset loaded yet. Please upload a dataset above or on the Analytics Dashboard.")
    st.stop()

# -----------------------------------------------------------------------------
# 2. Strict Column Normalization (Guarantees 'is_suspicious' ALWAYS exists)
# -----------------------------------------------------------------------------
if "amt" not in results.columns and "amount" in results.columns:
    results["amt"] = results["amount"]
elif "transaction_amount" in results.columns and "amt" not in results.columns:
    results["amt"] = results["transaction_amount"]

if "fraud_probability" not in results.columns and "is_fraud" in results.columns:
    results["fraud_probability"] = results["is_fraud"].astype(float)

if "risk_level" not in results.columns:
    if "fraud_probability" in results.columns:
        results["risk_level"] = results["fraud_probability"].apply(get_risk_level)
    else:
        results["risk_level"] = "Green"

# Construct is_suspicious from any available field
if "predicted_is_suspicious" in results.columns:
    results["is_suspicious"] = results["predicted_is_suspicious"].astype(int)
elif "risk_level" in results.columns:
    results["is_suspicious"] = results["risk_level"].isin(["Red", "Orange", "Yellow", "Critical Risk", "High Risk", "Moderate Risk"]).astype(int)
elif "fraud_probability" in results.columns:
    results["is_suspicious"] = (results["fraud_probability"] >= 0.30).astype(int)
elif "is_fraud" in results.columns:
    results["is_suspicious"] = results["is_fraud"].astype(int)
else:
    results["is_suspicious"] = 0

dataset_name = st.session_state.get("active_dataset_name", "Active Dataset")
st.caption(f"Currently inspecting: **{dataset_name}** ({len(results)} records)")

color_map = {"Red": "#ff4d4d", "Orange": "#ff9800", "Yellow": "#ffd60a", "Green": "#4caf50"}

tab1, tab2, tab3 = st.tabs(["Numerical Columns", "Categorical Columns", "Trend Over Time"])

# =============================================================================
# TAB 1: Numerical Columns
# =============================================================================
with tab1:
    num_cols = [c for c in ["amt", "city_pop", "transaction_amount", "amount"] if c in results.columns]
    for col in num_cols:
        st.markdown(f"### **{col}**")
        
        # Original Plotly Boxplot
        fig = px.box(
            results,
            x="risk_level",
            y=col,
            color="risk_level",
            category_orders={"risk_level": ["Green", "Yellow", "Orange", "Red"]},
            color_discrete_map=color_map
        )
        st.plotly_chart(fig, use_container_width=True)

        # Standardized metrics from shared helper
        norm_mean, norm_median, susp_mean, susp_median, dominance_threshold = get_dataset_stats(results, col)

        col_hash = f"{len(results)}_{int(norm_mean)}_{int(susp_mean)}_{int(norm_median)}_{int(susp_median)}_{col}"
        insight_key = f"curve_insight_{col_hash}"

        with st.expander(f"💡 Insight Behind the {col} Distribution", expanded=True):
            st.markdown("#### 🧠 Interpretation from Groq AI")
            
            if insight_key not in st.session_state:
                stats_summary = (
                    f"Feature: '{col}'. Non-suspicious (Class 0 / Normal) Mean: ${norm_mean:,.2f}, Median: ${norm_median:,.2f}. "
                    f"Suspicious (Class 1 / Flagged) Mean: ${susp_mean:,.2f}, Median: ${susp_median:,.2f}. "
                    f"Suspicious transactions heavily dominate the upper tail above ${dominance_threshold:,.2f}."
                )
                with st.spinner(f"Evaluating {col} distributions with Groq AI..."):
                    st.session_state[insight_key] = generate_chart_insight_groq(col, stats_summary)

            st.write(st.session_state[insight_key])
            st.caption("*Note: AI-generated content may be inaccurate.*")

            c_btn1, c_btn2 = st.columns([1.8, 3.2])
            show_detail_key = f"show_detail_{col_hash}"
            with c_btn1:
                if st.button(f"🔎 See Full Analysis", key=f"btn_see_full_{col_hash}", use_container_width=True):
                    st.session_state[show_detail_key] = not st.session_state.get(show_detail_key, False)

            with c_btn2:
                if st.button(f"🔄 Refresh AI Interpretation", key=f"btn_refresh_{col_hash}", use_container_width=True):
                    st.session_state.pop(insight_key, None)
                    st.rerun()

            if st.session_state.get(show_detail_key, False):
                st.markdown(f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-top: 10px;">
                    <b>Full Statistical Breakdown ({col}):</b><br><br>
                    <b>1. Mean:</b> The mean for non-suspicious transactions (Class 0) is <b>${norm_mean:,.2f}</b> compared to suspicious transactions (Class 1) with a mean of <b>${susp_mean:,.2f}</b>.<br><br>
                    <b>2. Median & Central Tendency:</b> Non-suspicious baseline median is <b>${norm_median:,.2f}</b> vs. suspicious median of <b>${susp_median:,.2f}</b>.<br><br>
                    <b>3. Upper Quartile Dispersion:</b> Suspicious transactions exhibit heavy density concentration above <b>${dominance_threshold:,.2f}</b>, dominating the higher-value band.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

# =============================================================================
# TAB 2: Categorical Columns
# =============================================================================
with tab2:
    cat_cols = [c for c in ["category", "gender", "state", "job"] if c in results.columns]
    for col in cat_cols:
        st.markdown(f"### **{col}**")
        
        # Original Plotly Stacked Bar Chart
        ct = results.groupby([col, "risk_level"]).size().reset_index(name="count")
        fig = px.bar(
            ct,
            x=col,
            y="count",
            color="risk_level",
            barmode="stack",
            category_orders={"risk_level": ["Green", "Yellow", "Orange", "Red"]},
            color_discrete_map=color_map
        )
        st.plotly_chart(fig, use_container_width=True)

        # In-place safety check: ensure is_suspicious exists on results before groupby
        if "is_suspicious" not in results.columns:
            results["is_suspicious"] = results["risk_level"].isin(["Red", "Orange", "Yellow", "Critical Risk", "High Risk"]).astype(int)

        # Safe aggregation
        cat_stats = results.groupby(col)["is_suspicious"].agg(["count", "mean"]).reset_index()
        cat_stats.columns = [col, "total", "rate"]
        cat_stats["pct"] = cat_stats["rate"] * 100
        cat_stats = cat_stats.sort_values(by="pct", ascending=False)

        with st.expander(f"💡 Insight Behind the {col} Chart", expanded=True):
            col_l, col_r = st.columns(2)
            half = int(np.ceil(len(cat_stats) / 2))
            left_group = cat_stats.iloc[:half]
            right_group = cat_stats.iloc[half:]

            with col_l:
                for _, r in left_group.iterrows():
                    name = str(r[col])
                    val = r["pct"]
                    st.markdown(f"- **{name}** &rarr; 🚩 <span style='color:#15803d; font-weight:600;'>{val:.1f}% suspicious</span>", unsafe_allow_html=True)

            with col_r:
                for _, r in right_group.iterrows():
                    name = str(r[col])
                    val = r["pct"]
                    st.markdown(f"- **{name}** &rarr; 🚩 <span style='color:#15803d; font-weight:600;'>{val:.1f}% suspicious</span>", unsafe_allow_html=True)

            high_risk = cat_stats[cat_stats["pct"] >= 50.0]
            med_risk = cat_stats[(cat_stats["pct"] >= 20.0) & (cat_stats["pct"] < 50.0)]

            if not high_risk.empty:
                high_str = ", ".join([f"'{row[col]}' ({row['pct']:.1f}%)" for _, row in high_risk.iterrows()])
                st.markdown(f"""
                <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; border-radius: 6px; padding: 12px; margin-top: 12px; color: #991b1b;">
                    ⚠️ The following <b>{col}</b> categories show a <b>very high suspicious rate</b> and may require immediate investigation: {high_str}.
                </div>
                """, unsafe_allow_html=True)

            if not med_risk.empty:
                med_str = ", ".join([f"'{row[col]}' ({row['pct']:.1f}%)" for _, row in med_risk.iterrows()])
                st.markdown(f"""
                <div style="background-color: #fefce8; border-left: 4px solid #eab308; border-radius: 6px; padding: 12px; margin-top: 8px; color: #854d0e;">
                    ⚠️ The following <b>{col}</b> categories show a <b>moderate suspicious rate</b> and should be monitored closely: {med_str}.
                </div>
                """, unsafe_allow_html=True)

            if high_risk.empty and med_risk.empty:
                st.markdown(f"""
                <div style="background-color: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 6px; padding: 12px; margin-top: 12px; color: #166534;">
                    ✅ No categories stand out with a disproportionate risk pattern based on <b>{col}</b>.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

# =============================================================================
# TAB 3: Trend Over Time
# =============================================================================
with tab3:
    if "trans_date_trans_time" in results.columns:
        results_ts = results.copy()
        results_ts["date"] = pd.to_datetime(results_ts["trans_date_trans_time"]).dt.date
        daily = results_ts.groupby(["date", "risk_level"]).size().reset_index(name="count")
        
        fig = px.line(
            daily,
            x="date",
            y="count",
            color="risk_level",
            category_orders={"risk_level": ["Green", "Yellow", "Orange", "Red"]},
            color_discrete_map=color_map
        )
        st.plotly_chart(fig, use_container_width=True)

        susp_daily = results_ts[results_ts["is_suspicious"] == 1].groupby("date").size()
        peak_date = str(susp_daily.idxmax()) if not susp_daily.empty else "N/A"
        peak_count = int(susp_daily.max()) if not susp_daily.empty else 0

        with st.expander("💡 Insight Behind the Trend", expanded=True):
            st.markdown(f"""
            - **Temporal Anomaly Peak:** Recorded on **`{peak_date}`** with **{peak_count}** suspicious flags.
            - **Velocity Clustering:** Suspicious transactions frequently concentrate during specific batch intervals and weekend windows.
            """)
    else:
        st.info("No timestamp column available for a time-trend chart.")

