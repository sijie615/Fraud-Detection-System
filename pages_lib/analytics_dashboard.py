import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.styling import apply_custom_style
from utils.db import set_active_dataset, get_latest_dataset, get_dataset_stats, get_risk_level
from utils.genai import ask_analyst_groq, generate_chart_insight_groq

apply_custom_style()

st.title("📊 Analytics Dashboard")
st.caption("Class distribution and feature-level insights for the uploaded/scored dataset.")

# -----------------------------------------------------------------------------
# 1. Dynamic Dataset Detection & Upload Handling
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload new dataset for Analytics (CSV):", type=["csv"], key="analytics_csv_uploader")

if uploaded_file is not None:
    new_df = pd.read_csv(uploaded_file)
    set_active_dataset(new_df, source_name=uploaded_file.name)
    st.success(f"✅ Loaded new dataset: `{uploaded_file.name}` ({len(new_df)} records)")

# Retrieve the latest active dataset across all pages
df = get_latest_dataset()

if df.empty:
    st.warning("No dataset found. Please upload a CSV file above or score transactions on the Prediction page.")
    st.stop()

# -----------------------------------------------------------------------------
# Standardize Columns and Ensure 'predicted_is_suspicious' ALWAYS Exists
# -----------------------------------------------------------------------------
if "amt" not in df.columns and "amount" in df.columns:
    df["amt"] = df["amount"]
elif "transaction_amount" in df.columns and "amt" not in df.columns:
    df["amt"] = df["transaction_amount"]

if "fraud_probability" not in df.columns and "is_fraud" in df.columns:
    df["fraud_probability"] = df["is_fraud"].astype(float)

if "risk_level" not in df.columns:
    if "fraud_probability" in df.columns:
        df["risk_level"] = df["fraud_probability"].apply(get_risk_level)
    else:
        df["risk_level"] = "Green"

# Safe derivation of predicted_is_suspicious to match Full Chart Insight view completely
if "predicted_is_suspicious" not in df.columns:
    if "is_suspicious" in df.columns:
        df["predicted_is_suspicious"] = df["is_suspicious"].astype(int)
    elif "risk_level" in df.columns:
        df["predicted_is_suspicious"] = df["risk_level"].isin(["Red", "Orange", "Yellow", "Critical Risk", "High Risk", "Moderate Risk"]).astype(int)
    elif "fraud_probability" in df.columns:
        df["predicted_is_suspicious"] = (df["fraud_probability"] >= 0.30).astype(int)
    else:
        df["predicted_is_suspicious"] = 0

dataset_name = st.session_state.get("active_dataset_name", "Active Dataset")
st.caption(f"Currently inspecting: **{dataset_name}** ({len(df)} records)")

color_map = {
    "Green": "#4caf50",
    "Yellow": "#ffd60a",
    "Orange": "#ff9800",
    "Red": "#ff4d4d"
}

# -----------------------------------------------------------------------------
# 2. Overall Class Distribution
# -----------------------------------------------------------------------------
st.subheader("Overall Class Distribution")
risk_counts = df["risk_level"].value_counts().reset_index()
risk_counts.columns = ["risk_level", "count"]

fig_pie = px.pie(
    risk_counts,
    values="count",
    names="risk_level",
    hole=0.55,
    color="risk_level",
    category_orders={"risk_level": ["Green", "Yellow", "Orange", "Red"]},
    color_discrete_map=color_map
)
fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------------------------------------------------------
# 3. Numerical Columns Distribution & "Insight Behind the Curve"
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Class Distribution by Numerical Columns")

candidate_nums = [c for c in ["amt", "city_pop", "transaction_amount", "amount"] if c in df.columns]
num_col = st.selectbox("Select a numerical column", candidate_nums, index=0)

fig_hist = px.histogram(
    df,
    x=num_col,
    color="risk_level",
    barmode="overlay",
    category_orders={"risk_level": ["Green", "Yellow", "Orange", "Red"]},
    color_discrete_map=color_map,
    opacity=0.75,
    nbins=40
)
fig_hist.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20))
st.plotly_chart(fig_hist, use_container_width=True)
st.caption(f"Insight: shows how `{num_col}` values differ across risk levels — rightward skew in Red/Orange bars indicates discriminating signal.")

# Standardized stats
norm_mean, norm_median, susp_mean, susp_median, dominance_threshold = get_dataset_stats(df, num_col)

data_hash = f"{len(df)}_{int(norm_mean)}_{int(susp_mean)}_{int(norm_median)}_{int(susp_median)}_{num_col}"
ai_curve_key = f"curve_insight_{data_hash}"

with st.expander("💡 Insight Behind the Curve", expanded=True):
    st.markdown("### 🧠 Interpretation from Groq AI")

    if ai_curve_key not in st.session_state:
        stats_summary = (
            f"Feature: '{num_col}'. Non-suspicious (Class 0 / Normal) Mean: ${norm_mean:,.2f}, Median: ${norm_median:,.2f}. "
            f"Suspicious (Class 1 / Flagged) Mean: ${susp_mean:,.2f}, Median: ${susp_median:,.2f}. "
            f"Suspicious transactions heavily dominate the upper tail above ${dominance_threshold:,.2f}."
        )
        with st.spinner("Generating AI curve interpretation for new dataset..."):
            st.session_state[ai_curve_key] = generate_chart_insight_groq(num_col, stats_summary)

    st.write(st.session_state[ai_curve_key])
    st.caption("*Note: AI-generated content may be inaccurate.*")

    col_btn1, col_btn2 = st.columns([1.8, 3.2])
    show_full_key = f"show_full_{data_hash}"
    with col_btn1:
        if st.button("🔎 See Full Analysis", key=f"btn_full_{data_hash}", use_container_width=True):
            st.session_state[show_full_key] = not st.session_state.get(show_full_key, False)

    with col_btn2:
        if st.button("🔄 Refresh AI Insight", key=f"btn_ref_{data_hash}", use_container_width=True):
            st.session_state.pop(ai_curve_key, None)
            st.rerun()

    if st.session_state.get(show_full_key, False):
        st.markdown(f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-top: 10px;">
            <b>Full Statistical Breakdown ({num_col}):</b><br><br>
            <b>1. Mean:</b> The mean for non-suspicious transactions (Class 0) is <b>${norm_mean:,.2f}</b> compared to suspicious transactions (Class 1) with a mean of <b>${susp_mean:,.2f}</b>.<br><br>
            <b>2. Median & Central Tendency:</b> Non-suspicious baseline median is <b>${norm_median:,.2f}</b> vs. suspicious median of <b>${susp_median:,.2f}</b>.<br><br>
            <b>3. Suspicious Dominance:</b> Suspicious transactions exhibit heavy density concentration above <b>${dominance_threshold:,.2f}</b>, dominating the higher-value band.
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Categorical Columns Distribution & Synchronized Layout
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Class Distribution by Categorical Columns")

cat_choices = [c for c in ["category", "state", "gender", "job"] if c in df.columns]
if cat_choices:
    selected_cat = st.selectbox("Select a categorical column", cat_choices)

    # Fully synchronized calculation matching Full Chart Insight logic
    cat_summary = df.groupby(selected_cat).agg(
        total=(selected_cat, "count"),
        suspicious=("predicted_is_suspicious", "sum")
    ).reset_index()
    
    cat_summary["suspicious_rate"] = (cat_summary["suspicious"] / cat_summary["total"]) * 100
    cat_summary = cat_summary.sort_values(by="suspicious_rate", ascending=False).reset_index(drop=True)

    fig_cat = px.bar(
        cat_summary,
        x=selected_cat,
        y="suspicious_rate",
        color="suspicious_rate",
        color_continuous_scale=["#4caf50", "#ff9800", "#ff4d4d"],
        labels={"suspicious_rate": "Suspicious Rate (%)"}
    )
    fig_cat.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_cat, use_container_width=True)

    with st.expander("💡 Insight Behind the Chart", expanded=True):
        col_c1, col_c2 = st.columns(2)
        half = int(np.ceil(len(cat_summary) / 2))
        left_items = cat_summary.iloc[:half]
        right_items = cat_summary.iloc[half:]

        with col_c1:
            for _, r in left_items.iterrows():
                name = str(r[selected_cat])
                rate = r["suspicious_rate"]
                icon = "🚨" if rate > 10 else "🚩" if rate > 0 else "🟢"
                st.markdown(f"- `{name}` $\\rightarrow$ {icon} **{rate:.1f}%** suspicious", unsafe_allow_html=True)

        with col_c2:
            for _, r in right_items.iterrows():
                name = str(r[selected_cat])
                rate = r["suspicious_rate"]
                icon = "🚨" if rate > 10 else "🚩" if rate > 0 else "🟢"
                st.markdown(f"- `{name}` $\\rightarrow$ {icon} **{rate:.1f}%** suspicious", unsafe_allow_html=True)

        high_risk_cats = cat_summary[cat_summary["suspicious_rate"] >= 20.0]

        if not high_risk_cats.empty:
            high_names = ", ".join([f"'{row[selected_cat]}' ({row['suspicious_rate']:.1f}%)" for _, row in high_risk_cats.iterrows()])
            st.markdown(f"""
            <div style="background-color: #fefce8; border-left: 4px solid #eab308; border-radius: 6px; padding: 12px; margin-top: 12px; color: #854d0e;">
                ⚠️ The following <b>{selected_cat}</b> categories show a moderate to high suspicious rate and should be monitored closely: {high_names}.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 6px; padding: 12px; margin-top: 12px; color: #166534;">
                ✅ No categories stand out with an abnormally strong suspicious pattern based on <b>{selected_cat}</b>.
            </div>
            """, unsafe_allow_html=True)