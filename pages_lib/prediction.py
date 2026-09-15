# pages_lib/prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import json
import hashlib
import joblib
from lime.lime_tabular import LimeTabularExplainer

from utils.preprocessing import load_artifacts, prepare_model_input, predict_with_threshold
from utils.db import (
    get_risk_level,
    RISK_COLORS,
    set_active_dataset,
    add_case,
)
from utils.genai import ask_analyst_groq

st.title("🔍 Prediction")
st.caption("Apply the saved preprocessing pipeline, trained XGBoost model, and explainability frameworks to the uploaded dataset.")

if "uploaded_df" not in st.session_state:
    st.warning("No data uploaded yet. Go to **Upload Data** first.")
    st.stop()

THRESHOLD = 0.9388

# Keep a clean copy of the raw upload
df_raw = st.session_state.get("raw_uploaded_df", st.session_state["uploaded_df"]).copy()
st.session_state["raw_uploaded_df"] = df_raw.copy()

st.info(
    f"The saved XGBoost decision threshold is **{THRESHOLD:.4f}**. "
    "All rows are scored. High-risk cases + a sample of low-risk cases are sent to the Fraud Analyst queues."
)

def make_case_key(row):
    preferred = []
    for col in ["trans_num", "transaction_id", "trans_date_trans_time", "amt", "merchant", "category"]:
        if col in row.index:
            preferred.append(f"{col}={row[col]}")
    return hashlib.sha1("|".join(preferred).encode("utf-8", errors="ignore")).hexdigest()

def json_safe_record(row):
    return {str(k): (None if pd.isna(v) else str(v)) for k, v in row.items()}

def extract_shap_vector(values):
    arr = np.asarray(values)
    if isinstance(values, list):
        arr = np.asarray(values[-1])
    if arr.ndim == 3:
        arr = arr[0, :, -1]
    elif arr.ndim == 2:
        arr = arr[0]
    return arr.reshape(-1)

def build_shap_text(shap_explainer, X_display, result_row):
    idx = int(result_row.name)
    shap_values = shap_explainer.shap_values(X_display.loc[[idx]])
    vector = extract_shap_vector(shap_values)
    feature_names = list(X_display.columns)
    pairs = []
    for i in np.argsort(np.abs(vector))[::-1][:5]:
        if i >= len(feature_names):
            continue
        direction = "increases fraud risk" if vector[i] > 0 else "decreases fraud risk"
        pairs.append(f"{feature_names[i]}: {vector[i]:+.4f} ({direction})")
    return "; ".join(pairs), vector

# =============================================================================
# RUN PREDICTION
# =============================================================================
if st.button("🚀 Apply Data Preprocessing Pipeline & Run Prediction", type="primary"):
    try:
        with st.spinner("Loading saved model and applying preprocessing..."):
            model, preprocessor = load_artifacts()
            X, df_eng = prepare_model_input(df_raw, preprocessor)
            y_prob, y_pred = predict_with_threshold(model, X, THRESHOLD)

            results = df_raw.copy()
            results["prediction_row_id"] = np.arange(len(results))
            results["fraud_probability"] = np.round(y_prob, 4)
            results["suspicious_score"] = np.round(y_prob, 4)
            results["predicted_is_suspicious"] = y_pred.astype(int)
            results["prediction_label"] = np.where(y_pred == 1, "Suspicious", "Not Flagged")
            results["risk_level"] = [get_risk_level(p) for p in y_prob]
            results["alert"] = np.where(y_pred == 1, "🚨", "")

            results = results.sort_values(by="fraud_probability", ascending=False).reset_index(drop=True)

            X_display = X.copy()
            X_display["_original_row"] = np.arange(len(X_display))
            X_display = X_display.iloc[results["prediction_row_id"].astype(int).to_numpy()].reset_index(drop=True)
            X_display = X_display.drop(columns=["_original_row"], errors="ignore")
            X_display.index = results.index

            st.session_state["prediction_results"] = results
            st.session_state["prediction_X"] = X_display
            st.session_state["prediction_engineered_df"] = df_eng
            st.session_state["raw_uploaded_df"] = df_raw.copy()

            # =================================================================
            # TWO-QUEUE LOGIC
            # =================================================================
            # 1. High-Risk cases (always send)
            high_risk = results[results["risk_level"].isin(
                ["Red", "Orange", "Critical Risk", "High Risk"]
            )].copy()

            # 2. Low-Risk cases (sample only)
            low_risk_all = results[results["risk_level"].isin(
                ["Yellow", "Green", "Low Risk", "Normal", "Moderate Risk"]
            )].copy()

            sample_rate = 0.08  # 8% of low-risk cases
            if len(low_risk_all) > 0:
                low_risk_sample = low_risk_all.sample(
                    frac=min(sample_rate, 1.0),
                    random_state=42
                )
            else:
                low_risk_sample = low_risk_all.copy()

            cases_to_send = pd.concat([high_risk, low_risk_sample], ignore_index=True)

            shap_cache = {}
            sent_count = 0

            try:
                shap_explainer = joblib.load("models/shap_explainer.pkl")
            except Exception:
                shap_explainer = None

            for result_idx, row in cases_to_send.iterrows():
                try:
                    text = ""
                    top_feature = ""
                    if shap_explainer is not None:
                        try:
                            text, vector = build_shap_text(shap_explainer, X_display, row)
                            shap_cache[int(result_idx)] = text
                            top_feature = X_display.columns[int(np.argmax(np.abs(vector)))] if len(vector) else ""
                        except Exception:
                            text = "SHAP analysis could not be generated for this transaction."
                            top_feature = ""

                    transaction_id = str(row.get("trans_num", row.get("transaction_id", f"Row {int(result_idx)}")))
                    case_key = make_case_key(row)

                    add_case(
                        transaction_id=transaction_id,
                        amt=row.get("amt", row.get("amount", 0)),
                        category=row.get("category", "General"),
                        fraud_probability=row.get("fraud_probability", 0),
                        risk_level=row.get("risk_level", "Green"),
                        top_feature=top_feature,
                        timestamp=row.get("trans_date_trans_time", "N/A"),
                        status="Pending Review",
                        prediction_row_id=int(row.get("prediction_row_id", 0)),
                        case_key=case_key,
                        transaction_json=json_safe_record(row),
                        shap_analysis=text,
                    )
                    sent_count += 1
                except Exception:
                    continue

            st.session_state["case_shap_cache"] = shap_cache
            set_active_dataset(results, source_name="Prediction Batch Results")

        st.success(
            f"Prediction completed for **{len(results):,}** transaction(s).\n\n"
            f"- High-Risk cases sent: **{len(high_risk):,}**\n"
            f"- Low-Risk sample sent: **{len(low_risk_sample):,}**\n"
            f"- Total cases sent to Fraud Analyst: **{sent_count:,}**"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

if "prediction_results" not in st.session_state:
    st.stop()

results = st.session_state["prediction_results"]

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
st.markdown("---")
st.subheader("Prediction Results")

m1, m2, m3, m4, m5 = st.columns(5)
total_cnt = len(results)
risk = results["risk_level"]
red_cnt = int(risk.isin(["Red", "Critical Risk"]).sum())
orange_cnt = int(risk.isin(["Orange", "High Risk"]).sum())
yellow_cnt = int(risk.isin(["Yellow", "Moderate Risk", "Low-Medium"]).sum())
green_cnt = int(risk.isin(["Green", "Low Risk", "Normal"]).sum())
review_cnt = int(pd.to_numeric(results.get("predicted_is_suspicious", 0), errors="coerce").fillna(0).sum())

m1.metric("Total Transactions", f"{total_cnt:,}")
m2.metric("High-Risk (Review)", f"{red_cnt + orange_cnt:,}")
m3.metric("Critical (Red)", f"{red_cnt:,}")
m4.metric("Medium (Orange)", f"{orange_cnt:,}")
m5.metric("Low / Normal", f"{yellow_cnt + green_cnt:,}")

page_col, rows_col, info_col = st.columns([1, 1, 3])
with rows_col:
    rows_choice = st.selectbox("Rows to display", [25, 50, 100, 250, 500, 1000, "All"], index=2)

if rows_choice == "All":
    page_size = total_cnt
else:
    page_size = int(rows_choice)

total_pages = max(1, int(np.ceil(total_cnt / page_size)))
with page_col:
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
with info_col:
    st.write("")
    if rows_choice == "All":
        st.caption(f"Displaying all {total_cnt:,} records.")
    else:
        st.caption(f"Displaying page {page_num} of {total_pages} — {total_cnt:,} total records.")

start_idx = (page_num - 1) * page_size
end_idx = min(start_idx + page_size, total_cnt)
page_subset = results.iloc[start_idx:end_idx].copy()

def highlight_risk(row):
    color = RISK_COLORS.get(row.get("risk_level"), "#ffffff")
    return [f"background-color: {color}33"] * len(row)

st.dataframe(
    page_subset.style.apply(highlight_risk, axis=1).format({
        "fraud_probability": "{:.4f}",
        "suspicious_score": "{:.4f}"
    }),
    use_container_width=True,
    height=500,
)

# =============================================================================
# EXPLAINABILITY SECTION (kept from your original)
# =============================================================================
st.markdown("---")
st.markdown("### 🧠 Explainability: SHAP & LIME Analysis")

selected_row_idx = st.selectbox(
    "Select a transaction from the displayed rows",
    options=page_subset.index.tolist(),
)

curr_row = results.loc[selected_row_idx]
curr_prob = float(curr_row.get("fraud_probability", 0.0))
curr_risk = str(curr_row.get("risk_level", "Green"))
curr_amt = float(curr_row.get("amt", curr_row.get("transaction_amount", 0.0)))
curr_cat = str(curr_row.get("category", "General"))

col_shap, col_lime, col_insight = st.columns([1, 1, 1.2], gap="large")
top_shap_summary = []

with col_shap:
    st.markdown("#### SHAP Analysis")
    try:
        shap_explainer = joblib.load("models/shap_explainer.pkl")
        X = st.session_state["prediction_X"]
        shap_values = shap_explainer.shap_values(X.loc[[selected_row_idx]])
        row_shap = extract_shap_vector(shap_values)
        top_idx = np.argsort(np.abs(row_shap))[::-1][:5]
        for i in top_idx:
            if i >= len(X.columns):
                continue
            direction = "increases risk" if row_shap[i] > 0 else "decreases risk"
            st.markdown(f"- `{X.columns[i]}` : **{row_shap[i]:+.4f}** ({direction})")
            top_shap_summary.append(f"{X.columns[i]} ({row_shap[i]:+.4f})")
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")

with col_lime:
    st.markdown("#### LIME Explanation")
    try:
        X = st.session_state["prediction_X"]
        model, _ = load_artifacts()
        lime_explainer = LimeTabularExplainer(
            training_data=np.array(X),
            feature_names=list(X.columns),
            class_names=["Legitimate", "Fraud"],
            mode="classification"
        )
        exp = lime_explainer.explain_instance(
            data_row=X.loc[selected_row_idx].to_numpy(),
            predict_fn=model.predict_proba,
            num_features=5
        )
        for feat_desc, weight in exp.as_list():
            direction = "increases risk" if weight > 0 else "decreases risk"
            st.markdown(f"- `{feat_desc}` : **{weight:+.4f}**")
    except Exception as e:
        st.info(f"LIME explanation unavailable: {e}")

with col_insight:
    st.markdown("#### Insight")
    shap_str = ", ".join(top_shap_summary) if top_shap_summary else "No SHAP values available"
    row_hash = f"{selected_row_idx}_{int(curr_amt)}_{int(curr_prob * 10000)}_{curr_risk}"
    insight_key = f"admin_shap_insight_{row_hash}"
    if insight_key not in st.session_state:
        insight_prompt = f"""
You are a fraud analytics officer.
Interpret this transaction for an administrator using the model result and SHAP/LIME contributors.
Transaction amount: RM {curr_amt:,.2f}
Category: {curr_cat}
Fraud probability: {curr_prob:.4f}
Risk level: {curr_risk}
Top key contributors: {shap_str}
Write one concise paragraph under 80 words explaining the main risk pattern and why the transaction deserves attention. Do not provide operational recommendations and do not claim the model proves fraud.
"""
        with st.spinner("Preparing transaction insight..."):
            st.session_state[insight_key] = ask_analyst_groq([
                {"role": "system", "content": "You write concise fraud analytics insights for administrators."},
                {"role": "user", "content": insight_prompt},
            ])
    st.markdown(
        f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;line-height:1.6;'>"
        f"{st.session_state[insight_key]}</div>",
        unsafe_allow_html=True,
    )
    st.caption("Model-based insight only; it does not independently establish fraud.")

# =============================================================================
# DOWNLOAD + NAVIGATION
# =============================================================================
st.markdown("---")
b1, b2, b3 = st.columns(3)

with b1:
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            results.to_excel(writer, index=False, sheet_name="Predictions")
        st.download_button(
            "⬇️ Download Results (Excel)",
            data=buffer.getvalue(),
            file_name="prediction_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception:
        st.download_button(
            "⬇️ Download Results (CSV)",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="prediction_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

with b2:
    if st.button("📊 Visualize Dashboard", use_container_width=True):
        st.switch_page("pages_lib/analytics_dashboard.py")

with b3:
    if st.button("📤 Upload New Dataset", use_container_width=True):
        for key in [
            "uploaded_df", "raw_uploaded_df", "prediction_results", "prediction_X",
            "prediction_engineered_df", "active_analytics_df", "scored_df", "case_shap_cache"
        ]:
            st.session_state.pop(key, None)
        st.switch_page("pages_lib/upload_data.py")