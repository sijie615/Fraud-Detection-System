# pages_lib/explainability.py -- Global XAI page (Admin). Not in the
# reference report's page list (her XAI is embedded in Prediction/Case
# Management too, same as this dashboard) -- added here as a genuine
# improvement, since "what does the model rely on OVERALL" is a different,
# useful question from "why did THIS transaction get flagged" (which
# Prediction/Case Management already answer).
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.title("🧠 Explainability (Global)")
st.caption("What the model relies on across ALL transactions -- complements the per-transaction "
           "SHAP explanation on the Prediction page.")

try:
    shap_explainer = joblib.load("models/shap_explainer.pkl")
    shap_values = np.load("models/shap_values.npy")
    shap_sample = pd.read_csv("models/shap_sample_transactions.csv")
except FileNotFoundError as e:
    st.error(f"Could not load explainability artifacts: {e}. Make sure `shap_explainer.pkl`, "
             f"`shap_values.npy`, and `shap_sample_transactions.csv` are all in `models/`.")
    st.stop()

tab1, tab2 = st.tabs(["Feature Importance", "Summary Plot (Beeswarm)"])

with tab1:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": shap_sample.columns, "Mean |SHAP value|": mean_abs_shap
    }).sort_values("Mean |SHAP value|", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["Feature"][::-1], importance_df["Mean |SHAP value|"][::-1], color="#4c72b0")
    ax.set_xlabel("Mean |SHAP value| (average impact on model output)")
    ax.set_title("Top 15 Features by Global Importance")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Ranks features by their average impact on the model's output across all sampled "
               "transactions -- tells you WHICH features matter most overall, not which direction "
               "each pushes a prediction.")

with tab2:
    fig2 = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, shap_sample, show=False)
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption("Each dot is one transaction. Position shows whether that feature pushed the "
               "prediction toward fraud (right) or legitimate (left); colour shows whether the "
               "feature's value was high (red) or low (blue) for that transaction.")
