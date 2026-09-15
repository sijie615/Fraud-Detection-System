# pages_lib/upload_data.py -- Figures 5.10.4-5.10.5: Upload page (Browse
# files, Preview of uploaded data)
import streamlit as st
import pandas as pd

st.title("📤 Upload Data")
st.caption("Upload a batch of transactions to score on the Prediction page.")

# --- Dataset Guidance Section for Users ---
with st.expander("📖 View Detailed Dataset Upload Guidance & Requirements"):
    st.markdown("""
    To ensure successful batch processing and automated feature engineering, please review the following data guidelines:
    
    * **File Format:** Strictly CSV format (`.csv`).
    * **Raw Data Policy:** Upload uncleaned/raw feature sets. Transformations like cyclical time, scaling, and distance calculations happen automatically on the prediction page.
    * **Missing Data:** Ensure there are **no null or missing values** in any of the required raw columns.
    
    #### **Column Schema Reference:**
    * `trans_date_trans_time` (Timestamp of transaction)
    * `dob` (Date of birth of cardholder)
    * `lat` & `long` (Cardholder location coordinates)
    * `merch_lat` & `merch_long` (Merchant location coordinates)
    * `amt` (Transaction amount)
    * `city_pop` (Population of the city)
    * `gender` (Cardholder gender)
    * `category` (Transaction category type)
    * `state` (State location abbreviation/name)
    """)

uploaded_file = st.file_uploader("Browse files", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.session_state["uploaded_df"] = df_raw.copy()
    st.session_state["raw_uploaded_df"] = df_raw.copy()
    # A new upload starts a new prediction batch, while previously stored cases
    # remain in SQLite as historical case records.
    for key in ["prediction_results", "prediction_X", "prediction_engineered_df", "active_analytics_df", "scored_df", "case_shap_cache"]:
        st.session_state.pop(key, None)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.caption(f"{len(df_raw):,} row(s), {len(df_raw.columns)} column(s) uploaded.")

    required = ["trans_date_trans_time", "dob", "lat", "long", "merch_lat",
                "merch_long", "amt", "city_pop", "gender", "category", "state"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}")
    else:
        st.success("✅ All required columns present. Go to **Prediction** to run the model.")
