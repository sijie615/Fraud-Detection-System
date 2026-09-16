"""
Feature engineering utilities -- MUST mirror fyp2_fraud_detection_v2.ipynb
exactly, or predictions on new data will be inconsistent with the model's
training distribution. If you change the notebook's preprocessing, update
this file to match.
"""
import sys
import sklearn.compose._column_transformer

# Compatibility Patch for scikit-learn version mismatch
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class DummyRemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = DummyRemainderColsList


import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

ARTIFACT_DIR = "models"

# FIX 1: target_encoder.pkl's category_encoders.TargetEncoder cannot be
# transformed with ANY currently-installable category_encoders version --
# its internal OrdinalEncoder was pickled without an `index_start` attribute
# that every current version's _transform() expects. This isn't a version
# to pin around; the pickle itself is permanently incompatible. Fix: use the
# plain {state: target_mean} mapping extracted once from the fitted encoder
# (see state_target_encoding.json), bypassing category_encoders at
# inference time entirely.
with open(f"{ARTIFACT_DIR}/state_target_encoding.json") as f:
    _STATE_ENCODING = json.load(f)


def target_encode_state(state_series):
    """Drop-in replacement for target_encoder.transform(df['state'])."""
    return state_series.map(_STATE_ENCODING["mapping"]).fillna(_STATE_ENCODING["fallback"])


@st.cache_resource
def load_artifacts():
    """Load the trained model + preprocessing pipeline once per session."""
    model = joblib.load(f"{ARTIFACT_DIR}/final_fraud_model.pkl")
    preprocessor = joblib.load(f"{ARTIFACT_DIR}/preprocessor.pkl")
    return model, preprocessor


def calculate_haversine(df):
    lat1, lon1 = np.radians(df['lat']), np.radians(df['long'])
    lat2, lon2 = np.radians(df['merch_lat']), np.radians(df['merch_long'])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6371


def engineer_features(df_raw):
    """
    Takes raw transaction data (same columns as fraudTrain.csv) and produces
    the exact feature set the model was trained on. Mirrors the notebook's
    Sections 1.5-1.6 (feature extraction + cyclical encoding).
    """
    df = df_raw.copy()

    # Datetime parsing
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])

    # Age
    df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year

    # Distance
    df['distance_km'] = calculate_haversine(df)

    # Temporal components (before cyclical transform)
    hour = df['trans_date_trans_time'].dt.hour
    day_of_week = df['trans_date_trans_time'].dt.dayofweek
    month = df['trans_date_trans_time'].dt.month
    df['day'] = df['trans_date_trans_time'].dt.day

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    # amt log-transform (critical fix from Chapter 3.4.2 -- never delete rows)
    df['amt_log'] = np.log1p(df['amt'])

    # gender label encoding (2 categories, safe)
    df['gender_encoded'] = (df['gender'] == 'M').astype(int)

    return df


def prepare_model_input(df_raw, preprocessor):
    """
    Full pipeline: raw transaction data -> model-ready feature matrix.
    Returns (X_model_ready, df_engineered) so the caller can display both
    the engineered features and the raw transaction alongside predictions.
    """
    df_eng = engineer_features(df_raw)

    # FIX 2: numeric column order must come from the FITTED preprocessor,
    # not a hand-typed list. A hardcoded numeric_cols list here previously
    # went out of sync with the ColumnTransformer's actual fitted order
    # (which is independent of what order you pass columns INTO .transform()
    # -- it always outputs in its own fitted order). That mismatch didn't
    # crash silently: XGBoost's strict feature-name validation caught it as
    # a ValueError, but the underlying bug was real -- the DataFrame's
    # column LABELS didn't match its VALUES. Reading the order directly
    # from the preprocessor makes this impossible to get out of sync again.
    numeric_cols = list(preprocessor.transformers_[0][2])
    onehot_cols = ['category']

    X_arr = preprocessor.transform(df_eng[numeric_cols + onehot_cols])
    feature_names = numeric_cols + list(
        preprocessor.named_transformers_['cat'].get_feature_names_out(onehot_cols)
    )
    X = pd.DataFrame(X_arr, columns=feature_names, index=df_eng.index)

    # Target-encoded state (see Fix 1 above)
    X['state_encoded'] = target_encode_state(df_eng['state'])

    return X, df_eng


def predict_with_threshold(model, X, threshold=0.9388):
    """Score transactions and apply the tuned decision threshold (Chapter 5.4)."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred
