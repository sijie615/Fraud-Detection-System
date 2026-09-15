import os
import json
import sqlite3
import pandas as pd
import streamlit as st

DB_PATH = os.path.join("models", "fraud_cases.db")

RISK_COLORS = {
    "Red": "#ff4d4d",
    "Orange": "#ff9800",
    "Yellow": "#ffd60a",
    "Green": "#4caf50"
}


def get_connection():
    """Establishes connection to the local SQLite case database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_columns(conn):
    """Adds the newer analyst-review fields without breaking an older database."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(fraud_cases)").fetchall()}
    additions = {
        "prediction_row_id": "INTEGER",
        "case_key": "TEXT",
        "transaction_json": "TEXT",
        "shap_analysis": "TEXT",
        "recommendation": "TEXT",
        "analyst_decision": "TEXT",
        "analyst_notes": "TEXT",
        "analyst_name": "TEXT",
        "reviewed_at": "TEXT",
    }
    for column, definition in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE fraud_cases ADD COLUMN {column} {definition}")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_fraud_cases_case_key ON fraud_cases(case_key)")


def init_db():
    """Initializes the persistent case table and performs a safe schema migration."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fraud_cases (
            case_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            amt REAL,
            category TEXT,
            fraud_probability REAL,
            risk_level TEXT,
            top_feature TEXT,
            timestamp TEXT,
            status TEXT DEFAULT 'Pending Review',
            notes TEXT,
            investigator TEXT,
            reviewer TEXT,
            ai_recommendation TEXT
        )
    """)
    _ensure_columns(conn)
    conn.commit()
    conn.close()


def get_risk_level(prob: float) -> str:
    """Maps prediction probability to 4 operational risk tiers."""
    try:
        val = float(prob)
    except (ValueError, TypeError):
        return "Green"

    if val >= 0.80:
        return "Red"
    elif val >= 0.50:
        return "Orange"
    elif val >= 0.30:
        return "Yellow"
    else:
        return "Green"


def get_all_cases() -> pd.DataFrame:
    """Retrieves all persistent cases."""
    init_db()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM fraud_cases ORDER BY case_id DESC", conn)
    conn.close()
    return df


def get_cases_by_status(status: str, investigator: str = None, reviewer: str = None, *args, **kwargs) -> pd.DataFrame:
    """Retrieves cases filtered by status with optional legacy filters."""
    init_db()
    conn = get_connection()
    query = "SELECT * FROM fraud_cases WHERE status = ?"
    params = [status]

    if investigator:
        query += " AND (investigator = ? OR investigator IS NULL OR investigator = '')"
        params.append(str(investigator))

    if reviewer:
        query += " AND (reviewer = ? OR reviewer IS NULL OR reviewer = '')"
        params.append(str(reviewer))

    query += " ORDER BY case_id DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def add_case(**kwargs):
    """Inserts a persistent case record, ignoring duplicate case keys."""
    init_db()
    conn = get_connection()

    transaction_id = str(kwargs.get("transaction_id", ""))
    case_key = str(kwargs.get("case_key", transaction_id)).strip()
    if not case_key:
        case_key = f"row_{kwargs.get('prediction_row_id', '')}_{kwargs.get('timestamp', '')}"

    transaction_json = kwargs.get("transaction_json", "")
    if isinstance(transaction_json, dict):
        transaction_json = json.dumps(transaction_json, ensure_ascii=False, default=str)

    conn.execute("""
        INSERT OR IGNORE INTO fraud_cases (
            transaction_id, amt, category, fraud_probability, risk_level,
            top_feature, timestamp, status, notes, investigator, reviewer,
            prediction_row_id, case_key, transaction_json, shap_analysis,
            recommendation, analyst_decision, analyst_notes, analyst_name, reviewed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        transaction_id,
        float(kwargs.get("amt", kwargs.get("amount", 0.0))),
        str(kwargs.get("category", "General")),
        float(kwargs.get("fraud_probability", kwargs.get("fraud_prob", 0.0))),
        str(kwargs.get("risk_level", "Green")),
        str(kwargs.get("top_feature", "")),
        str(kwargs.get("timestamp", "N/A")),
        str(kwargs.get("status", "Pending Review")),
        str(kwargs.get("notes", "")),
        str(kwargs.get("investigator", "")),
        str(kwargs.get("reviewer", "")),
        kwargs.get("prediction_row_id"),
        case_key,
        str(transaction_json),
        str(kwargs.get("shap_analysis", "")),
        str(kwargs.get("recommendation", "")),
        str(kwargs.get("analyst_decision", "")),
        str(kwargs.get("analyst_notes", "")),
        str(kwargs.get("analyst_name", "")),
        str(kwargs.get("reviewed_at", "")),
    ))
    conn.commit()
    conn.close()


def get_case_by_key(case_key: str):
    init_db()
    conn = get_connection()
    row = conn.execute("SELECT * FROM fraud_cases WHERE case_key = ? LIMIT 1", (str(case_key),)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_case_analysis(case_key: str, shap_analysis: str = None, recommendation: str = None):
    """Persists SHAP analysis and/or the recommendation shown to the analyst."""
    init_db()
    conn = get_connection()
    fields, values = [], []
    if shap_analysis is not None:
        fields.append("shap_analysis = ?")
        values.append(str(shap_analysis))
    if recommendation is not None:
        fields.append("recommendation = ?")
        values.append(str(recommendation))
        fields.append("ai_recommendation = ?")
        values.append(str(recommendation))
    if fields:
        values.append(str(case_key))
        conn.execute(f"UPDATE fraud_cases SET {', '.join(fields)} WHERE case_key = ?", values)
        conn.commit()
    conn.close()


def update_analyst_review(case_key: str, decision: str, notes: str, analyst_name: str):
    """Persists the Fraud Analyst's final verification decision."""
    init_db()
    conn = get_connection()
    status = "Verified - Suspicious" if decision == "Suspicious" else "Verified - Not a Fraud Case"
    conn.execute("""
        UPDATE fraud_cases
        SET analyst_decision = ?, analyst_notes = ?, analyst_name = ?,
            reviewed_at = datetime('now'), status = ?, notes = ?, reviewer = ?
        WHERE case_key = ?
    """, (
        str(decision), str(notes), str(analyst_name), status,
        str(notes), str(analyst_name), str(case_key)
    ))
    conn.commit()
    conn.close()


def report_case(case_id: int, notes: str, investigator: str = ""):
    """Legacy helper retained for compatibility with older pages."""
    init_db()
    conn = get_connection()
    conn.execute(
        "UPDATE fraud_cases SET status = 'Reported', notes = ?, investigator = ? WHERE case_id = ?",
        (str(notes), str(investigator), int(case_id))
    )
    conn.commit()
    conn.close()


def verify_case(case_id: int, notes: str, reviewer: str = ""):
    """Legacy helper retained for compatibility with older pages."""
    init_db()
    conn = get_connection()
    conn.execute(
        "UPDATE fraud_cases SET status = 'Verified', notes = ?, reviewer = ? WHERE case_id = ?",
        (str(notes), str(reviewer), int(case_id))
    )
    conn.commit()
    conn.close()


def update_ai_recommendation(case_id: int, recommendation: str):
    """Legacy helper retained for compatibility with older pages."""
    init_db()
    conn = get_connection()
    conn.execute(
        "UPDATE fraud_cases SET ai_recommendation = ?, recommendation = ? WHERE case_id = ?",
        (str(recommendation), str(recommendation), int(case_id))
    )
    conn.commit()
    conn.close()


def get_dataset_stats(df: pd.DataFrame, col: str = "amt"):
    """Standardizes metric calculations so Analytics and Full Chart Insight produce identical values."""
    if df.empty or col not in df.columns:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    if "risk_level" in df.columns:
        susp_mask = df["risk_level"].isin(["Red", "Orange", "Yellow", "Critical Risk", "High Risk", "Moderate Risk"])
    elif "fraud_probability" in df.columns:
        susp_mask = df["fraud_probability"] >= 0.30
    elif "is_suspicious" in df.columns:
        susp_mask = df["is_suspicious"] == 1
    else:
        susp_mask = pd.Series([False] * len(df))

    norm_data = df[~susp_mask][col].dropna()
    susp_data = df[susp_mask][col].dropna()

    norm_mean = float(norm_data.mean()) if len(norm_data) > 0 else 0.0
    norm_median = float(norm_data.median()) if len(norm_data) > 0 else 0.0
    susp_mean = float(susp_data.mean()) if len(susp_data) > 0 else 0.0
    susp_median = float(susp_data.median()) if len(susp_data) > 0 else 0.0

    dominance_threshold = float(norm_median * 2.5) if norm_median > 0 else 300.0
    return norm_mean, norm_median, susp_mean, susp_median, dominance_threshold


def set_active_dataset(new_df: pd.DataFrame, source_name: str = "Uploaded Dataset"):
    """Synchronizes active dataset across session states and purges old insight caches."""
    clean_df = new_df.copy()

    if "amt" not in clean_df.columns and "amount" in clean_df.columns:
        clean_df["amt"] = clean_df["amount"]
    elif "transaction_amount" in clean_df.columns and "amt" not in clean_df.columns:
        clean_df["amt"] = clean_df["transaction_amount"]

    if "fraud_probability" not in clean_df.columns and "is_fraud" in clean_df.columns:
        clean_df["fraud_probability"] = clean_df["is_fraud"].astype(float)

    if "risk_level" not in clean_df.columns:
        if "fraud_probability" in clean_df.columns:
            clean_df["risk_level"] = clean_df["fraud_probability"].apply(get_risk_level)
        else:
            clean_df["risk_level"] = "Green"

    if "fraud_probability" in clean_df.columns:
        clean_df["is_suspicious"] = (clean_df["fraud_probability"] >= 0.30).astype(int)
    else:
        clean_df["is_suspicious"] = clean_df["risk_level"].isin(["Red", "Orange", "Yellow", "Critical Risk", "High Risk"]).astype(int)

    st.session_state["active_analytics_df"] = clean_df
    st.session_state["scored_df"] = clean_df
    st.session_state["uploaded_df"] = clean_df
    st.session_state["prediction_results"] = clean_df
    st.session_state["active_dataset_name"] = source_name

    for k in list(st.session_state.keys()):
        if any(k.startswith(p) for p in ["curve_insight_", "full_interp_", "full_chart_num_ai_", "analytics_"]):
            del st.session_state[k]


def get_latest_dataset() -> pd.DataFrame:
    """Retrieves the latest dataset available across session state and database."""
    for k in ["active_analytics_df", "scored_df", "prediction_results", "uploaded_df"]:
        if k in st.session_state and isinstance(st.session_state[k], pd.DataFrame) and not st.session_state[k].empty:
            return st.session_state[k].copy()

    all_cases = get_all_cases()
    if not all_cases.empty:
        df = all_cases.copy()
        if "amt" not in df.columns and "amount" in df.columns:
            df["amt"] = df["amount"]
        if "is_suspicious" not in df.columns:
            df["is_suspicious"] = df["risk_level"].isin(["Red", "Orange", "Yellow"]).astype(int)
        return df

    return pd.DataFrame()
