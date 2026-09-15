import streamlit as st
from utils.auth import is_logged_in, do_login, do_logout
from utils.styling import apply_custom_style

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_style()

# ==============================================================================
# Not logged in -> welcome + login form
# ==============================================================================
if not is_logged_in():
    st.title("🛡️ Online Transaction Fraud Detection System")
    st.caption("Machine Learning + Explainable AI + Fraud Analyst Verification")
    st.markdown("---")

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.subheader("Welcome")
        st.markdown(
            """
            This system detects potentially fraudulent transactions using the
            trained XGBoost model with SHAP/LIME explainability.

            **Roles:**
            - **Admin** — upload datasets, run predictions, and view analytics.
            - **Fraud Analyst** — review stored review cases and decide whether
              each case is **Suspicious** or **Not a Fraud Case**.
            """
        )

    with col2:
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", type="primary", use_container_width=True):
            if do_login(username, password):
                st.rerun()
            else:
                st.error("Invalid username or password")

        with st.expander("Demo accounts"):
            st.code("admin / admin123\nanalyst / analyst123")

    st.stop()

# ==============================================================================
# After login -> two-role navigation
# ==============================================================================
role = st.session_state.get("role", "Admin")
username = st.session_state.get("username", "User")

with st.sidebar:
    st.success(f"👤 {username}")
    st.caption(f"Role: {role}")

    if st.button("Logout", use_container_width=True):
        do_logout()
        st.rerun()

    st.divider()

if role == "Admin":
    pages = {
        "Admin": [
            st.Page("pages_lib/admin_home.py", title="Homepage", icon="🏠", default=True),
            st.Page("pages_lib/upload_data.py", title="Upload Data", icon="📤"),
            st.Page("pages_lib/prediction.py", title="Prediction", icon="🔍"),
            st.Page("pages_lib/transaction_detail.py", title="Transaction Detail", icon="📄"),
            st.Page("pages_lib/analytics_dashboard.py", title="Analytics Dashboard", icon="📊"),
            st.Page("pages_lib/full_chart_insight.py", title="Full Chart Insight", icon="📈"),
            st.Page("pages_lib/explainability.py", title="Explainability", icon="🧠"),
            st.Page("pages_lib/view_case.py", title="View Results", icon="🔎"),
            st.Page("pages_lib/model_performance.py", title="Model Performance", icon="📈"),
        ]
    }
    valid_pages = [
        "Homepage", "Upload Data", "Prediction", "Transaction Detail",
        "Analytics Dashboard", "Full Chart Insight", "Explainability", "View Results",
    ]

elif role == "Fraud Analyst":
    pages = {
        "Fraud Analyst": [
            st.Page("pages_lib/fraud_analyst.py", title="Analyst Review", icon="🕵️", default=True),
            st.Page("pages_lib/model_performance.py", title="Model Performance", icon="📈"),
        ]
    }
    valid_pages = ["Analyst Review"]

else:
    do_logout()
    st.rerun()

# If the URL still has a page from a previous role, clear it.
if "page" in st.query_params:
    if st.query_params["page"] not in valid_pages:
        del st.query_params["page"]
        st.rerun()

pg = st.navigation(pages)
pg.run()