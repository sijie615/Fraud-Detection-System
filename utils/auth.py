import streamlit as st

# Two-role prototype authentication.
# No case records or role assignments are stored in the database.
USERS = {
    "admin": {"password": "admin123", "role": "Admin"},
    "analyst": {"password": "analyst123", "role": "Fraud Analyst"},
}

ROLES = ["Admin", "Fraud Analyst"]


def is_logged_in() -> bool:
    return bool(st.session_state.get("logged_in", False))


def do_login(username: str, password: str) -> bool:
    username = username.strip().lower()
    user = USERS.get(username)

    if user and user["password"] == password:
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        st.session_state["role"] = user["role"]
        return True

    return False


def do_logout():
    for key in ["logged_in", "username", "role"]:
        st.session_state.pop(key, None)
    st.query_params.clear()


def require_role(allowed_roles):
    if isinstance(allowed_roles, str):
        allowed_roles = [allowed_roles]

    if not is_logged_in():
        st.warning("Please login first.")
        st.stop()

    user_role = st.session_state.get("role")
    if user_role not in allowed_roles:
        st.error(
            f"Access Denied. This page requires one of: {', '.join(allowed_roles)}"
        )
        st.stop()
