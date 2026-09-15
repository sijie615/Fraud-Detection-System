"""
utils/styling.py -- visual polish: background colour, card-style metrics,
and a styled sidebar. Injected once in Home.py before st.navigation() runs
the selected page -- since Home.py calls pg.run() rather than exiting, this
CSS applies to every page for the rest of that script execution.
"""
import streamlit as st


def apply_custom_style():
    st.markdown(
        """
        <style>
        /* Overall page background -- soft, professional, not stark white */
        .stApp {
            background: linear-gradient(180deg, #f4f7fb 0%, #eef2f7 100%);
        }

        /* Tighter top padding */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }

        /* Card-style metric boxes */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }

        /* Section headers with a subtle rule */
        h1 { color: #1a2b4c; }
        h2, h3 { border-bottom: 1px solid #dde3ec; padding-bottom: 6px; color: #24344d; }

        /* Sidebar -- dark navy, matches a "security system" feel */
        section[data-testid="stSidebar"] {
            background-color: #0f1b2d;
        }
        section[data-testid="stSidebar"] * { color: #e8ecf1 !important; }
        section[data-testid="stSidebar"] .stButton button {
            background-color: #1e3a5f;
            color: white !important;
            border: none;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background-color: #2a4d7a;
        }

        /* Expander headers -- slightly rounded, subtle background */
        .streamlit-expanderHeader {
            background-color: #ffffff;
            border-radius: 8px;
        }

        /* Primary buttons */
        button[kind="primary"] {
            background-color: #1e3a5f;
            border: none;
        }
        button[kind="primary"]:hover {
            background-color: #2a4d7a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
