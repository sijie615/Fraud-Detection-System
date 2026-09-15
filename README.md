# Fraud Detection System -- Lim Aun Chir Structure (Chapter 5.10)

This dashboard mirrors the reference report's Chapter 5.10 "Website
Development" structure: 3 roles (Admin, Investigator, Reviewer -- the
reference report's 4th role, Developer, is scoped out here), each with
their OWN page set (not shared pages with role-gating), built using
Streamlit's native `st.navigation()` / `st.Page()` API.

## Structure

```
Home.py                     <- Homepage + Login (Fig 5.10.1, 5.10.2), then
                                routes to the logged-in role's page set
pages_lib/
  admin_home.py              Fig 5.10.3  Admin's Homepage
  upload_data.py             Fig 5.10.4-5.10.5  Upload page
  prediction.py              Fig 5.10.6-5.10.11  Prediction (colour-coded + SHAP)
  transaction_detail.py      Fig 5.10.12  Transaction Detail
  analytics_dashboard.py     Fig 5.10.13-5.10.17  Dashboard Page
  full_chart_insight.py      Fig 5.10.18  Full Chart Insight
  view_case.py                Fig 5.10.19-5.10.20  View Case (search by ID)
  investigator_home.py       Fig 5.10.21  Investigator's Homepage
  case_management.py         Fig 5.10.22-5.10.26, 5.10.30  (shared Investigator/Reviewer)
  case_resolved.py           Fig 5.10.27-5.10.28, 5.10.31-5.10.32  (shared)
  reviewer_home.py           Fig 5.10.29  Reviewer's Homepage
utils/
  auth.py                    Login + role check (3 roles)
  db.py                      SQLite, 4-tier risk colour coding, Pending->Reported->Verified workflow
  preprocessing.py           Feature engineering pipeline (unchanged from earlier work)
  genai.py                   AI Recommendations (Groq API, matches Fig 5.10.25)
models/                      Put your .pkl/.json/.csv/.npy artifacts here
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy your model artifacts into `models/`:
```
final_fraud_model.pkl, preprocessor.pkl, state_target_encoding.json,
shap_explainer.pkl, shap_values.npy, shap_sample_transactions.csv,
lime_training_background.csv
```

## Run

```bash
streamlit run Home.py
```

## Demo accounts

```
admin / admin123
investigator / invest123
reviewer / review123
```

## What differs from the earlier flat-page-list version of this dashboard

The previous version had one shared sidebar with every page listed, gated
per-page via `require_role()` (a page would show "Access Denied" if your
role didn't match). This version uses `st.navigation()` to build a
**completely different sidebar per role** at login time -- an Investigator
never even sees "Upload Data" in their menu, matching how the reference
report describes genuinely separate Admin/Investigator/Reviewer/Developer
"Views," not one shared menu with items disabled.

## Known simplification vs. the reference report

- **SHAP View** in Case Management shows the case's single stored top
  feature (captured at the time a case was created on the Prediction page),
  not a full re-computed waterfall plot -- the case log doesn't store the
  full feature vector needed to regenerate one. The Prediction page itself
  *does* show a full per-transaction SHAP breakdown before a case is saved.
- **Firebase and local Ollama hosting** (Sections 5.11-5.12 of the
  reference report) are not implemented here -- this dashboard uses SQLite
  (matching the rest of this project's existing architecture) and Groq's
  API instead. Both are swappable later without touching page code, since
  every page calls `utils/db.py` and `utils/genai.py` rather than talking
  to SQLite or an LLM API directly.
