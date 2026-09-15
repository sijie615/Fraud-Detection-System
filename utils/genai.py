import os
import re
import streamlit as st

def clean_model_output(text: str) -> str:
    """Strips internal chain-of-thought tokens like <think>...</think>."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

def get_fallback_recommendation(amount, category, fraud_score, risk_level, top_feature="amt_log"):
    """
    Context-aware analytical recommendation tailored to category, amount, and features.
    Used if Groq API cannot be reached or returns an error.
    """
    cat_lower = str(category).lower()

    if any(k in cat_lower for k in ["misc_net", "shopping_net", "online"]):
        channel_risk = "Card-Not-Present (CNP) e-commerce vector"
        step1 = "Verify device digital fingerprint and IP geolocation against shipping address."
    elif any(k in cat_lower for k in ["grocery", "gas", "pos"]):
        channel_risk = "Point-of-Sale (POS) terminal vector with potential credential cloning"
        step1 = "Verify physical card terminal entry mode (chip/contactless vs magnetic swipe fallback)."
    elif any(k in cat_lower for k in ["travel", "hotel", "airline"]):
        channel_risk = "High-ticket travel/booking cross-border vector"
        step1 = "Cross-check passenger manifest details against the primary cardholder identity."
    else:
        channel_risk = f"Uncharacteristic spending pattern in '{category}'"
        step1 = f"Review 30-day merchant baseline spend for '{category}'."

    if amount >= 800:
        step2 = f"Transaction size (${amount:,.2f}) exceeds single-day limit; initiate immediate phone callback."
    else:
        step2 = f"Send automated two-factor SMS/push authorization for ${amount:,.2f} authorization check."

    if fraud_score >= 0.80 or risk_level in ["Red", "Critical Risk"]:
        urgency = "CRITICAL ALERT"
        action = f"Primary trigger '{top_feature}' indicates significant model deviation. Place a temporary block on further debit authorizations."
    elif fraud_score >= 0.50 or risk_level in ["Orange", "High Risk"]:
        urgency = "ELEVATED ALERT"
        action = f"Secondary flag triggered on '{top_feature}'. Flag account for enhanced monitoring over next 48 hours."
    else:
        urgency = "STANDARD VERIFICATION"
        action = "Transaction characteristics align with acceptable limits. Perform soft verification."

    return (
        f"[{urgency}] {channel_risk} (Score: {fraud_score:.4f}, Tier: {risk_level}).\n"
        f"• Risk Finding: {action}\n"
        f"• Action 1: {step1}\n"
        f"• Action 2: {step2}"
    )

def get_groq_recommendation(transaction_id, amount, category, fraud_score, risk_level, top_feature="amt_log"):
    """
    Directly queries a single Groq LLM to generate an authentic expert opinion.
    Falls back to analytical heuristics if API fails or key is missing.
    """
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", None))

    if not api_key or not str(api_key).strip().startswith("gsk_"):
        return get_fallback_recommendation(amount, category, fraud_score, risk_level, top_feature)

    clean_key = str(api_key).strip().strip('"').strip("'")

    prompt = f"""
You are a senior Anti-Money Laundering (AML) fraud forensics specialist.
Review the flagged transaction below:

[TRANSACTION DATA]
- Case Reference ID: {transaction_id}
- Transaction Amount: ${amount:,.2f}
- Merchant Category: {category}
- Model Fraud Probability: {fraud_score:.4f}
- Risk Level Tier: {risk_level}
- Primary Decisive Metric: {top_feature}

[INSTRUCTIONS]
Provide:
1. Executive Assessment: Briefly explain what makes this transaction pattern anomalous.
2. Direct Action Plan: Exactly 2 prioritized operational steps for the investigator.

Keep the tone professional, authoritative, and under 80 words total. Do not include internal thinking tags.
"""

    try:
        from groq import Groq
        client = Groq(api_key=clean_key)

        model_list = client.models.list()
        available_ids = [m.id for m in model_list.data]

        preferred_models = [
            "groq/compound-mini",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "qwen/qwen3.8-27b",
            "llama3-70b-8192"
        ]

        chosen_model = None
        for candidate in preferred_models:
            if candidate in available_ids:
                chosen_model = candidate
                break

        if not chosen_model:
            text_models = [m for m in available_ids if not any(x in m.lower() for x in ["whisper", "guard", "embed"])]
            chosen_model = text_models[0] if text_models else "llama-3.1-8b-instant"

        completion = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": "You are a professional financial crimes intelligence officer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=180
        )

        raw_output = completion.choices[0].message.content
        cleaned_output = clean_model_output(raw_output)

        if cleaned_output and cleaned_output.strip():
            return f"**[{chosen_model}]**\n\n{cleaned_output}"

    except Exception:
        pass

    return get_fallback_recommendation(amount, category, fraud_score, risk_level, top_feature)

def ask_analyst_groq(messages: list) -> str:
    """
    Executes a multi-turn chat completion using Groq SDK and the configured GROQ_API_KEY.
    """
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", None))

    if not api_key or not str(api_key).strip().startswith("gsk_"):
        return "⚠️ Groq API key is missing or invalid in `.streamlit/secrets.toml`."

    clean_key = str(api_key).strip().strip('"').strip("'")
    try:
        from groq import Groq
        client = Groq(api_key=clean_key)

        model_list = client.models.list()
        available_ids = [m.id for m in model_list.data]

        preferred_models = [
            "groq/compound-mini",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "qwen/qwen3.8-27b",
            "llama3-70b-8192"
        ]

        chosen_model = None
        for candidate in preferred_models:
            if candidate in available_ids:
                chosen_model = candidate
                break

        if not chosen_model:
            text_models = [m for m in available_ids if not any(x in m.lower() for x in ["whisper", "guard", "embed"])]
            chosen_model = text_models[0] if text_models else "llama-3.1-8b-instant"

        completion = client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            temperature=0.3,
            max_tokens=350
        )

        raw = completion.choices[0].message.content
        return clean_model_output(raw)

    except Exception as e:
        return f"⚠️ Groq Assistant Error: {e}"

def generate_chart_insight_groq(feature_name: str, stats_summary: str) -> str:
    """
    Calls Groq to generate the exact executive paragraph shown in 'Insight Behind the Curve'.
    """
    prompt = f"""
You are an expert fraud forensics analytics officer.
Write a concise, high-level executive interpretation for the feature '{feature_name}' based on the distribution summary:
{stats_summary}

Requirements:
- Start directly with: "In conclusion, while there is substantial overlap between the two classes..." (or similar analytical synthesis).
- Explain how '{feature_name}' helps distinguish suspicious from normal transactions and where the suspicious density dominates.
- Mention real-world fraud behavior (e.g. intentional structuring or synthetic mimicry).
- Tone: Exact financial crime reporting style.
- Length: Exactly 1 paragraph (under 90 words).
"""
    return ask_analyst_groq([
        {"role": "system", "content": "You write authoritative, concise fraud detection statistical interpretations."},
        {"role": "user", "content": prompt}
    ])