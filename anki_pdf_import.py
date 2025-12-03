# anki_pdf_import.py

import re
import streamlit as st
import pdfplumber

from clusters import assign_cluster_from_features, CLUSTERS


def extract_features_from_anki_pdf(file) -> dict:
    """Reads an Anki stats PDF and extracts key metrics."""

    with pdfplumber.open(file) as pdf:
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)

    # Helper: strip everything except digits
    def to_int(num_str: str) -> int:
        digits_only = re.sub(r"[^\d]", "", num_str)
        return int(digits_only) if digits_only else 0

    # 1) Total number of reviews
    matches_total = re.findall(r"Insgesamt:\s*([\d\s\.,]+)\s*Wiederholungen", text)
    if not matches_total:
        raise ValueError("Couldn't find 'Insgesamt: ... Wiederholungen' in the PDF.")
    total_reviews = max(to_int(m) for m in matches_total)

    # 2) Study days / overall period
    days_active = None
    days_total = None

    # Variant 1: classic line "Lerntage: X von Y"
    m_days = re.search(r"Lerntage:\s*([\d\s\.,]+)\s*von\s*([\d\s\.,]+)", text)
    if m_days:
        days_active = to_int(m_days.group(1))
        days_total = to_int(m_days.group(2))
    else:
        # Variant 2: only average available → "Durchschnitt: 4 Wiederholungen/Tag"
        m_avg = re.search(r"Durchschnitt:\s*([\d\s\.,]+)\s*Wiederholungen/Tag", text)
        if m_avg:
            avg_per_day = float(m_avg.group(1).replace(",", "."))
            # Estimate the total period
            days_total = int(round(total_reviews / avg_per_day)) if avg_per_day > 0 else 1
            days_active = days_total  # assume learning on almost all days
        else:
            # Minimal fallback if everything is missing
            days_total = 1
            days_active = 1

    # 3) Recall rate (accuracy) – derived from all percentage values
    pct_matches = re.findall(r"(\d+,\d+)\s*%", text)
    if not pct_matches:
        raise ValueError("Couldn't find any percentage values (recall rate) in the PDF.")

    values = [float(p.replace(",", ".")) for p in pct_matches]
    candidates = [v for v in values if 50.0 <= v <= 100.0]
    if candidates:
        accuracy_pct = max(candidates)
    else:
        accuracy_pct = max(values)
    accuracy = accuracy_pct / 100.0

    # 4) Derived metrics
    learning_days_ratio = days_active / days_total if days_total > 0 else 0.0
    reviews_per_learning_day = total_reviews / days_active if days_active > 0 else 0.0
    daily_reviews = total_reviews / days_total if days_total > 0 else 0.0

    return {
        "total_reviews": total_reviews,
        "days_active": days_active,
        "days_total": days_total,
        "learning_days_ratio": learning_days_ratio,
        "reviews_per_learning_day": reviews_per_learning_day,
        "daily_reviews": daily_reviews,
        "accuracy": accuracy,
    }





# ----------------- Streamlit UI ----------------- #

st.title("Anki Learning Type Analysis (PDF Import)")

st.write(
    "Upload your Anki statistics as a **PDF** "
    "(the stats page from Anki, exported as PDF). "
    "The app calculates learning metrics and assigns you to a learning-type cluster."
)

uploaded_file = st.file_uploader("Upload Anki stats PDF", type=["pdf"])

if uploaded_file is not None:
    try:
        features = extract_features_from_anki_pdf(uploaded_file)

        st.subheader("Extracted learning metrics")
        features_pretty = {
            "total_reviews": features["total_reviews"],
            "days_active": features["days_active"],
            "days_total": features["days_total"],
            "learning_days_ratio": round(features["learning_days_ratio"], 3),
            "reviews_per_learning_day": round(features["reviews_per_learning_day"], 1),
            "daily_reviews": round(features["daily_reviews"], 1),
            "accuracy": round(features["accuracy"] * 100, 1),  # in %
        }
        st.json(features_pretty)

        cluster_key = assign_cluster_from_features(features)
        profile = CLUSTERS[cluster_key]

        st.subheader("Your learning type (based on Anki)")
        st.success(f"**{profile.name}**")
        st.write(profile.description)
        st.info(profile.recommendation)

    except Exception as e:
        st.error(f"Error while reading the PDF: {e}")
else:
    st.info("Please select an Anki stats PDF above.")
