# anki_pdf_import.py

import re
import streamlit as st
import pdfplumber

from clusters import assign_cluster_from_features, CLUSTERS

from anki_pdf_analysis import extract_features_from_anki_pdf

# Streamlit User Interface

# Title
st.title("Anki Learning Type Analysis (PDF Import)")

# Description
st.write(
    "Upload your Anki statistics as a **PDF** "
    "(the stats page from Anki, exported as PDF). "
    "The app calculates learning metrics and assigns you to a learning-type cluster."
)

# File uploader and analysis
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
        # Assign Cluster 
        cluster_key = assign_cluster_from_features(features)
        profile = CLUSTERS[cluster_key]

        st.subheader("Your learning type (based on Anki)")
        st.success(f"**{profile.name}**")
        st.write(profile.description)
        st.info(profile.recommendation)

    # Error handling
    except Exception as e:
        st.error(f"Error while reading the PDF: {e}")
else:
    st.info("Please select an Anki stats PDF above.")
