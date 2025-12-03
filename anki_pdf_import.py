# anki_pdf_import.py

import re
import streamlit as st
import pdfplumber

from clusters import assign_cluster_from_features, CLUSTERS


def extract_features_from_anki_pdf(file) -> dict:
    """Liest eine Anki-Statistik-PDF und extrahiert Kennzahlen."""

    with pdfplumber.open(file) as pdf:
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)

    # Helper: alles außer Ziffern entfernen
    def to_int(num_str: str) -> int:
        digits_only = re.sub(r"[^\d]", "", num_str)
        return int(digits_only) if digits_only else 0

    # 1) Gesamtzahl der Wiederholungen
    matches_total = re.findall(r"Insgesamt:\s*([\d\s\.,]+)\s*Wiederholungen", text)
    if not matches_total:
        raise ValueError("Konnte 'Insgesamt: ... Wiederholungen' nicht im PDF finden.")
    total_reviews = max(to_int(m) for m in matches_total)

    # 2) Lerntage / Zeitraum
    days_active = None
    days_total = None

    # Variante 1: klassische Zeile "Lerntage: X von Y"
    m_days = re.search(r"Lerntage:\s*([\d\s\.,]+)\s*von\s*([\d\s\.,]+)", text)
    if m_days:
        days_active = to_int(m_days.group(1))
        days_total = to_int(m_days.group(2))
    else:
        # Variante 2: nur Durchschnitt vorhanden → "Durchschnitt: 4 Wiederholungen/Tag"
        m_avg = re.search(r"Durchschnitt:\s*([\d\s\.,]+)\s*Wiederholungen/Tag", text)
        if m_avg:
            avg_per_day = float(m_avg.group(1).replace(",", "."))
            # Schätzung des Zeitraums
            days_total = int(round(total_reviews / avg_per_day)) if avg_per_day > 0 else 1
            days_active = days_total  # wir nehmen an, dass an fast allen Tagen gelernt wurde
        else:
            # Minimal-Fallback, falls alles fehlt
            days_total = 1
            days_active = 1

    # 3) Erinnerungsquote (Accuracy) – universell aus allen Prozentzahlen
    pct_matches = re.findall(r"(\d+,\d+)\s*%", text)
    if not pct_matches:
        raise ValueError("Konnte keine Prozentwerte (Erinnerungsquote) im PDF finden.")

    values = [float(p.replace(",", ".")) for p in pct_matches]
    candidates = [v for v in values if 50.0 <= v <= 100.0]
    if candidates:
        accuracy_pct = max(candidates)
    else:
        accuracy_pct = max(values)
    accuracy = accuracy_pct / 100.0

    # 4) Abgeleitete Kennzahlen
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

st.title("Anki-Lerntyp Analyse (PDF-Import)")

st.write(
    "Lade hier deine Anki-Statistik als **PDF** hoch "
    "(die Statistik-Seite aus Anki, exportiert als PDF). "
    "Die App berechnet daraus Lernkennzahlen und ordnet dich einem Lerntyp-Cluster zu."
)

uploaded_file = st.file_uploader("Anki-Statistik-PDF hochladen", type=["pdf"])

if uploaded_file is not None:
    try:
        features = extract_features_from_anki_pdf(uploaded_file)

        st.subheader("Extrahierte Lernkennzahlen")
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

        st.subheader("Dein Lerntyp (basierend auf Anki)")
        st.success(f"**{profile.name}**")
        st.write(profile.description)
        st.info(profile.recommendation)

    except Exception as e:
        st.error(f"Fehler beim Auslesen der PDF: {e}")
else:
    st.info("Bitte oben eine Anki-Statistik-PDF auswählen.")
