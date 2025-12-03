import pdfplumber
import re
from typing import Dict, Any
from io import BytesIO

def to_int(num_str: str) -> int:
    digits_only = re.sub(r"[^\d]", "", num_str)
    return int(digits_only) if digits_only else 0

def extract_features_from_anki_pdf(file_bytes_or_path) -> Dict[str, Any]:
    """
    Accepts either bytes (PDF content) or a file path. Returns the features dict:
    {
      total_reviews, days_active, days_total, learning_days_ratio,
      reviews_per_learning_day, daily_reviews, accuracy
    }
    """
    # handle bytes input
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        file_obj = BytesIO(file_bytes_or_path)
    else:
        file_obj = file_bytes_or_path

    with pdfplumber.open(file_obj) as pdf:
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)

    matches_total = re.findall(r"Insgesamt:\s*([\d\s\.,]+)\s*Wiederholungen", text)
    if not matches_total:
        raise ValueError("Couldn't find 'Insgesamt: ... Wiederholungen' in the PDF.")
    total_reviews = max(to_int(m) for m in matches_total)

    days_active = None
    days_total = None

    m_days = re.search(r"Lerntage:\s*([\d\s\.,]+)\s*von\s*([\d\s\.,]+)", text)
    if m_days:
        days_active = to_int(m_days.group(1))
        days_total = to_int(m_days.group(2))
    else:
        m_avg = re.search(r"Durchschnitt:\s*([\d\s\.,]+)\s*Wiederholungen/Tag", text)
        if m_avg:
            avg_per_day = float(m_avg.group(1).replace(",", "."))
            days_total = int(round(total_reviews / avg_per_day)) if avg_per_day > 0 else 1
            days_active = days_total
        else:
            days_total = 1
            days_active = 1

    pct_matches = re.findall(r"(\d+,\d+)\s*%", text)
    if not pct_matches:
        raise ValueError("Couldn't find any percentage values (recall rate) in the PDF.")

    values = [float(p.replace(",", ".")) for p in pct_matches]
    candidates = [v for v in values if 50.0 <= v <= 100.0]
    accuracy_pct = max(candidates) if candidates else max(values)
    accuracy = accuracy_pct / 100.0

    learning_days_ratio = days_active / days_total if days_total > 0 else 0.0
    reviews_per_learning_day = total_reviews / days_active if days_active > 0 else 0.0
    daily_reviews = total_reviews / days_total if days_total > 0 else 0.0

    return {
        "total_reviews": int(total_reviews),
        "days_active": int(days_active),
        "days_total": int(days_total),
        "learning_days_ratio": float(learning_days_ratio),
        "reviews_per_learning_day": float(reviews_per_learning_day),
        "daily_reviews": float(daily_reviews),
        "accuracy": float(accuracy),
    }