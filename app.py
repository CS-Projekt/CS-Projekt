-import streamlit as st
-
-st.write ("connected")
+"""Streamlit interface for the study planner prototype.
+
+The application captures the user inputs that will later feed the
+machine-learning component (e.g. the Ridge Regression models). It focuses on
+collecting session metadata and post-session feedback. The data is kept in the
+current Streamlit session so it can be reviewed immediately and exported later
+on.
+"""
+from __future__ import annotations
+
+from datetime import datetime, timedelta
+from typing import List, Optional
+
+import pandas as pd
+import streamlit as st
+
+st.set_page_config(page_title="Study Session Tracker", layout="centered")
+
+# ---------------------------------------------------------------------------
+# Helper utilities
+# ---------------------------------------------------------------------------
+
+
+def _initialize_session_state() -> None:
+    """Ensure all required keys exist in ``st.session_state``."""
+
+    if "session_active" not in st.session_state:
+        st.session_state.session_active = False
+    if "session_start" not in st.session_state:
+        st.session_state.session_start: Optional[datetime] = None
+    if "session_duration" not in st.session_state:
+        st.session_state.session_duration: Optional[timedelta] = None
+    if "session_logs" not in st.session_state:
+        st.session_state.session_logs: List[dict] = []
+
+
+def _format_duration(duration: timedelta | None) -> str:
+    """Return a human-readable duration string."""
+
+    if not duration:
+        return "—"
+    minutes = int(duration.total_seconds() // 60)
+    seconds = int(duration.total_seconds() % 60)
+    return f"{minutes:02d}:{seconds:02d}"
+
+
+_initialize_session_state()
+
+st.title("📚 Study Session Companion")
+st.write(
+    "Plan deine Lernsession, tracke deinen Fortschritt und gib danach Feedback. "
+    "Die gesammelten Daten können anschließend vom ML-Team genutzt werden, um "
+    "personalisierte Lernpläne zu erstellen."
+)
+
+# ---------------------------------------------------------------------------
+# Pre-session configuration
+# ---------------------------------------------------------------------------
+
+with st.form("session_setup_form"):
+    st.subheader("1. Session planen")
+    learning_strategy = st.selectbox(
+        "Lernstrategie",
+        (
+            "Pomodoro",
+            "Deep-Work-Sprints",
+            "Interleaving",
+            "Feynman-Methode",
+            "Spaced Repetition",
+        ),
+    )
+    subject = st.selectbox(
+        "Fach",
+        ("BWL 1", "Mikroökonomie", "Mathematik 1", "Privatrecht"),
+    )
+    planned_duration = st.number_input(
+        "Geplante Gesamtdauer (Minuten)",
+        min_value=15,
+        max_value=480,
+        step=5,
+        value=90,
+    )
+    baseline_concentration = st.slider(
+        "Erwartete Konzentration zu Beginn", 1, 10, value=7
+    )
+    submit_setup = st.form_submit_button("Session-Konfiguration speichern")
+
+if submit_setup:
+    st.success("Konfiguration gespeichert. Du kannst jetzt mit der Session starten.")
+
+# ---------------------------------------------------------------------------
+# Session controls
+# ---------------------------------------------------------------------------
+
+st.subheader("2. Session starten")
+col_start, col_status = st.columns([1, 3])
+
+with col_start:
+    if not st.session_state.session_active:
+        if st.button("▶️ Session starten", type="primary"):
+            st.session_state.session_active = True
+            st.session_state.session_start = datetime.now()
+            st.session_state.session_duration = None
+    else:
+        if st.button("⏹️ Session beenden", type="primary"):
+            st.session_state.session_active = False
+            start = st.session_state.session_start
+            if start is not None:
+                st.session_state.session_duration = datetime.now() - start
+
+with col_status:
+    if st.session_state.session_active and st.session_state.session_start:
+        elapsed = datetime.now() - st.session_state.session_start
+        st.info(f"Session läuft seit {_format_duration(elapsed)}")
+    else:
+        st.info(
+            "Session inaktiv. Starte eine neue Session oder beende die aktuelle, "
+            "um Feedback zu geben."
+        )
+
+# ---------------------------------------------------------------------------
+# Post-session feedback
+# ---------------------------------------------------------------------------
+
+st.subheader("3. Feedback zur Session")
+
+if st.session_state.session_duration is None:
+    st.caption("Sobald eine Session beendet wurde, kannst du hier dein Feedback abgeben.")
+else:
+    with st.form("feedback_form"):
+        st.write(
+            "Bewerte deine Konzentration und teile mit, wann du dich am besten "
+            "konzentrieren konntest."
+        )
+        concentration_rating = st.slider("Konzentrationsbewertung", 1, 10, value=7)
+        best_learning_time = st.selectbox(
+            "Beste Lernzeit",
+            ("Früh morgens", "Vormittag", "Nachmittag", "Abend", "Später Abend"),
+        )
+        distraction_factor = st.multiselect(
+            "Ablenkungsfaktoren",
+            (
+                "Zu lange Session",
+                "Zu spät begonnen",
+                "Zu wenig Pausen",
+                "Benachrichtigungen/Handy",
+                "Umgebungslärm",
+                "Keine Angabe",
+            ),
+            default=["Keine Angabe"],
+        )
+        additional_notes = st.text_area("Weitere Notizen", placeholder="Optional")
+        save_feedback = st.form_submit_button("Feedback speichern")
+
+    if save_feedback:
+        session_entry = {
+            "timestamp": datetime.now(),
+            "learning_strategy": learning_strategy,
+            "subject": subject,
+            "planned_duration_min": planned_duration,
+            "actual_duration_min": (
+                st.session_state.session_duration.total_seconds() / 60
+                if st.session_state.session_duration
+                else None
+            ),
+            "baseline_concentration": baseline_concentration,
+            "concentration_rating": concentration_rating,
+            "best_learning_time": best_learning_time,
+            "distraction_factors": ", ".join(distraction_factor),
+            "notes": additional_notes,
+        }
+        st.session_state.session_logs.append(session_entry)
+        # Reset duration so that users need to run another session before re-submitting
+        st.session_state.session_duration = None
+        st.success("Feedback gespeichert!")
+
+# ---------------------------------------------------------------------------
+# Session overview and basic analysis
+# ---------------------------------------------------------------------------
+
+st.subheader("4. Übersicht & Analyse")
+
+if st.session_state.session_logs:
+    df = pd.DataFrame(st.session_state.session_logs)
+    st.dataframe(df, use_container_width=True, hide_index=True)
+
+    st.markdown("### Kennzahlen")
+    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
+
+    with metrics_col1:
+        avg_rating = df["concentration_rating"].mean()
+        st.metric("Ø Konzentrationsrating", f"{avg_rating:.1f}")
+
+    with metrics_col2:
+        avg_actual_duration = df["actual_duration_min"].dropna().mean()
+        duration_text = f"{avg_actual_duration:.0f} Min" if pd.notna(avg_actual_duration) else "—"
+        st.metric("Ø reale Dauer", duration_text)
+
+    with metrics_col3:
+        favorite_time = (
+            df["best_learning_time"].mode()[0]
+            if not df["best_learning_time"].empty
+            else "—"
+        )
+        st.metric("Beliebteste Lernzeit", favorite_time)
+
+    st.markdown("### Häufigste Ablenkungsfaktoren")
+    distraction_counts = (
+        df["distraction_factors"].str.get_dummies(sep=", ").sum().sort_values(ascending=False)
+    )
+    st.bar_chart(distraction_counts)
+else:
+    st.info(
+        "Noch keine Sessions erfasst. Führe eine Session durch und speichere das Feedback, "
+        "um hier Auswertungen zu sehen."
+    )
