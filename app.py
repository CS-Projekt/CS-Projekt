import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time
import os
import streamlit.components.v1 as components

# Seiten-Konfiguration
st.set_page_config(
    page_title="AI Lernplan Generator",
    page_icon="📚",
    layout="wide"
)

# Modelle laden
@st.cache_resource
def load_models():
    """Lädt die trainierten ML-Modelle"""
    try:
        with open('learning_models.pkl', 'rb') as f:
            models = pickle.load(f)
        return models
    except FileNotFoundError:
        st.error("⚠️ Modell-Datei nicht gefunden! Bitte führe zuerst `train_model.py` aus.")
        return None

GOALS_DB_FILE = "goals.csv"


def load_goals_db():
    """Lädt die Goals-DB oder erstellt eine leere Struktur."""
    if os.path.exists(GOALS_DB_FILE):
        try:
            df = pd.read_csv(GOALS_DB_FILE)
            expected_cols = {'week', 'target_minutes'}
            if not expected_cols.issubset(df.columns):
                return pd.DataFrame(columns=['week', 'target_minutes'])
            return df
        except Exception:
            return pd.DataFrame(columns=['week', 'target_minutes'])
    return pd.DataFrame(columns=['week', 'target_minutes'])


def save_goals_db(df: pd.DataFrame):
    """Speichert die Goals-DB."""
    df.to_csv(GOALS_DB_FILE, index=False)

# Initialisierung
if 'models' not in st.session_state:
    st.session_state.models = load_models()

if 'user_history' not in st.session_state:
    st.session_state.user_history = pd.DataFrame(columns=[
        'timestamp', 'total_duration', 'time_of_day', 'concentration_baseline',
        'days_since_last', 'previous_rating', 'actual_rating', 'feedback'
    ])

if 'goals' not in st.session_state:
    st.session_state.goals = load_goals_db()

# Timer State
if 'timer_running' not in st.session_state:
    st.session_state.timer_running = False
if 'timer_start_time' not in st.session_state:
    st.session_state.timer_start_time = None
if 'current_block_index' not in st.session_state:
    st.session_state.current_block_index = 0
if 'timer_paused' not in st.session_state:
    st.session_state.timer_paused = False
if 'pause_time' not in st.session_state:
    st.session_state.pause_time = 0
if 'show_celebration' not in st.session_state:
    st.session_state.show_celebration = False
if 'remaining_at_pause' not in st.session_state:
    st.session_state.remaining_at_pause = 0

# Prüfen ob Modelle geladen wurden
if st.session_state.models is None:
    st.stop()

# Titel
st.title("AI-gestützter Lernplan Generator")
st.markdown("Erstelle optimierte Lernpläne basierend auf deinem Lernverhalten und KI-Vorhersagen")

# Navigation über Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    view_mode = st.radio(
        "Welche Ansicht möchtest du sehen?",
        options=["Lernplan", "Statistiken", "Goal Setting", "About"],
        index=0,
        key="view_mode"
    )

if view_mode == "Lernplan":
    # Sidebar für User-Input
    st.sidebar.header("Deine Lernsession planen")

    # Input: Gesamtdauer
    total_duration = st.sidebar.slider(
        "Wie lange möchtest du insgesamt lernen?",
        min_value=30,
        max_value=240,
        value=120,
        step=15,
        help="Gesamtdauer in Minuten"
    )

    # Input: Tageszeit
    time_of_day = st.sidebar.selectbox(
        "Zu welcher Tageszeit lernst du?",
        options=['morning', 'afternoon', 'evening', 'night'],
        format_func=lambda x: {
            'morning': '🌅 Morgen (6-12 Uhr)',
            'afternoon': '☀️ Nachmittag (12-18 Uhr)',
            'evening': '🌆 Abend (18-22 Uhr)',
            'night': '🌙 Nacht (22-6 Uhr)'
        }[x]
    )

    # Input: Konzentrationslevel
    concentration = st.sidebar.slider(
        "Wie konzentriert fühlst du dich gerade?",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
        help="1 = sehr unkonzentriert, 10 = hochkonzentriert"
    )

    # Input: Tage seit letzter Session
    if len(st.session_state.user_history) > 0:
        last_session = st.session_state.user_history.iloc[-1]['timestamp']
        days_since = (datetime.now() - last_session).days
        st.sidebar.info(f"Letzte Session: vor {days_since} Tag(en)")
    else:
        days_since = st.sidebar.number_input(
            "Wie viele Tage ist deine letzte Lernsession her?",
            min_value=0,
            max_value=30,
            value=1
        )

    # Input: Vorheriges Rating
    if len(st.session_state.user_history) > 0:
        previous_rating = st.session_state.user_history.iloc[-1]['actual_rating']
        st.sidebar.info(f"Letztes Session-Rating: {previous_rating}/10")
    else:
        previous_rating = st.sidebar.slider(
            "Wie gut lief deine letzte Lernsession?",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.5
        )

    # Button: Lernplan generieren
    generate_plan = st.sidebar.button("🚀 Lernplan generieren", type="primary")
else:
    # Platzhalterwerte für Statistiken-Ansicht
    total_duration = None
    time_of_day = None
    concentration = None
    days_since = None
    previous_rating = None
    generate_plan = False

if view_mode == "Lernplan" and generate_plan:
    
    # Features vorbereiten
    features = pd.DataFrame([{
        'total_session_duration': total_duration,
        'time_morning': 1 if time_of_day == 'morning' else 0,
        'time_afternoon': 1 if time_of_day == 'afternoon' else 0,
        'time_evening': 1 if time_of_day == 'evening' else 0,
        'time_night': 1 if time_of_day == 'night' else 0,
        'concentration_baseline': concentration,
        'days_since_last_session': days_since,
        'previous_session_rating': previous_rating
    }])
    
    # Skalieren
    models = st.session_state.models
    features_scaled = models['scaler'].transform(features)
    
    # Vorhersagen
    pred_work = int(round(models['work_duration'].predict(features_scaled)[0]))
    pred_break = int(round(models['break_duration'].predict(features_scaled)[0]))
    pred_next = models['next_session'].predict(features_scaled)[0]
    
    # Sicherstellen dass Vorhersagen sinnvoll sind
    pred_work = max(15, min(45, pred_work))
    pred_break = max(5, min(15, pred_break))
    
    # Anzahl Blöcke so berechnen dass Gesamtzeit passt
    cycle_duration = pred_work + pred_break
    pred_blocks = max(1, int((total_duration + pred_break) / cycle_duration))
    
    # Schedule erstellen
    schedule = []
    total_calculated = 0
    
    for block in range(pred_blocks):
        schedule.append({
            'type': 'Lernen',
            'duration': pred_work,
            'block': block + 1
        })
        total_calculated += pred_work
        
        if block < pred_blocks - 1:
            schedule.append({
                'type': 'Pause',
                'duration': pred_break,
                'block': block + 1
            })
            total_calculated += pred_break
    
    # In Session State speichern
    st.session_state.current_plan = {
        'blocks': pred_blocks,
        'work_duration': pred_work,
        'break_duration': pred_break,
        'next_session_hours': pred_next,
        'total_duration': total_duration,
        'actual_duration': total_calculated,
        'time_of_day': time_of_day,
        'concentration': concentration,
        'schedule': schedule
    }
    
    # Timer zurücksetzen
    st.session_state.timer_running = False
    st.session_state.current_block_index = 0
    st.session_state.timer_paused = False
    st.session_state.pause_time = 0
    st.session_state.show_celebration = False

# Hilfsfunktion für die Willkommensseite
def render_welcome_content():
    st.header("Willkommen beim AI Lernplan Generator")
    st.info("Nutze die Sidebar, um deinen personalisierten Lernplan zu erstellen oder Statistiken einzusehen.")
    st.markdown("""
    ### So funktioniert's:

    1. **Gib deine Parameter ein** (Dauer, Tageszeit, Konzentration)
    2. **Klicke auf "Lernplan generieren"**
    3. **Nutze den interaktiven Timer** mit Countdown und Animationen
    4. **Gib nach der Session Feedback**, um die KI zu trainieren

    ### Was macht die KI?

    Die Ridge Regression analysiert:
    - Deine Konzentrationsfähigkeit
    - Die Tageszeit (Chronobiologie)
    - Dein bisheriges Lernverhalten
    - Erholungszeiten zwischen Sessions

    Und empfiehlt dir:
    - Optimale Anzahl und Länge der Lernblöcke
    - Passende Pausenzeiten
    - Den besten Zeitpunkt für die nächste Session
    - Interaktive Timer-Steuerung
    """)


# Hauptbereich abhängig von der Navigation anzeigen
if view_mode == "Goal Setting":
    st.header("🎯 Goal Setting")
    goals_df = st.session_state.goals.copy()
    now_ts = pd.Timestamp.now()
    iso_calendar = now_ts.isocalendar()
    current_week_label = f"{iso_calendar.year}-KW{iso_calendar.week:02d}"
    week_start = now_ts.normalize() - pd.Timedelta(days=now_ts.weekday())
    week_end = week_start + pd.Timedelta(days=7)
    existing_goal = goals_df[goals_df['week'] == current_week_label]
    stored_target = existing_goal['target_minutes'].iloc[0] if not existing_goal.empty else None
    default_target = int(stored_target) if stored_target is not None else 240

    st.subheader(f"Wöchentliches Ziel – {current_week_label}")
    target_input = st.number_input(
        "Ziel-Minuten für diese Woche",
        min_value=30,
        max_value=1200,
        step=30,
        value=int(default_target)
    )

    target_for_progress = stored_target
    if st.button("Ziel speichern", type="primary"):
        updated_df = goals_df[goals_df['week'] != current_week_label]
        new_row = pd.DataFrame([{
            'week': current_week_label,
            'target_minutes': int(target_input)
        }])
        updated_df = pd.concat([updated_df, new_row], ignore_index=True)
        st.session_state.goals = updated_df
        save_goals_db(updated_df)
        goals_df = updated_df
        target_for_progress = int(target_input)
        st.success("Wochenziel gespeichert.")

    history_goal = st.session_state.user_history.copy()
    if len(history_goal) > 0:
        history_goal['timestamp_dt'] = pd.to_datetime(history_goal['timestamp'], errors='coerce')
        week_minutes = history_goal.loc[
            (history_goal['timestamp_dt'] >= week_start) &
            (history_goal['timestamp_dt'] < week_end),
            'total_duration'
        ].sum()
    else:
        week_minutes = 0

    col_goal = st.columns(2)
    with col_goal[0]:
        target_display = f"{int(target_for_progress)} min" if target_for_progress else "–"
        st.metric("Aktuelles Ziel", target_display)
    with col_goal[1]:
        st.metric("Gelernt diese Woche", f"{int(week_minutes)} min")

    if target_for_progress and target_for_progress > 0:
        progress_pct = min(100, (week_minutes / target_for_progress) * 100)
        st.progress(min(1.0, progress_pct / 100), text=f"{week_minutes:.0f} / {target_for_progress:.0f} Minuten")
        st.info(f"Du bist bei {progress_pct:.0f}% deines Wochenziels.")
    else:
        st.info("Lege ein Wochenziel fest, um den Fortschritt zu verfolgen.")

    st.subheader("Zielhistorie")
    if len(goals_df) == 0:
        st.caption("Noch keine Ziele gespeichert.")
    else:
        goal_history = goals_df.copy().sort_values('week', ascending=False)
        goal_history = goal_history.rename(columns={
            'week': 'Kalenderwoche',
            'target_minutes': 'Ziel-Minuten'
        })
        goal_history['Ziel-Minuten'] = goal_history['Ziel-Minuten'].astype(int)

        history_goal_for_table = st.session_state.user_history.copy()
        if len(history_goal_for_table) > 0:
            history_goal_for_table['timestamp_dt'] = pd.to_datetime(
                history_goal_for_table['timestamp'], errors='coerce'
            )
            history_goal_for_table = history_goal_for_table.dropna(subset=['timestamp_dt'])
            iso_weeks = history_goal_for_table['timestamp_dt'].dt.isocalendar()
            history_goal_for_table['week_label'] = (
                iso_weeks['year'].astype(str) + "-KW" + iso_weeks['week'].astype(str).str.zfill(2)
            )
            actual_week_minutes = history_goal_for_table.groupby('week_label')['total_duration'].sum()
        else:
            actual_week_minutes = pd.Series(dtype=float)

        goal_history['Gelernt (min)'] = (
            goal_history['Kalenderwoche'].map(actual_week_minutes).fillna(0).astype(int)
        )
        goal_history['Erfüllung (%)'] = goal_history.apply(
            lambda row: (row['Gelernt (min)'] / row['Ziel-Minuten'] * 100) if row['Ziel-Minuten'] > 0 else 0,
            axis=1
        )
        goal_history['Erfüllung (%)'] = goal_history['Erfüllung (%)'].round(0).astype(int).astype(str) + "%"
        st.dataframe(goal_history, use_container_width=True, hide_index=True)

elif view_mode == "About":
    render_welcome_content()

elif view_mode == "Statistiken":
    history = st.session_state.user_history
    st.header("📊 Statistik-Dashboard")

    if len(history) == 0:
        st.info("Noch keine Daten vorhanden. Gib nach deiner ersten Session Feedback, um Statistiken aufzubauen.")
    else:
        sessions_completed = len(history)
        avg_rating = history['actual_rating'].mean()
        avg_duration = history['total_duration'].mean()
        history_with_ts = history.copy()
        history_with_ts['timestamp_dt'] = pd.to_datetime(history_with_ts['timestamp'], errors='coerce')
        today = pd.Timestamp.now().date()
        today_minutes = history_with_ts.loc[
            history_with_ts['timestamp_dt'].dt.date == today, 'total_duration'
        ].sum()
        last_session_time = history.iloc[-1]['timestamp']
        last_session_str = last_session_time.strftime("%d.%m.%Y %H:%M") if hasattr(last_session_time, 'strftime') else str(last_session_time)

        col_stats = st.columns(4)
        col_stats[0].metric("Absolvierte Sessions", sessions_completed)
        col_stats[1].metric("Heute gelernte Minuten", f"{int(today_minutes)} min")
        col_stats[2].metric("Ø Session-Rating", f"{avg_rating:.1f}/10")
        col_stats[3].metric("Ø Sessiondauer", f"{avg_duration:.0f} min")
        st.caption(f"Letzte Session: {last_session_str}")

        chart_df = history[['timestamp', 'actual_rating']].copy().sort_values('timestamp')
        chart_df['timestamp'] = chart_df['timestamp'].astype(str)
        chart_df = chart_df.set_index('timestamp')
        st.subheader("Rating-Verlauf")
        st.line_chart(chart_df, height=280)

        st.subheader("Session-Historie")
        history_display = history.copy()
        history_display['timestamp'] = pd.to_datetime(history_display['timestamp'], errors='coerce')
        history_display['Datum'] = history_display['timestamp'].apply(
            lambda ts: ts.strftime("%d.%m") if pd.notna(ts) else ""
        )
        history_display['Uhrzeit'] = history_display['timestamp'].apply(
            lambda ts: ts.strftime("%H.%M") if pd.notna(ts) else ""
        )
        history_display = history_display[[
            'Datum', 'Uhrzeit', 'total_duration', 'time_of_day', 'concentration_baseline',
            'days_since_last', 'previous_rating', 'actual_rating', 'feedback'
        ]]
        history_display = history_display.rename(columns={
            'total_duration': 'Dauer (min)',
            'time_of_day': 'Tageszeit',
            'concentration_baseline': 'Konzentration',
            'days_since_last': 'Tage seither',
            'previous_rating': 'Vorheriges Rating',
            'actual_rating': 'Aktuelles Rating',
            'feedback': 'Feedback'
        })
        st.dataframe(history_display, use_container_width=True, hide_index=True)

        st.subheader("Kalender nach Tageszeit & Wochentag")
        weekday_labels = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
        time_labels = ["Morgen", "Mittag", "Abend", "Nacht"]
        calendar_df = pd.DataFrame(index=time_labels, columns=weekday_labels, dtype=float)

        history_for_calendar = history.copy()
        history_for_calendar['timestamp'] = pd.to_datetime(history_for_calendar['timestamp'], errors='coerce')

        weekday_map = {0: "Mo", 1: "Di", 2: "Mi", 3: "Do", 4: "Fr", 5: "Sa", 6: "So"}
        time_map = {
            'morning': "Morgen",
            'afternoon': "Mittag",
            'evening': "Abend",
            'night': "Nacht"
        }

        for _, entry in history_for_calendar.iterrows():
            timestamp = entry.get('timestamp')
            time_of_day_value = entry.get('time_of_day')
            rating_value = entry.get('actual_rating')
            if pd.isna(timestamp) or pd.isna(time_of_day_value) or pd.isna(rating_value):
                continue

            weekday_label = weekday_map.get(timestamp.weekday())
            time_label = time_map.get(time_of_day_value)

            if weekday_label in calendar_df.columns and time_label in calendar_df.index:
                calendar_df.loc[time_label, weekday_label] = rating_value

        styled_calendar = calendar_df.style.background_gradient(
            axis=None,
            cmap="RdYlGn",
            vmin=1,
            vmax=10
        )
        styled_calendar = styled_calendar.applymap(
            lambda v: "background-color: #ffffff" if pd.isna(v) else ""
        ).format(lambda v: f"{v:.1f}" if pd.notna(v) else "")

        st.dataframe(styled_calendar, use_container_width=True)

else:
    if 'current_plan' in st.session_state:
        plan = st.session_state.current_plan
        
        # Metriken anzeigen
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Lernblöcke", f"{plan['blocks']}")
        
        with col2:
            st.metric("Lernblock-Dauer", f"{plan['work_duration']} min")
        
        with col3:
            st.metric("Pausen-Dauer", f"{plan['break_duration']} min")
        
        with col4:
            st.metric("Tatsächliche Dauer", f"{plan['actual_duration']} min")
        
        with col5:
            st.metric("Nächste Session in", f"{plan['next_session_hours']:.1f} h")

        # TIMER BEREICH
        st.markdown("---")

        # Celebration Animation
        if st.session_state.show_celebration:
            st.balloons()
            st.success("🎉 Großartig! Block abgeschlossen!")
            st.session_state.show_celebration = False

        schedule = plan['schedule']
        current_idx = st.session_state.current_block_index

        if current_idx < len(schedule):
            current_item = schedule[current_idx]

            # Timer-Header
            st.subheader("Timer")

            # Fortschritt
            progress = current_idx / len(schedule) if len(schedule) > 0 else 0
            st.progress(progress, text=f"Block {current_idx + 1} von {len(schedule)}")

            # Aktueller Block Info
            col_timer1, col_timer2 = st.columns([2, 1])

            with col_timer1:
                if current_item['type'] == 'Lernen':
                    st.markdown(f"### Lernblock {current_item['block']}")
                    timer_color = "#4CAF50"
                else:
                    st.markdown(f"### Pause nach Block {current_item['block']}")
                    timer_color = "#FF9800"

            # Timer berechnen
            if st.session_state.timer_running and not st.session_state.timer_paused:
                elapsed = (time.time() - st.session_state.timer_start_time) - st.session_state.pause_time
                remaining_seconds = max(0, current_item['duration'] * 60 - elapsed)
            elif st.session_state.timer_paused:
                remaining_seconds = st.session_state.remaining_at_pause
            else:
                remaining_seconds = current_item['duration'] * 60

            minutes = int(remaining_seconds // 60)
            seconds = int(remaining_seconds % 60)

            # Timer Display (simpel ohne Box)
            with col_timer2:
                auto_update = st.session_state.timer_running and not st.session_state.timer_paused
                timer_html = f"""
                    <div style='text-align: center;'>
                        <h1 id="timer-display" style='margin: 10px 0; font-size: 4em; color: {timer_color}; font-weight: bold;'>{minutes:02d}:{seconds:02d}</h1>
                    </div>
                    <script>
                        const autoUpdate = {str(auto_update).lower()};
                        let secondsRemaining = {int(remaining_seconds)};
                        const displayEl = document.getElementById("timer-display");
                        if (window.timerInterval) {{
                            clearInterval(window.timerInterval);
                        }}
                        if (autoUpdate) {{
                            window.timerInterval = setInterval(() => {{
                                secondsRemaining = Math.max(secondsRemaining - 1, 0);
                                const mins = String(Math.floor(secondsRemaining / 60)).padStart(2, '0');
                                const secs = String(secondsRemaining % 60).padStart(2, '0');
                                displayEl.textContent = `${{mins}}:${{secs}}`;
                                if (secondsRemaining === 0) {{
                                    clearInterval(window.timerInterval);
                                }}
                            }}, 1000);
                        }}
                    </script>
                """
                components.html(timer_html, height=140, scrolling=False)

            # Timer Kontrollen
            col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

            with col_btn1:
                if not st.session_state.timer_running:
                    if st.button("▶️ Start", use_container_width=True, key="start_btn"):
                        st.session_state.timer_running = True
                        st.session_state.timer_start_time = time.time()
                        st.session_state.pause_time = 0
                        st.session_state.timer_paused = False
                        st.rerun()
                else:
                    if not st.session_state.timer_paused:
                        if st.button("⏸️ Pause", use_container_width=True, key="pause_btn"):
                            st.session_state.timer_paused = True
                            st.session_state.remaining_at_pause = remaining_seconds
                            st.rerun()
                    else:
                        if st.button("▶️ Weiter", use_container_width=True, key="continue_btn"):
                            st.session_state.timer_paused = False
                            elapsed_pause = time.time() - st.session_state.timer_start_time
                            st.session_state.pause_time = elapsed_pause - (current_item['duration'] * 60 - st.session_state.remaining_at_pause)
                            st.session_state.timer_start_time = time.time() - (current_item['duration'] * 60 - st.session_state.remaining_at_pause)
                            st.rerun()

            with col_btn2:
                if st.button("⏭️ Skip", use_container_width=True, key="skip_btn"):
                    st.session_state.show_celebration = True
                    st.session_state.current_block_index += 1
                    st.session_state.timer_running = False
                    st.session_state.timer_paused = False
                    st.session_state.pause_time = 0
                    st.rerun()

            with col_btn3:
                if st.button("🔄 Reset", use_container_width=True, key="reset_btn"):
                    st.session_state.timer_running = False
                    st.session_state.timer_start_time = None
                    st.session_state.timer_paused = False
                    st.session_state.pause_time = 0
                    st.rerun()

            with col_btn4:
                if st.button("⏹️ Beenden", use_container_width=True, key="stop_btn"):
                    st.session_state.current_block_index = 0
                    st.session_state.timer_running = False
                    st.session_state.timer_paused = False
                    st.rerun()

            # Hinweis wenn Timer abgelaufen
            if remaining_seconds <= 0 and st.session_state.timer_running:
                st.warning("⏰ Zeit abgelaufen! Klicke auf 'Weiter zum nächsten Block'")

                # Button für nächsten Block
                if st.button("➡️ Weiter zum nächsten Block", use_container_width=True, type="primary", key="next_block_btn"):
                    st.session_state.show_celebration = True
                    st.session_state.current_block_index += 1
                    st.session_state.timer_running = False
                    st.session_state.timer_paused = False
                    st.session_state.pause_time = 0
                    st.rerun()

        else:
            st.success("🎊 Glückwunsch! Du hast alle Lernblöcke abgeschlossen!")
            st.balloons()
            if st.button("🔄 Neue Session starten", key="new_session_btn"):
                st.session_state.current_block_index = 0
                st.session_state.timer_running = False
                st.session_state.timer_paused = False
                st.rerun()

        st.markdown("---")

        # Zeitplan visualisieren
        st.subheader("Dein Lernplan im Detail")

        # Zeitplan-Tabelle
        schedule_display = []

        for i, item in enumerate(schedule):
            status = "✅" if i < current_idx else ("🔄" if i == current_idx else "⏳")
            schedule_display.append({
                'Nr.': i + 1,
                'Status': status,
                'Aktivität': item['type'],
                'Dauer': f"{item['duration']} min"
            })

        st.dataframe(
            pd.DataFrame(schedule_display),
            use_container_width=True,
            hide_index=True
        )

        # Gantt-Chart (verbesserte Darstellung ohne Stern)
        fig = go.Figure()

        # Sammle alle Lernblöcke und Pausen
        work_blocks_x = []
        work_blocks_y = []
        pause_blocks_x = []
        pause_blocks_y = []

        for i, item in enumerate(schedule):
            if item['type'] == 'Lernen':
                work_blocks_x.append(item['duration'])
                work_blocks_y.append(len(schedule) - i - 1)  # Umgedrehte Y-Achse
            else:
                pause_blocks_x.append(item['duration'])
                pause_blocks_y.append(len(schedule) - i - 1)

        # Lernblöcke hinzufügen
        if work_blocks_x:
            fig.add_trace(go.Bar(
                name='Lernen',
                x=work_blocks_x,
                y=work_blocks_y,
                orientation='h',
                marker=dict(color='#4CAF50'),
                text=[f"Lernen {x} min" for x in work_blocks_x],
                textposition='inside',
                hovertemplate='Lernen: %{x} min<extra></extra>'
            ))

        # Pausen hinzufügen
        if pause_blocks_x:
            fig.add_trace(go.Bar(
                name='Pause',
                x=pause_blocks_x,
                y=pause_blocks_y,
                orientation='h',
                marker=dict(color='#FF9800'),
                text=[f"Pause {x} min" for x in pause_blocks_x],
                textposition='inside',
                hovertemplate='Pause: %{x} min<extra></extra>'
            ))

        fig.update_layout(
            title="Zeitlicher Ablauf deiner Lernsession",
            xaxis_title="Dauer (Minuten)",
            yaxis_title="",
            barmode='overlay',
            height=max(300, len(schedule) * 40),
            yaxis=dict(
                showticklabels=False,
                range=[-0.5, len(schedule) - 0.5]
            ),
            xaxis=dict(range=[0, max([item['duration'] for item in schedule]) * 1.1]),
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Info über Zeitabweichung
        time_diff = abs(plan['total_duration'] - plan['actual_duration'])
        if time_diff > 5:
            st.info(f"ℹ️ Die tatsächliche Session-Dauer ({plan['actual_duration']} min) weicht von deiner Wunschdauer ({plan['total_duration']} min) ab. Das liegt an der Optimierung der Lernblock-Längen für maximale Effizienz.")

        # Tipps basierend auf Vorhersagen
        st.subheader("Personalisierte Tipps")

        tips = []
        if plan['concentration'] < 5:
            tips.append("⚠️ Niedrige Konzentration erkannt. Versuche kurze Lernblöcke mit längeren Pausen.")
        if plan['time_of_day'] == 'night':
            tips.append("🌙 Spätabends zu lernen kann ineffizient sein. Überlege, ob eine frühere Zeit möglich ist.")
        if plan['blocks'] > 5:
            tips.append("🔋 Viele Lernblöcke geplant! Denk an ausreichend Flüssigkeit und Snacks.")
        if plan['next_session_hours'] < 6:
            tips.append("⏰ Kurze Pause bis zur nächsten Session empfohlen. Achte auf Erholung!")

        if tips:
            for tip in tips:
                st.info(tip)
        else:
            st.success("✅ Dein Lernplan sieht optimal aus! Viel Erfolg!")

        # Feedback nach der Session
        st.subheader("Session-Feedback")
        st.markdown("*Nach deiner Lernsession kannst du Feedback geben, um die KI zu verbessern:*")

        with st.form("feedback_form"):
            actual_rating = st.slider(
                "Wie gut war deine Konzentration während der Session?",
                min_value=1.0,
                max_value=10.0,
                value=7.0,
                step=0.5
            )

            feedback_reasons = st.multiselect(
                "Falls es nicht optimal lief, was waren die Gründe?",
                options=[
                    "Zu lange Lernblöcke",
                    "Zu kurze Pausen",
                    "Zu späte Uhrzeit",
                    "Zu frühe Uhrzeit",
                    "Zu wenig Schlaf",
                    "Ablenkungen",
                    "Schwieriges Thema",
                    "Andere"
                ]
            )

            submitted = st.form_submit_button("💾 Feedback speichern")

            if submitted:
                new_entry = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'total_duration': plan['total_duration'],
                    'time_of_day': plan['time_of_day'],
                    'concentration_baseline': plan['concentration'],
                    'days_since_last': days_since,
                    'previous_rating': previous_rating,
                    'actual_rating': actual_rating,
                    'feedback': ', '.join(feedback_reasons)
                }])

                st.session_state.user_history = pd.concat(
                    [st.session_state.user_history, new_entry],
                    ignore_index=True
                )

                st.success("✅ Feedback gespeichert! Die KI lernt mit jedem Feedback dazu.")

    else:
        render_welcome_content()
