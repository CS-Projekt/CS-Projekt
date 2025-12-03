import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time
import os
import streamlit.components.v1 as components
import pdfplumber
import database as db
from clusters import (
    assign_cluster_from_features,
    assign_cluster_id,
    CLUSTERS,
)
from ml_api_client import (
    PredictionAPIError,
    predict_session_via_api,
    is_api_available,
)

try:
    import matplotlib  # noqa: F401
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="AI Study Plan Generator",
    page_icon="📚",
    layout="wide"
)

# --- Database setup ---
db.init_db()
DEFAULT_USER_ID = db.get_or_create_default_user()
# You can use DEFAULT_USER_ID later whenever you save or load sessions
DEFAULT_CLUSTER_ID = 1  # Fallback to structured planner

@st.cache_resource
def load_models():
    """Loads the trained ML models."""
    try:
        with open('learning_models.pkl', 'rb') as f:
            models = pickle.load(f)
        return models
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please run `train_model.py` first.")
        return None


def get_ml_api_status(force_refresh: bool = False) -> bool:
    """
    Cache the ML API availability check so the UI stays responsive.
    """
    if force_refresh or 'ml_api_available' not in st.session_state:
        st.session_state.ml_api_available = is_api_available()
    return st.session_state.ml_api_available


GOALS_DB_FILE = "goals.csv"


def load_goals_db():
    """Loads the goals DB or returns an empty structure."""
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
    """Persists the goals DB."""
    df.to_csv(GOALS_DB_FILE, index=False)


# Initialization
def extract_features_from_anki_pdf(file) -> dict:
    """Reads an Anki statistics PDF and extracts key metrics."""

    with pdfplumber.open(file) as pdf:
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)

    def to_int(num_str: str) -> int:
        digits_only = re.sub(r"[^\d]", "", num_str)
        return int(digits_only) if digits_only else 0

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
        "total_reviews": total_reviews,
        "days_active": days_active,
        "days_total": days_total,
        "learning_days_ratio": learning_days_ratio,
        "reviews_per_learning_day": reviews_per_learning_day,
        "daily_reviews": daily_reviews,
        "accuracy": accuracy,
    }

# Initialization
if 'models' not in st.session_state:
    st.session_state.models = load_models()

if 'user_history' not in st.session_state:
    st.session_state.user_history = pd.DataFrame(columns=[
        'timestamp', 'total_duration', 'time_of_day', 'concentration_baseline',
        'days_since_last', 'previous_rating', 'actual_rating', 'feedback'
    ])

if 'goals' not in st.session_state:
    st.session_state.goals = load_goals_db()

# Timer state
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
if 'cluster_id' not in st.session_state:
    st.session_state.cluster_id = None
# Title
st.title("AI-powered Study Plan Generator")
st.markdown("Create optimized study plans based on your habits and AI predictions.")

# Navigation
with st.sidebar:
    st.markdown("### Navigation")
    view_mode = st.radio(
        "Which view would you like to see?",
        options=["Study Plan", "Statistics", "Goal Setting", "About"],
        index=0,
        key="view_mode"
    )

if view_mode == "Study Plan":
    # Sidebar inputs
    st.sidebar.header("Plan your study session")

    total_duration = st.sidebar.slider(
        "How long do you want to study in total?",
        min_value=30,
        max_value=240,
        value=120,
        step=15,
        help="Total duration in minutes"
    )

    time_of_day = st.sidebar.selectbox(
        "Which time of day are you studying?",
        options=['morning', 'afternoon', 'evening', 'night'],
        format_func=lambda x: {
            'morning': '🌅 Morning (6-12h)',
            'afternoon': '☀️ Afternoon (12-18h)',
            'evening': '🌆 Evening (18-22h)',
            'night': '🌙 Night (22-6h)'
        }[x]
    )

    concentration = st.sidebar.slider(
        "How focused do you feel right now?",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
        help="1 = very distracted, 10 = laser focused"
    )

    if len(st.session_state.user_history) > 0:
        last_session = st.session_state.user_history.iloc[-1]['timestamp']
        days_since = (datetime.now() - last_session).days
        st.sidebar.info(f"Last session: {days_since} day(s) ago")
    else:
        days_since = st.sidebar.number_input(
            "How many days ago was your last session?",
            min_value=0,
            max_value=30,
            value=1
        )

    if len(st.session_state.user_history) > 0:
        previous_rating = st.session_state.user_history.iloc[-1]['actual_rating']
        st.sidebar.info(f"Last session rating: {previous_rating}/10")
    else:
        previous_rating = st.sidebar.slider(
            "How well did your last session go?",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.5
        )

    generate_plan = st.sidebar.button("🚀 Generate study plan", type="primary")

    api_status = get_ml_api_status()
    status_label = "🟢 ML API connected" if api_status else "⚪️ ML API offline – using local models"
    st.sidebar.caption(status_label)
    if st.sidebar.button("🔄 Re-check ML API connection"):
        refreshed = get_ml_api_status(force_refresh=True)
        if refreshed:
            st.sidebar.success("ML API available. Future plans will use it automatically.")
        else:
            st.sidebar.warning("Still offline. Start the API server to enable remote predictions.")
else:
    total_duration = None
    time_of_day = None
    concentration = None
    days_since = None
    previous_rating = None
    generate_plan = False

if view_mode == "Study Plan" and generate_plan:
    cluster_id = st.session_state.cluster_id
    if cluster_id is None:
        st.sidebar.info(
            "No Anki data imported yet. Defaulting to the Structured Planner cluster."
        )
        cluster_id = DEFAULT_CLUSTER_ID

    features = pd.DataFrame([{
        'total_session_duration': total_duration,
        'time_morning': 1 if time_of_day == 'morning' else 0,
        'time_afternoon': 1 if time_of_day == 'afternoon' else 0,
        'time_evening': 1 if time_of_day == 'evening' else 0,
        'time_night': 1 if time_of_day == 'night' else 0,
        'concentration_baseline': concentration,
        'days_since_last_session': days_since,
        'previous_session_rating': previous_rating,
        'cluster_id': cluster_id,
    }])
    feature_payload = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in features.iloc[0].items()
    }

    api_prediction = None
    api_error = None
    use_remote_api = get_ml_api_status()
    if use_remote_api:
        try:
            api_prediction = predict_session_via_api(
                feature_payload,
                desired_total_duration=total_duration,
            )
        except PredictionAPIError as err:
            api_error = err
            st.session_state.ml_api_available = False
        except Exception as err:
            api_error = err
            st.session_state.ml_api_available = False

    if api_prediction is not None:
        pred_work = int(api_prediction['work_duration'])
        pred_break = int(api_prediction['break_duration'])
        pred_next = api_prediction['next_session_hours']
        pred_blocks = int(api_prediction['blocks'])
        schedule = api_prediction.get('schedule', [])
        total_calculated = int(api_prediction.get('total_calculated_duration', total_duration))
    else:
        models = st.session_state.models
        if models is None:
            error_msg = "Unable to contact the ML API and no local models are available. Start the API or train local models."
            if api_error:
                error_msg += f"\nDetails: {api_error}"
            st.error(error_msg)
            st.stop()

        feature_columns = models.get('feature_columns')
        if feature_columns:
            missing_cols = [col for col in feature_columns if col not in features.columns]
            for col in missing_cols:
                features[col] = 0
            features = features[feature_columns]

        features_scaled = models['scaler'].transform(features)

        pred_work = int(round(models['work_duration'].predict(features_scaled)[0]))
        pred_break = int(round(models['break_duration'].predict(features_scaled)[0]))
        pred_next = models['next_session'].predict(features_scaled)[0]

        pred_work = max(15, min(45, pred_work))
        pred_break = max(5, min(15, pred_break))

        cycle_duration = pred_work + pred_break
        pred_blocks = max(1, int((total_duration + pred_break) / cycle_duration))

        schedule = []
        total_calculated = 0

        for block in range(pred_blocks):
            schedule.append({
                'type': 'Study',
                'duration': pred_work,
                'block': block + 1
            })
            total_calculated += pred_work

            if block < pred_blocks - 1:
                schedule.append({
                    'type': 'Break',
                    'duration': pred_break,
                    'block': block + 1
                })
                total_calculated += pred_break

        if api_error is not None:
            st.warning(
                "Using local models because the ML API request failed: "
                f"{api_error}"
            )
        elif not get_ml_api_status():
            st.info("Using local models because the ML API is offline.")

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

    st.session_state.timer_running = False
    st.session_state.current_block_index = 0
    st.session_state.timer_paused = False
    st.session_state.pause_time = 0
    st.session_state.show_celebration = False


def render_welcome_content():
    st.header("Welcome to the AI Study Plan Generator")
    st.info("Use the sidebar to build your personalized plan or review statistics.")
    st.markdown("""
    ### How it works

    1. **Enter your parameters** (duration, time of day, focus level)
    2. **Click \"Generate study plan\"**
    3. **Use the interactive timer** with countdown and animations
    4. **Provide feedback after each session** so the AI can learn

    ### What does the AI do?

    The ridge regression looks at:
    - Your focus levels
    - Time of day (chronobiology)
    - Your previous study behavior
    - Recovery time between sessions

    It then suggests:
    - Optimal number and length of study blocks
    - Suitable break durations
    - The best time for your next session
    - Interactive timer controls
    """)


if view_mode == "Goal Setting":
    st.header("🎯 Goal Setting")
    goals_df = st.session_state.goals.copy()
    now_ts = pd.Timestamp.now()
    iso_calendar = now_ts.isocalendar()
    current_week_label = f"{iso_calendar.year}-W{iso_calendar.week:02d}"
    week_start = now_ts.normalize() - pd.Timedelta(days=now_ts.weekday())
    week_end = week_start + pd.Timedelta(days=7)
    existing_goal = goals_df[goals_df['week'] == current_week_label]
    stored_target = existing_goal['target_minutes'].iloc[0] if not existing_goal.empty else None
    default_target = int(stored_target) if stored_target is not None else 240

    st.subheader(f"Weekly goal – {current_week_label}")
    target_input = st.number_input(
        "Target minutes for this week",
        min_value=30,
        max_value=1200,
        step=30,
        value=int(default_target)
    )

    target_for_progress = stored_target
    if st.button("Save goal", type="primary"):
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
        st.success("Weekly goal saved.")

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
        st.metric("Current goal", target_display)
    with col_goal[1]:
        st.metric("Minutes studied this week", f"{int(week_minutes)} min")

    if target_for_progress and target_for_progress > 0:
        progress_pct = min(100, (week_minutes / target_for_progress) * 100)
        st.progress(min(1.0, progress_pct / 100), text=f"{week_minutes:.0f} / {target_for_progress:.0f} minutes")
        st.info(f"You are at {progress_pct:.0f}% of your weekly goal.")
    else:
        st.info("Set a weekly goal to track your progress.")

    st.subheader("Goal history")
    if len(goals_df) == 0:
        st.caption("No goals saved yet.")
    else:
        goal_history = goals_df.copy().sort_values('week', ascending=False)
        goal_history = goal_history.rename(columns={
            'week': 'Calendar week',
            'target_minutes': 'Target minutes'
        })
        goal_history['Target minutes'] = goal_history['Target minutes'].astype(int)

        history_goal_for_table = st.session_state.user_history.copy()
        if len(history_goal_for_table) > 0:
            history_goal_for_table['timestamp_dt'] = pd.to_datetime(
                history_goal_for_table['timestamp'], errors='coerce'
            )
            history_goal_for_table = history_goal_for_table.dropna(subset=['timestamp_dt'])
            iso_weeks = history_goal_for_table['timestamp_dt'].dt.isocalendar()
            history_goal_for_table['week_label'] = (
                iso_weeks['year'].astype(str) + "-W" + iso_weeks['week'].astype(str).str.zfill(2)
            )
            actual_week_minutes = history_goal_for_table.groupby('week_label')['total_duration'].sum()
        else:
            actual_week_minutes = pd.Series(dtype=float)

        goal_history['Minutes studied'] = (
            goal_history['Calendar week'].map(actual_week_minutes).fillna(0).astype(int)
        )
        goal_history['Completion (%)'] = goal_history.apply(
            lambda row: (row['Minutes studied'] / row['Target minutes'] * 100) if row['Target minutes'] > 0 else 0,
            axis=1
        )
        goal_history['Completion (%)'] = goal_history['Completion (%)'].round(0).astype(int).astype(str) + "%"
        st.dataframe(goal_history, use_container_width=True, hide_index=True)
                # --- Progress vs. weekly goal chart ---
        st.subheader("Progress vs weekly goal")

        progress_chart_df = goal_history[['Calendar week', 'Target minutes', 'Minutes studied']].copy()
        progress_chart_df = progress_chart_df.sort_values('Calendar week')

        fig_goal = go.Figure()

        # Ziel-Minuten pro Woche
        fig_goal.add_trace(go.Bar(
            name="Target minutes",
            x=progress_chart_df['Calendar week'],
            y=progress_chart_df['Target minutes'],
            marker=dict(color="#90CAF9"),
            hovertemplate="Week %{x}<br>Target: %{y} min<extra></extra>",
        ))

        # Tatsächlich gelernte Minuten pro Woche
        fig_goal.add_trace(go.Bar(
            name="Minutes studied",
            x=progress_chart_df['Calendar week'],
            y=progress_chart_df['Minutes studied'],
            marker=dict(color="#4CAF50"),
            hovertemplate="Week %{x}<br>Studied: %{y} min<extra></extra>",
        ))

        fig_goal.update_layout(
            barmode="group",
            xaxis_title="Calendar week",
            yaxis_title="Minutes",
            height=350,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=40, r=20, t=40, b=40),
        )

        st.plotly_chart(fig_goal, use_container_width=True)


elif view_mode == "About":
    render_welcome_content()

elif view_mode == "Statistics":
    history = st.session_state.user_history
    st.header("📊 Statistics Dashboard")

    if len(history) == 0:
        st.info("No data yet. Submit feedback after your first study session to build stats.")
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
        col_stats[0].metric("Sessions completed", sessions_completed)
        col_stats[1].metric("Minutes studied today", f"{int(today_minutes)} min")
        col_stats[2].metric("Avg. session rating", f"{avg_rating:.1f}/10")
        col_stats[3].metric("Avg. session duration", f"{avg_duration:.0f} min")
        st.caption(f"Last session: {last_session_str}")

        chart_df = history[['timestamp', 'actual_rating']].copy().sort_values('timestamp')
        chart_df['timestamp'] = chart_df['timestamp'].astype(str)
        chart_df = chart_df.set_index('timestamp')
        st.subheader("Rating trend")
        st.line_chart(chart_df, height=280)
                # --- Dauer vs. Qualität (Sweet Spot der Sessionlänge) ---
        st.subheader("Session length vs. focus rating")

        scatter_df = history[['total_duration', 'actual_rating']].copy()
        scatter_df['total_duration'] = pd.to_numeric(scatter_df['total_duration'], errors='coerce')
        scatter_df['actual_rating'] = pd.to_numeric(scatter_df['actual_rating'], errors='coerce')
        scatter_df = scatter_df.dropna()

        if len(scatter_df) == 0:
            st.caption("Not enough data yet to show this chart.")
        else:
            fig_scatter = go.Figure()

            # Punkte: jede Session
            fig_scatter.add_trace(go.Scatter(
                x=scatter_df['total_duration'],
                y=scatter_df['actual_rating'],
                mode='markers',
                name='Sessions',
                marker=dict(size=9, opacity=0.8),
                hovertemplate="Duration: %{x} min<br>Rating: %{y}/10<extra></extra>",
            ))

            # Einfache Trendlinie (lineare Regression)
            if len(scatter_df) >= 2:
                x_vals = scatter_df['total_duration'].to_numpy()
                y_vals = scatter_df['actual_rating'].to_numpy()
                m, b = np.polyfit(x_vals, y_vals, 1)  # Steigung & Achsenabschnitt

                x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                y_line = m * x_line + b

                fig_scatter.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name='Trend',
                    line=dict(dash='dash'),
                    hoverinfo='skip'
                ))

            fig_scatter.update_layout(
                xaxis_title="Session duration (minutes)",
                yaxis_title="Focus rating (1–10)",
                height=350,
                margin=dict(l=40, r=20, t=10, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )

            st.plotly_chart(fig_scatter, use_container_width=True)


        st.subheader("Session history")
        history_display = history.copy()
        history_display['timestamp'] = pd.to_datetime(history_display['timestamp'], errors='coerce')
        history_display['Date'] = history_display['timestamp'].apply(
            lambda ts: ts.strftime("%d.%m") if pd.notna(ts) else ""
        )
        history_display['Time'] = history_display['timestamp'].apply(
            lambda ts: ts.strftime("%H.%M") if pd.notna(ts) else ""
        )
        history_display = history_display[[
            'Date', 'Time', 'total_duration', 'time_of_day', 'concentration_baseline',
            'days_since_last', 'previous_rating', 'actual_rating', 'feedback'
        ]]
        history_display = history_display.rename(columns={
            'total_duration': 'Duration (min)',
            'time_of_day': 'Time of day',
            'concentration_baseline': 'Focus level',
            'days_since_last': 'Days since last',
            'previous_rating': 'Previous rating',
            'actual_rating': 'Current rating',
            'feedback': 'Feedback'
        })
        st.dataframe(history_display, use_container_width=True, hide_index=True)

        st.subheader("Calendar by time of day & weekday")
        weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        time_labels = ["Morning", "Midday", "Evening", "Night"]
        calendar_df = pd.DataFrame(index=time_labels, columns=weekday_labels, dtype=float)

        history_for_calendar = history.copy()
        history_for_calendar['timestamp'] = pd.to_datetime(history_for_calendar['timestamp'], errors='coerce')

        weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        time_map = {
            'morning': "Morning",
            'afternoon': "Midday",
            'evening': "Evening",
            'night': "Night"
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

        # Formatter für leere Felder
        def calendar_formatter(v):
            if pd.isna(v):
                return "You lazy bum, start studying!"
            return f"{v:.1f}"

        # Styling mit Heatmap (fallback ohne Matplotlib)
        styled_calendar = calendar_df.style
        if MATPLOTLIB_AVAILABLE:
            styled_calendar = styled_calendar.background_gradient(
                axis=None,
                cmap="RdYlGn",
                vmin=1,
                vmax=10
            )
        styled_calendar = (
            styled_calendar
            .applymap(
                lambda v: "background-color: #ffffff" if pd.isna(v) else ""
            )
            .format(calendar_formatter)
        )

        # WICHTIG: st.table statt st.dataframe, sonst wird der Formatter ignoriert
        st.table(styled_calendar)



    st.subheader("Import Anki statistics")
    st.caption("Upload your Anki statistics PDF, we will calculate key metrics and assign you a learning profile.")
    uploaded_file = st.file_uploader("Upload Anki PDF", type=["pdf"], key="anki_pdf_uploader")

    if uploaded_file is not None:
        try:
            pdf_features = extract_features_from_anki_pdf(uploaded_file)

            features_pretty = {
                "total_reviews": pdf_features["total_reviews"],
                "days_active": pdf_features["days_active"],
                "days_total": pdf_features["days_total"],
                "learning_days_ratio": round(pdf_features["learning_days_ratio"], 3),
                "reviews_per_learning_day": round(pdf_features["reviews_per_learning_day"], 1),
                "daily_reviews": round(pdf_features["daily_reviews"], 1),
                "accuracy_pct": round(pdf_features["accuracy"] * 100, 1),
            }

            st.json(features_pretty)

            cluster_id = assign_cluster_id(pdf_features)
            st.session_state.cluster_id = cluster_id

            cluster_key = assign_cluster_from_features(pdf_features)
            profile = CLUSTERS[cluster_key]

            st.success(f"{profile.name}")
            st.write(profile.description)

        except Exception as e:
            st.error(f"Error while reading the PDF: {e}")

else:
    if 'current_plan' in st.session_state:
        plan = st.session_state.current_plan

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Study blocks", f"{plan['blocks']}")

        with col2:
            st.metric("Study block duration", f"{plan['work_duration']} min")

        with col3:
            st.metric("Break duration", f"{plan['break_duration']} min")

        with col4:
            st.metric("Actual duration", f"{plan['actual_duration']} min")

        with col5:
            st.metric("Next session in", f"{plan['next_session_hours']:.1f} h")

        st.markdown("---")

        if st.session_state.show_celebration:
            st.balloons()
            st.success("🎉 Great job! Block completed!")
            st.session_state.show_celebration = False

        schedule = plan['schedule']
        current_idx = st.session_state.current_block_index

        if current_idx < len(schedule):
            current_item = schedule[current_idx]

            st.subheader("Timer")

            progress = current_idx / len(schedule) if len(schedule) > 0 else 0
            st.progress(progress, text=f"Block {current_idx + 1} of {len(schedule)}")

            col_timer1, col_timer2 = st.columns([2, 1])

            with col_timer1:
                if current_item['type'] == 'Study':
                    st.markdown(f"### Study block {current_item['block']}")
                    timer_color = "#4CAF50"
                else:
                    st.markdown(f"### Break after block {current_item['block']}")
                    timer_color = "#FF9800"

            if st.session_state.timer_running and not st.session_state.timer_paused:
                elapsed = (time.time() - st.session_state.timer_start_time) - st.session_state.pause_time
                remaining_seconds = max(0, current_item['duration'] * 60 - elapsed)
            elif st.session_state.timer_paused:
                remaining_seconds = st.session_state.remaining_at_pause
            else:
                remaining_seconds = current_item['duration'] * 60

            minutes = int(remaining_seconds // 60)
            seconds = int(remaining_seconds % 60)

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
                        if st.button("▶️ Resume", use_container_width=True, key="continue_btn"):
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
                if st.button("⏹️ End", use_container_width=True, key="stop_btn"):
                    st.session_state.current_block_index = 0
                    st.session_state.timer_running = False
                    st.session_state.timer_paused = False
                    st.rerun()

            if remaining_seconds <= 0 and st.session_state.timer_running:
                st.warning("⏰ Time's up! Click 'Continue to next block'.")

                if st.button("➡️ Continue to next block", use_container_width=True, type="primary", key="next_block_btn"):
                    st.session_state.show_celebration = True
                    st.session_state.current_block_index += 1
                    st.session_state.timer_running = False
                    st.session_state.timer_paused = False
                    st.session_state.pause_time = 0
                    st.rerun()

        else:
            st.success("🎊 Congrats! You've finished all study blocks!")
            st.balloons()
            if st.button("🔄 Start a new session", key="new_session_btn"):
                st.session_state.current_block_index = 0
                st.session_state.timer_running = False
                st.session_state.timer_paused = False
                st.rerun()

        st.markdown("---")

        st.subheader("Your study plan in detail")

        schedule_display = []

        for i, item in enumerate(schedule):
            status = "✅" if i < current_idx else ("🔄" if i == current_idx else "⏳")
            schedule_display.append({
                'No.': i + 1,
                'Status': status,
                'Activity': item['type'],
                'Duration': f"{item['duration']} min"
            })

        st.dataframe(
            pd.DataFrame(schedule_display),
            use_container_width=True,
            hide_index=True
        )

               # --- Neue Timeline-Visualisierung mit echter Uhrzeit-Achse ---

        fig = go.Figure()

        study_x, study_base, study_y = [], [], []
        break_x, break_base, break_y = [], [], []

        current_start = 0  # Minuten seit Beginn der Session

        for item in schedule:
            duration = item["duration"]

            if item["type"] == "Study":
                study_x.append(duration)
                study_base.append(current_start)
                study_y.append("Session")
            else:
                break_x.append(duration)
                break_base.append(current_start)
                break_y.append("Session")

            current_start += duration

        total_minutes = current_start

        # Startzeit: aktuelle Uhrzeit (auf volle Minute gerundet)
        session_start = datetime.now().replace(second=0, microsecond=0)

        # Ticks für die Zeitachse berechnen
        if total_minutes <= 60:
            tick_step = 10   # alle 10 Minuten
        elif total_minutes <= 180:
            tick_step = 30   # alle 30 Minuten
        else:
            tick_step = 60   # stündlich

        tickvals = list(range(0, total_minutes + 1, tick_step))
        ticktext = [
            (session_start + timedelta(minutes=m)).strftime("%H:%M")
            for m in tickvals
        ]

        # Study-Segmente
        if study_x:
            fig.add_trace(go.Bar(
                name="Study",
                x=study_x,
                y=study_y,
                base=study_base,           # Startpunkt auf der Zeitachse (in Minuten)
                orientation="h",
                marker=dict(color="#4CAF50"),
                text=[f"Study {x} min" for x in study_x],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate="Study: %{x} min<br>Start: %{base} min nach Beginn<extra></extra>",
            ))

        # Break-Segmente
        if break_x:
            fig.add_trace(go.Bar(
                name="Break",
                x=break_x,
                y=break_y,
                base=break_base,
                orientation="h",
                marker=dict(color="#FF9800"),
                text=[f"Break {x} min" for x in break_x],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate="Break: %{x} min<br>Start: %{base} min nach Beginn<extra></extra>",
            ))

        fig.update_layout(
            title="Timeline of your study session",
            xaxis_title="Time",
            yaxis_title="",
            barmode="overlay",
            height=260,                  # <-- mac
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[0, total_minutes],
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        time_diff = abs(plan['total_duration'] - plan['actual_duration'])
        if time_diff > 5:
            st.info(f"ℹ️ The actual session duration ({plan['actual_duration']} min) differs from your desired duration ({plan['total_duration']} min). This is due to optimizing study block lengths for maximum efficiency.")

        st.subheader("Personalized tips")

        tips = []
        if plan['concentration'] < 5:
            tips.append("⚠️ Low focus detected. Try shorter study blocks with longer breaks.")
        if plan['time_of_day'] == 'night':
            tips.append("🌙 Late-night studying can be inefficient. Consider an earlier slot if possible.")
        if plan['blocks'] > 5:
            tips.append("🔋 Lots of study blocks planned! Stay hydrated and grab snacks.")
        if plan['next_session_hours'] < 6:
            tips.append("⏰ A short gap until the next session is suggested. Make sure to recover properly!")

        if tips:
            for tip in tips:
                st.info(tip)
        else:
            st.success("✅ Your plan looks great! Good luck!")

        st.subheader("Session feedback")
        st.markdown("*After your study session you can provide feedback so the AI improves further.*")

        with st.form("feedback_form"):
            actual_rating = st.slider(
                "How good was your focus during this session?",
                min_value=1.0,
                max_value=10.0,
                value=7.0,
                step=0.5
            )

            feedback_reasons = st.multiselect(
                "If it was not ideal, what were the reasons?",
                options=[
                    "Study blocks were too long",
                    "Breaks were too short",
                    "Too late in the day",
                    "Too early in the day",
                    "Not enough sleep",
                    "Distractions",
                    "Topic was difficult",
                    "Other"
                ]
            )

            submitted = st.form_submit_button("💾 Save feedback")

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

                # 1) Keep in-memory history (for current session & charts)
                st.session_state.user_history = pd.concat(
                    [st.session_state.user_history, new_entry],
                    ignore_index=True
                )

                # 2) ALSO store the session in the SQLite database
                db.create_session(
                    user_id=DEFAULT_USER_ID,
                    duration_minutes=int(plan['total_duration']),   # full session length
                    pause_minutes=int(plan['break_duration']),      # break length from plan
                    focus_score=int(round(actual_rating)),          # user rating 1–10
                    subject=None,                                   # or pass a subject string if you add one
                    start_time=datetime.now(),                      # when feedback is saved
                    end_time=None,
                    source="app"                                    # label to know it came from the app
                )

                st.success("✅ Feedback saved! The AI learns with every submission.")

    else:
        render_welcome_content()
