import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time
import os
import streamlit.components.v1 as components
import database as db
from anki_pdf_analysis import extract_features_from_anki_pdf
from clusters import (
    assign_cluster_from_features,
    assign_cluster_id,
    CLUSTERS,
    ClusterKey,
    CLUSTER_ID_TO_KEY,
)
from ml_models import load_models as load_ridge_models, predict_plan as predict_ridge_plan, train_models_from_db
from train_clustering import train_and_save_clustering

try:
    import matplotlib  
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Set up the title page
st.set_page_config(
    page_title="Machine Learning Study Plan Generator",
    layout="wide"
)

#Create the Database
db.init_db()
DEFAULT_USER_ID = db.get_or_create_default_user()
GOALS_DB_FILE = "goals.csv"
DEFAULT_CLUSTER_ID = 1

#This Section is created by the help of CODEX (ChatGPT). It ensures that the Goals are saved and loaded and to initialize the ML models once.
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


def ensure_models_initialized():
    if 'models' not in st.session_state:
        try:
            st.session_state.models = load_ridge_models()
        except FileNotFoundError:
            st.session_state.models = None
#End of AI Section

#Build the welcome page of our website
def render_welcome_content():
    st.header("Welcome to the Machine Learning Study Plan Generator")
    st.info("The sidebar is your command center. Pick a view, enter your data, and let the app guide you step by step.")
    st.markdown("""
    ### What this site does

    - **Study Plan**: Collects your current mood, time of day, and recent sessions. A timer and schedule help you work through the proposed plan.
    - **Evaluation**: Import your Anki statistics or manually pick a cluster profile that feeds into every ML prediction.
    - **Statistics**: Shows how your ratings and session lengths evolve, plus a heatmap of when you tend to study.
    - **Goal Setting**: Lets you define weekly targets and compares them with your actual study minutes.

    ### How the Machine Learning works

    Behind the scenes a Ridge Regression model predicts your block length, break duration,
    number of study blocks, and when you should tackle the next session. The model is trained
    on the historical sessions stored in your local SQLite database. Use the sidebar button
    to retrain it whenever new learning samples are available.
                
    """)


# Initializes the session data: creates an empty user_history table and loads goals from file if needed,
# and ensure_models_initialized() makes sure the ML models are loaded once per session. Created by AI
if 'user_history' not in st.session_state:
    st.session_state.user_history = pd.DataFrame(columns=[
        'timestamp', 'total_duration', 'time_of_day', 'concentration_baseline',
        'days_since_last', 'previous_rating', 'actual_rating', 'feedback'
    ])

if 'goals' not in st.session_state:
    st.session_state.goals = load_goals_db()

ensure_models_initialized()
#End of AI Section

#Initialize the Timer states (timer_running, timer_start_time, current_block_index, pause_time, show_celebration_ remaining_at_pause, cluster_id) once 
#to ensure that later in the Code (line 668 and following) the timer is working. 
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
    st.session_state.cluster_id = DEFAULT_CLUSTER_ID

# Title
st.title("Machine Learning powered Study Plan Generator")
st.markdown("Create optimized study plans based on your habits and Machine Learning predictions.")

# Build a userfriendly Navitation Section on the Sidebar. 
with st.sidebar: 
    st.markdown("### Navigation")
    view_mode = st.radio(
        "Which view would you like to see?",
        options=["Study Plan", "Evaluation", "Statistics", "Goal Setting", "About"],
        index=0,
        key="view_mode"
    )

if view_mode == "Study Plan":
    # Setup the Title for the Sidebar
    st.sidebar.header("Plan your study session")

#Create the slider control on the sidebar to plan your learning session.
    total_duration = st.sidebar.slider(
        "How long do you want to study in total?",
        min_value=30,
        max_value=240,
        value=120,
        step=15,
        help="Total duration in minutes"
    )
#Choose on which time of the day you want to study
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
#Choose how high your concentration level is
    concentration = st.sidebar.slider(
        "How focused do you feel right now?",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
        help="1 = very distracted, 10 = laser focused"
    )
#Create the option to choose how many days ago your last session was. Build by help of AI
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

#Give the user the option to evaluate his last session
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

#Insert a button to retrain the ML model and to generate the study plan
    if st.sidebar.button("🔁 Retrain ML model"):
        with st.spinner("Training models..."):
            try:
                models = train_models_from_db()
                st.session_state.models = models
                train_and_save_clustering()
                st.sidebar.success("Models and clustering retrained.")
            except Exception as exc:
                st.sidebar.error(f"Training failed: {exc}")

    generate_plan = st.sidebar.button("Generate study plan", type="primary")

    if st.session_state.models is None:
        st.sidebar.caption("⚪️ Kein Modell geladen – bitte auf 'Retrain ML model' klicken.")
    else:
        st.sidebar.caption("🟢 ML-Modell geladen – Vorhersagen bereit.")

#Learning type is shown on the sidebar
    current_cluster_id = st.session_state.get('cluster_id', DEFAULT_CLUSTER_ID)
    cluster_key = CLUSTER_ID_TO_KEY.get(current_cluster_id, ClusterKey.PLANNER)
    profile = CLUSTERS.get(cluster_key, CLUSTERS[ClusterKey.PLANNER])
    st.sidebar.info(f"Cluster: {profile.name}")
else:
    total_duration = None
    time_of_day = None
    concentration = None
    days_since = None
    previous_rating = None
    generate_plan = False

# When the user is in "Study Plan" view and has clicked "Generate study plan",
# this block loads the trained ML models, shows an error if none exist,
# and otherwise uses the current cluster and input features to predict and display a study plan.
#This Section was created by the help of AI
if view_mode == "Study Plan" and generate_plan:
    models = st.session_state.get('models')
    if models is None:
        st.error("There is no trained model available. Please run ‘Retrain ML model’ first.")
    else:
        active_cluster_id = st.session_state.get('cluster_id') or DEFAULT_CLUSTER_ID
        feature_payload = {
            'total_session_duration': total_duration,
            'time_of_day': time_of_day,
            'concentration_baseline': concentration,
            'days_since_last_session': days_since,
            'previous_session_rating': previous_rating,
            'cluster_id': active_cluster_id,
        }
        try:
            prediction = predict_ridge_plan(
                models=models,
                features=feature_payload,
                desired_total_duration=total_duration,
            )
        except Exception as exc:
            st.error(f"Vorhersage fehlgeschlagen: {exc}")
            st.stop()

        st.session_state.current_plan = {
            **prediction,
            'time_of_day': time_of_day,
            'concentration': concentration,
            'days_since_last': days_since,
            'previous_rating': previous_rating,
            'cluster_id': active_cluster_id,
        }

        st.session_state.timer_running = False
        st.session_state.current_block_index = 0
        st.session_state.timer_paused = False
        st.session_state.pause_time = 0
        st.session_state.show_celebration = False
#End of AI supported Section 

#This section was partially created with AI. 
#The block calculates the current calendar week, displays/edits the learning goal (minutes per week) for that specific week, and saves it permanently.
if view_mode == "Goal Setting":
    st.header("Goal Setting")
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

#block was created by the support of AI. Not every line, but we had to correct some errors and look up some functions
#Setup the weekly goal chart
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

        # Goal for minutes studied in a week
        fig_goal.add_trace(go.Bar(
            name="Target minutes",
            x=progress_chart_df['Calendar week'],
            y=progress_chart_df['Target minutes'],
            marker=dict(color="#90CAF9"),
            hovertemplate="Week %{x}<br>Target: %{y} min<extra></extra>",
        ))

        # Actual minutes studied per week
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
    st.header("Statistics Dashboard")
#If there is no date show a information sign, otherwise the code should calculate the numbers.
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
#Duration vs. quality (sweet spot of session length)
        st.subheader("Session length vs. focus rating")

        scatter_df = history[['total_duration', 'actual_rating']].copy()
        scatter_df['total_duration'] = pd.to_numeric(scatter_df['total_duration'], errors='coerce')
        scatter_df['actual_rating'] = pd.to_numeric(scatter_df['actual_rating'], errors='coerce')
        scatter_df = scatter_df.dropna()

        if len(scatter_df) == 0:
            st.caption("Not enough data yet to show this chart.")
        else:
            fig_scatter = go.Figure()

            # Build a Scatter-Trace in the Stastics view
            fig_scatter.add_trace(go.Scatter(
                x=scatter_df['total_duration'],
                y=scatter_df['actual_rating'],
                mode='markers',
                name='Sessions',
                marker=dict(size=9, opacity=0.8),
                hovertemplate="Duration: %{x} min<br>Rating: %{y}/10<extra></extra>",
            ))

            # Setup a simple trend line (linear regression)
            if len(scatter_df) >= 2:
                x_vals = scatter_df['total_duration'].to_numpy()
                y_vals = scatter_df['actual_rating'].to_numpy()
                m, b = np.polyfit(x_vals, y_vals, 1)  # Slope & axis section

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
#End of AI Section
#Prepare the chart with the most important fields and formatted date/time
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
 #Create an empty grid and fill it with ratings (Days, Time)
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

        # Formatter for empty fields
        def calendar_formatter(v):
            if pd.isna(v):
                return "You lazy bum, start studying!"
            return f"{v:.1f}"

        # Create a Heatmap and style calender wiht colors
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

        st.table(styled_calendar)


# ----- Evaluation view -----
elif view_mode == "Evaluation":
    st.header("🧠 Evaluation & Cluster Profile")
    st.subheader("Import Anki statistics")
    st.caption("Upload your Anki statistics PDF. We will calculate key metrics and assign you a learning profile that influences the ML model.")
    uploaded_file = st.file_uploader("Upload Anki PDF", type=["pdf"], key="anki_pdf_uploader")
# PDF upload and processing
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
            st.info(profile.recommendation)

        except Exception as e:
            st.error(f"Error while reading the PDF: {e}")

# Self evaluation and cluster selection
    st.markdown("---")
    st.subheader("Self evaluation")
    st.write(
        "Already know which learning style fits you best? "
        "Choose one of the clusters below to override the current assignment."
    )

    cluster_descriptions = {
        ClusterKey.SPRINTER: "Studies often in short bursts with high review counts. Works best with quick cycles.",
        ClusterKey.MARATHONER: "Prefers rare but long and intense sessions. High recall and long intervals.",
        ClusterKey.PLANNER: "Keeps a steady schedule with mid-sized blocks and consistent study habits.",
    }

# Cluster selection
    cluster_keys = list(CLUSTERS.keys())
    default_key = CLUSTER_ID_TO_KEY.get(st.session_state.cluster_id, ClusterKey.PLANNER)
    selected_key = st.radio(
        "Choose your learning profile",
        options=cluster_keys,
        format_func=lambda key: f"{CLUSTERS[key].name}: {cluster_descriptions[key]}",
        index=cluster_keys.index(default_key) if default_key in cluster_keys else 0
    )

# save selection
    if st.button("✅ Apply selection"):
        selected_id = next((cid for cid, key in CLUSTER_ID_TO_KEY.items() if key == selected_key), DEFAULT_CLUSTER_ID)
        st.session_state.cluster_id = selected_id
        profile = CLUSTERS[selected_key]
        st.success(f"Cluster set to {profile.name}. {profile.recommendation}")


# ----- Study Plan view -----
elif view_mode == "Study Plan":
    if 'current_plan' in st.session_state:
        plan = st.session_state.current_plan

# Display plan details
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

# Celebration
        if st.session_state.show_celebration:
            st.balloons()
            st.success("Great job! Block completed!")
            st.session_state.show_celebration = False

        schedule = plan['schedule']
        current_idx = st.session_state.current_block_index

# Timer (this part was generated partally by the support of AI)
        if current_idx < len(schedule):
            current_item = schedule[current_idx]

            st.subheader("Timer")

            progress = current_idx / len(schedule) if len(schedule) > 0 else 0
            st.progress(progress, text=f"Block {current_idx + 1} of {len(schedule)}")

            col_timer1, col_timer2 = st.columns([2, 1])

            with col_timer1:
                if current_item['type'] == 'Study':
                    st.markdown(f"### Study block {current_item['block']}")
                    timer_color = "#4CAF50"                 # The color code is from AI
                else:
                    st.markdown(f"### Break after block {current_item['block']}")
                    timer_color = "#FF9800"                 # The color code is from AI

            if st.session_state.timer_running and not st.session_state.timer_paused:
                elapsed = (time.time() - st.session_state.timer_start_time) - st.session_state.pause_time
                remaining_seconds = max(0, current_item['duration'] * 60 - elapsed)
            elif st.session_state.timer_paused:
                remaining_seconds = st.session_state.remaining_at_pause
            else:
                remaining_seconds = current_item['duration'] * 60

            minutes = int(remaining_seconds // 60)
            seconds = int(remaining_seconds % 60)

# Auto-updating timer display (this part was generated by the support of AI)
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

# Control buttons (Start, Pause, Resume, Skip, Reset, End)
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
            st.success("Congrats! You've finished all study blocks!")
            st.balloons()
            if st.button("🔄 Start a new session", key="new_session_btn"):
                st.session_state.current_block_index = 0
                st.session_state.timer_running = False
                st.session_state.timer_paused = False
                st.rerun()

        st.markdown("---")

# Display detailed study plan
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

# Timeline chart with the study plan times
        fig = go.Figure()

        study_x, study_base, study_y = [], [], []
        break_x, break_base, break_y = [], [], []

        current_start = 0  # minutes from session start

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

# This line was created by the support of AI (to round to whole minutes)
        session_start = datetime.now().replace(second=0, microsecond=0)

        # Tick values and for the x-axis
        if total_minutes <= 60:
            tick_step = 10   # every 10 minutes
        elif total_minutes <= 180:
            tick_step = 30   # every 30 minutes
        else:
            tick_step = 60   # every 60 minutes

        tickvals = list(range(0, total_minutes + 1, tick_step))
        ticktext = [
            (session_start + timedelta(minutes=m)).strftime("%H:%M")
            for m in tickvals
        ]

        # Study parts
        if study_x:
            fig.add_trace(go.Bar(
                name="Study",
                x=study_x,
                y=study_y,
                base=study_base,           
                orientation="h",
                marker=dict(color="#4CAF50"),
                text=[f"Study {x} min" for x in study_x],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate="Study: %{x} min<br>Start: %{base} min after start<extra></extra>", # this line was created by the support of AI
            ))

        # Break parts
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
                hovertemplate="Break: %{x} min<br>Start: %{base} min after start<extra></extra>", # here we just adapted it from line 871 that was created by the support of AI
            ))

# Layout adjustments
        fig.update_layout(
            title="Timeline of your study session",
            xaxis_title="Time",
            yaxis_title="",
            barmode="overlay",
            height=260,                  
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[0, total_minutes],
        )

# Display the chart
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        time_diff = abs(plan['total_duration'] - plan['actual_duration'])
        if time_diff > 5:
            st.info(f"The actual session duration ({plan['actual_duration']} min) differs from your desired duration ({plan['total_duration']} min). This is due to optimizing study block lengths for maximum efficiency.")

# Personalized tips based on the plan
        st.subheader("Personalized tips")

        tips = []
        if plan['concentration'] < 5:
            tips.append("Low focus detected. Try shorter study blocks with longer breaks.")
        if plan['time_of_day'] == 'night':
            tips.append("Late-night studying can be inefficient. Consider an earlier slot next time.")
        if plan['blocks'] > 5:
            tips.append("Lots of study blocks planned! Stay hydrated and grab snacks.")
        if plan['next_session_hours'] < 6:
            tips.append("A short gap until the next session is suggested. Make sure to recover properly!")

        if tips:
            for tip in tips:
                st.info(tip)
        else:
            st.success("✅ Your plan looks great! Good luck!")

# Feedback subheader
        st.subheader("Session feedback")
        st.markdown("*After your study session you can provide feedback so the Machine Learning improves further.*")

# Feedback form
        with st.form("feedback_form"):
            actual_rating = st.slider(
                "How good was your focus during this session?",
                min_value=1.0,
                max_value=10.0,
                value=7.0,
                step=0.5
            )
# reasons for feedback
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

# Save and submit feedback button
            submitted = st.form_submit_button("Save feedback")

            if submitted:
                new_entry = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'total_duration': plan['total_duration'],
                    'time_of_day': plan['time_of_day'],
                    'concentration_baseline': plan['concentration'],
                    'days_since_last': plan.get('days_since_last', days_since),
                    'previous_rating': plan.get('previous_rating', previous_rating),
                    'actual_rating': actual_rating,
                    'feedback': ', '.join(feedback_reasons)
                }])

                # Keep in-memory history (for current session & charts) (this in-memory history part was created by the support of AI)
                st.session_state.user_history = pd.concat(
                    [st.session_state.user_history, new_entry],
                    ignore_index=True
                )

                # store the session in the SQLite database
                db.create_session(
                    user_id=DEFAULT_USER_ID,
                    duration_minutes=int(plan['total_duration']),   
                    pause_minutes=int(plan['break_duration']),      
                    focus_score=int(round(actual_rating)),          
                    subject=None,                                  
                    start_time=datetime.now(),                      
                    end_time=None,
                    source="app"                                    
                )
                time_encoding = {
                    'morning': 0,
                    'afternoon': 1,
                    'evening': 2,
                    'night': 3,
                }
                learning_sample = {
                    'total_session_duration': int(plan['total_duration']),
                    'time_of_day': plan['time_of_day'],
                    'time_of_day_encoded': time_encoding.get(plan['time_of_day'], -1),
                    'concentration_baseline': float(plan['concentration']),
                    'days_since_last_session': int(plan.get('days_since_last', 0) or 0),
                    'previous_session_rating': float(plan.get('previous_rating', previous_rating) or 0),
                    'optimal_work_blocks': int(plan['blocks']),
                    'work_block_duration': int(plan['work_duration']),
                    'break_duration': int(plan['break_duration']),
                    'concentration_score': float(actual_rating),
                    'next_session_recommendation_hours': float(plan['next_session_hours']),
                    'cluster_id': int(plan.get('cluster_id', DEFAULT_CLUSTER_ID)),
                }
                # Retrain the ML + clustering models with the new sample: here AI helped us a little bit, because we had some errors :(
                try:
                    db.insert_learning_sample(learning_sample)
                    with st.spinner("Updating models with your new data..."):
                        updated_models = train_models_from_db()
                        st.session_state.models = updated_models
                        train_and_save_clustering()
                    st.success("Models (planner + clustering) updated with your session!")
                except Exception as exc:
                    st.warning(f"Training sample could not be stored/retrained: {exc}")

                st.success("Feedback saved! The machine-learning pipeline improves with every submission.")
    else:
        render_welcome_content()

else:
    render_welcome_content()
