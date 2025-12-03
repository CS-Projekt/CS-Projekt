import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time
import os
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="AI Study Plan Generator",
    page_icon="📚",
    layout="wide"
)


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

# Stop early if models are missing
if st.session_state.models is None:
    st.stop()

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
else:
    total_duration = None
    time_of_day = None
    concentration = None
    days_since = None
    previous_rating = None
    generate_plan = False

if view_mode == "Study Plan" and generate_plan:
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

    models = st.session_state.models
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

        fig = go.Figure()

        work_blocks_x = []
        work_blocks_y = []
        pause_blocks_x = []
        pause_blocks_y = []

        for i, item in enumerate(schedule):
            if item['type'] == 'Study':
                work_blocks_x.append(item['duration'])
                work_blocks_y.append(len(schedule) - i - 1)
            else:
                pause_blocks_x.append(item['duration'])
                pause_blocks_y.append(len(schedule) - i - 1)

        if work_blocks_x:
            fig.add_trace(go.Bar(
                name='Study',
                x=work_blocks_x,
                y=work_blocks_y,
                orientation='h',
                marker=dict(color='#4CAF50'),
                text=[f"Study {x} min" for x in work_blocks_x],
                textposition='inside',
                hovertemplate='Study: %{x} min<extra></extra>'
            ))

        if pause_blocks_x:
            fig.add_trace(go.Bar(
                name='Break',
                x=pause_blocks_x,
                y=pause_blocks_y,
                orientation='h',
                marker=dict(color='#FF9800'),
                text=[f"Break {x} min" for x in pause_blocks_x],
                textposition='inside',
                hovertemplate='Break: %{x} min<extra></extra>'
            ))

        fig.update_layout(
            title="Timeline of your study session",
            xaxis_title="Duration (minutes)",
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

                st.session_state.user_history = pd.concat(
                    [st.session_state.user_history, new_entry],
                    ignore_index=True
                )

                st.success("✅ Feedback saved! The AI learns with every submission.")

    else:
        render_welcome_content()
