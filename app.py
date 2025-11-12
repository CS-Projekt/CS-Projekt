import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go

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

# Initialisierung
if 'models' not in st.session_state:
    st.session_state.models = load_models()

if 'user_history' not in st.session_state:
    st.session_state.user_history = pd.DataFrame(columns=[
        'timestamp', 'total_duration', 'time_of_day', 'concentration_baseline',
        'days_since_last', 'previous_rating', 'actual_rating', 'feedback'
    ])

# Prüfen ob Modelle geladen wurden
if st.session_state.models is None:
    st.stop()

# Titel
st.title("📚 AI-gestützter Lernplan Generator")
st.markdown("Erstelle optimierte Lernpläne basierend auf deinem Lernverhalten und KI-Vorhersagen")

# Sidebar für User-Input
st.sidebar.header("🎯 Deine Lernsession planen")

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
if st.sidebar.button("🚀 Lernplan generieren", type="primary"):
    
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
    pred_blocks = int(round(models['work_blocks'].predict(features_scaled)[0]))
    pred_work = int(round(models['work_duration'].predict(features_scaled)[0]))
    pred_break = int(round(models['break_duration'].predict(features_scaled)[0]))
    pred_next = models['next_session'].predict(features_scaled)[0]
    
    # Sicherstellen dass Vorhersagen sinnvoll sind
    pred_blocks = max(1, pred_blocks)
    pred_work = max(15, min(45, pred_work))
    pred_break = max(5, min(15, pred_break))
    
    # In Session State speichern
    st.session_state.current_plan = {
        'blocks': pred_blocks,
        'work_duration': pred_work,
        'break_duration': pred_break,
        'next_session_hours': pred_next,
        'total_duration': total_duration,
        'time_of_day': time_of_day,
        'concentration': concentration
    }

# Hauptbereich: Lernplan anzeigen
if 'current_plan' in st.session_state:
    plan = st.session_state.current_plan
    
    # Metriken anzeigen
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔢 Lernblöcke", f"{plan['blocks']}")
    
    with col2:
        st.metric("⏱️ Lernblock-Dauer", f"{plan['work_duration']} min")
    
    with col3:
        st.metric("☕ Pausen-Dauer", f"{plan['break_duration']} min")
    
    with col4:
        next_session_time = datetime.now() + timedelta(hours=plan['next_session_hours'])
        st.metric("📅 Nächste Session", next_session_time.strftime("%H:%M Uhr"))
    
    # Zeitplan visualisieren
    st.subheader("📊 Dein Lernplan im Detail")
    
    # Zeitstrahl erstellen
    current_time = datetime.now()
    schedule = []
    
    for block in range(plan['blocks']):
        # Lernblock
        schedule.append({
            'type': 'Lernen',
            'start': current_time,
            'duration': plan['work_duration'],
            'block': block + 1
        })
        current_time += timedelta(minutes=plan['work_duration'])
        
        # Pause (nicht nach dem letzten Block)
        if block < plan['blocks'] - 1:
            schedule.append({
                'type': 'Pause',
                'start': current_time,
                'duration': plan['break_duration'],
                'block': block + 1
            })
            current_time += timedelta(minutes=plan['break_duration'])
    
    # Zeitplan-Tabelle
    schedule_df = pd.DataFrame([
        {
            'Nr.': i + 1,
            'Aktivität': item['type'],
            'Start': item['start'].strftime('%H:%M'),
            'Ende': (item['start'] + timedelta(minutes=item['duration'])).strftime('%H:%M'),
            'Dauer': f"{item['duration']} min"
        }
        for i, item in enumerate(schedule)
    ])
    
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)
    
    # Gantt-Chart
    fig = go.Figure()
    
    for i, item in enumerate(schedule):
        color = '#4CAF50' if item['type'] == 'Lernen' else '#FF9800'
        fig.add_trace(go.Bar(
            name=item['type'],
            x=[item['duration']],
            y=[i],
            orientation='h',
            marker=dict(color=color),
            text=f"{item['type']}: {item['duration']} min",
            textposition='inside',
            showlegend=i == 0 or (i == 1 and item['type'] == 'Pause')
        ))
    
    fig.update_layout(
        title="Zeitlicher Ablauf deiner Lernsession",
        xaxis_title="Dauer (Minuten)",
        yaxis_title="",
        barmode='stack',
        height=400,
        yaxis=dict(showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tipps basierend auf Vorhersagen
    st.subheader("💡 Personalisierte Tipps")
    
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
    st.subheader("📝 Session-Feedback")
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
            # Feedback zur History hinzufügen
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
    # Willkommensbildschirm
    st.info("👈 Nutze die Sidebar, um deinen personalisierten Lernplan zu erstellen!")
    
    st.markdown("""
    ### 🎯 So funktioniert's:
    
    1. **Gib deine Parameter ein** (Dauer, Tageszeit, Konzentration)
    2. **Klicke auf "Lernplan generieren"**
    3. **Erhalte deinen optimierten Zeitplan** mit KI-Empfehlungen
    4. **Gib nach der Session Feedback** um die KI zu trainieren
    
    ### 🧠 Was macht die KI?
    
    Die Ridge Regression analysiert:
    - ✅ Deine Konzentrationsfähigkeit
    - ✅ Die Tageszeit (Chronobiologie)
    - ✅ Dein bisheriges Lernverhalten
    - ✅ Erholungszeiten zwischen Sessions
    
    Und empfiehlt dir:
    - 📊 Optimale Anzahl und Länge der Lernblöcke
    - ☕ Passende Pausenzeiten
    - 📅 Den besten Zeitpunkt für die nächste Session
    """)

# Footer mit Stats
if len(st.session_state.user_history) > 0:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Deine Statistiken")
    st.sidebar.metric("Absolvierte Sessions", len(st.session_state.user_history))
    avg_rating = st.session_state.user_history['actual_rating'].mean()
    st.sidebar.metric("Durchschnittliches Rating", f"{avg_rating:.1f}/10")