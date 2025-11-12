import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Seed für Reproduzierbarkeit
np.random.seed(42)

def generate_learning_sessions(n_samples=500):
    """
    Generiert synthetische Lernsession-Daten basierend auf
    Lernforschung und Pomodoro-Prinzipien
    """
    data = []
    
    for _ in range(n_samples):
        # INPUT FEATURES
        # Gesamte geplante Session-Dauer (30 min bis 240 min = 4 Stunden)
        total_duration = np.random.choice([30, 60, 90, 120, 150, 180, 210, 240])
        
        # Tageszeit (0=Morgen, 1=Nachmittag, 2=Abend, 3=Nacht)
        time_of_day = np.random.choice([0, 1, 2, 3], p=[0.3, 0.35, 0.25, 0.1])
        time_labels = ['morning', 'afternoon', 'evening', 'night']
        
        # Baseline Konzentrationsfähigkeit des Users (1-10)
        concentration_baseline = np.random.uniform(4, 9)
        
        # Tage seit letzter Session (0-7 Tage)
        days_since_last = np.random.randint(0, 8)
        
        # Rating der vorherigen Session (1-10)
        previous_rating = np.random.uniform(3, 9)
        
        # OUTPUT LABELS (basierend auf Forschung simuliert)
        # Tageszeit-Faktor für Effizienz
        time_factor = [1.2, 1.0, 0.8, 0.5][time_of_day]  # Morgen am besten
        
        # Pause-Faktor basierend auf Erholung
        rest_factor = min(1.0, days_since_last / 3.0)  # Max bei 3+ Tagen Pause
        
        # Basis-Effizienz berechnen
        base_efficiency = (concentration_baseline / 10) * time_factor * (0.7 + 0.3 * rest_factor)
        base_efficiency += (previous_rating / 50)  # Vorherige Erfolge helfen
        base_efficiency = np.clip(base_efficiency, 0.3, 1.0)
        
        # Optimale Arbeitsblock-Länge (Pomodoro: 25 min, aber variabel)
        if concentration_baseline > 7:
            work_block_base = 30  # Hohe Konzentration = längere Blöcke
        elif concentration_baseline > 5:
            work_block_base = 25  # Standard Pomodoro
        else:
            work_block_base = 20  # Niedrige Konzentration = kürzere Blöcke
        
        # Anpassung basierend auf Tageszeit
        work_block_duration = int(work_block_base * time_factor)
        work_block_duration = np.clip(work_block_duration, 15, 45)
        
        # Pausen-Länge (Standard 5 min, aber länger bei niedrigerer Konzentration)
        break_duration = int(5 + (10 - concentration_baseline) * 1.5)
        break_duration = np.clip(break_duration, 5, 15)
        
        # Anzahl der Arbeitsblöcke
        cycle_duration = work_block_duration + break_duration
        optimal_blocks = max(1, int(total_duration / cycle_duration))
        
        # Konzentrations-Score der Session (wie gut lief es?)
        concentration_score = base_efficiency * 10
        # Zu lange Sessions reduzieren Score
        if total_duration > 150:
            concentration_score *= 0.85
        # Zu wenig Pause reduziert Score
        if days_since_last == 0:
            concentration_score *= 0.9
        
        concentration_score = np.clip(concentration_score, 2, 10)
        
        # Empfehlung für nächste Session (in Stunden)
        if concentration_score > 7:
            next_session_hours = np.random.uniform(4, 8)  # Bei gutem Score: bald wieder
        elif concentration_score > 5:
            next_session_hours = np.random.uniform(6, 12)
        else:
            next_session_hours = np.random.uniform(12, 24)  # Bei schlechtem Score: mehr Pause
        
        # Rauschfaktor für Realismus
        concentration_score += np.random.normal(0, 0.5)
        concentration_score = np.clip(concentration_score, 1, 10)
        
        data.append({
            # Features
            'total_session_duration': total_duration,
            'time_of_day': time_labels[time_of_day],
            'time_of_day_encoded': time_of_day,
            'concentration_baseline': round(concentration_baseline, 2),
            'days_since_last_session': days_since_last,
            'previous_session_rating': round(previous_rating, 2),
            
            # Labels
            'optimal_work_blocks': optimal_blocks,
            'work_block_duration': work_block_duration,
            'break_duration': break_duration,
            'concentration_score': round(concentration_score, 2),
            'next_session_recommendation_hours': round(next_session_hours, 2)
        })
    
    return pd.DataFrame(data)

# Daten generieren
print("🔄 Generiere synthetische Trainingsdaten...")
df = generate_learning_sessions(n_samples=500)

# Speichern
df.to_csv('learning_sessions_data.csv', index=False)
print(f"✅ {len(df)} Trainingsbeispiele erstellt und gespeichert!")
print("\n📊 Erste 5 Zeilen:")
print(df.head())
print("\n📈 Statistiken:")
print(df.describe())