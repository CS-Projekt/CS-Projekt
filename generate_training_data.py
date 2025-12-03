import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Seed for reproducibility
np.random.seed(42)

def generate_learning_sessions(n_samples=500):
    """
    Generates synthetic learning session data based on
    learning research and Pomodoro principles.
    """
    data = []
    
    for _ in range(n_samples):
        # INPUT FEATURES
        # Total planned session duration (30–240 minutes = up to 4 hours)
        total_duration = np.random.choice([30, 60, 90, 120, 150, 180, 210, 240])
        
        # Time of day (0=morning, 1=afternoon, 2=evening, 3=night)
        time_of_day = np.random.choice([0, 1, 2, 3], p=[0.3, 0.35, 0.25, 0.1])
        time_labels = ['morning', 'afternoon', 'evening', 'night']
        
        # Baseline concentration ability of the user (1–10)
        concentration_baseline = np.random.uniform(4, 9)
        
        # Days since last session (0–7 days)
        days_since_last = np.random.randint(0, 8)
        
        # Rating of the previous session (1–10)
        previous_rating = np.random.uniform(3, 9)
        
        # OUTPUT LABELS (simulated based on research)
        # Time-of-day factor for efficiency
        time_factor = [1.2, 1.0, 0.8, 0.5][time_of_day]  # morning performs best
        
        # Rest factor based on recovery
        rest_factor = min(1.0, days_since_last / 3.0)  # capped at 3+ days off
        
        # Calculate base efficiency
        base_efficiency = (concentration_baseline / 10) * time_factor * (0.7 + 0.3 * rest_factor)
        base_efficiency += (previous_rating / 50)  # prior success helps
        base_efficiency = np.clip(base_efficiency, 0.3, 1.0)
        
        # Optimal work-block length (Pomodoro: ~25 min, but variable)
        if concentration_baseline > 7:
            work_block_base = 30  # high concentration -> longer blocks
        elif concentration_baseline > 5:
            work_block_base = 25  # standard Pomodoro
        else:
            work_block_base = 20  # lower concentration -> shorter blocks
        
        # Adjust based on time of day
        work_block_duration = int(work_block_base * time_factor)
        work_block_duration = np.clip(work_block_duration, 15, 45)
        
        # Break length (standard 5 min, longer for lower concentration)
        break_duration = int(5 + (10 - concentration_baseline) * 1.5)
        break_duration = np.clip(break_duration, 5, 15)
        
        # Number of work blocks
        cycle_duration = work_block_duration + break_duration
        optimal_blocks = max(1, int(total_duration / cycle_duration))
        
        # Session concentration score (how well did it go?)
        concentration_score = base_efficiency * 10
        # Overly long sessions reduce the score
        if total_duration > 150:
            concentration_score *= 0.85
        # Too little rest reduces the score
        if days_since_last == 0:
            concentration_score *= 0.9
        
        concentration_score = np.clip(concentration_score, 2, 10)
        
        # Recommendation for next session (in hours)
        if concentration_score > 7:
            next_session_hours = np.random.uniform(4, 8)  # good score: soon again
        elif concentration_score > 5:
            next_session_hours = np.random.uniform(6, 12)
        else:
            next_session_hours = np.random.uniform(12, 24)  # weaker score: more rest
        
        # Noise factor for realism
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

# Generate data
print("🔄 Generating synthetic training data...")
df = generate_learning_sessions(n_samples=500)

# Save
df.to_csv('learning_sessions_data.csv', index=False)
print(f"✅ Created and saved {len(df)} training samples!")
print("\n📊 First 5 rows:")
print(df.head())
print("\n📈 Statistics:")
print(df.describe())
