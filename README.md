### Apppy
## Make learning great again






## How to Run the App

Open a terminal in the project directory and execute:

1. `pip install -r requirements.txt`
    (will download: streamlit, pandas, numpy, plotly, scikit-learn, pdfplumber, uvicorn, matplotlib, fastapi, pydantic, requests)
2. `streamlit run app.py`

## Overview

An intelligent study-plan generator that uses **Ridge Regression** and **clustering** to create personalized plans based on focus level, time of day, and individual learning behaviour.

## 🎯 Features

- ✅ **Machine Learning**: Ridge Regression predicts optimal study/break lengths.
- ✅ **Personalized recommendations**: Driven by time of day, concentration, and historical data.
- ✅ **Interactive visualization**: Heatmap + dashboards.
- ✅ **Feedback loop**: User feedback is stored for future retraining.
- ✅ **Streamlit web app**: Simple interface for experiments and demos.

### 📊 How does it work?

**Machine-learning component**

The app uses **four separate Ridge Regression models**:

1. **Work blocks** – predicts the optimal number of study blocks.
2. **Block duration** – predicts the length of each study block.
3. **Break duration** – predicts the break length between blocks.
4. **Next session** – recommends when to study next.

**Input features**

- Total session duration (30–240 minutes)
- Time of day (morning/afternoon/evening/night)
- Concentration level (1–10)
- Days since the previous session
- Rating of the previous session

**Outputs**

- Optimized schedule with study and break blocks
- Personalized tips
- Next-session recommendation

## 🧠 Scientific Background

The models draw inspiration from:
- **Pomodoro technique**: 25 min work + 5 min break
- **Chronobiology**: performance varies over the day
- **Spacing effect**: optimal intervals between learning sessions

## 📁 Project Structure

```
CS-Projekt/
├── app.py                          # Streamlit web app
├── train_model.py                  # ML training script
├── generate_training_data.py       # Synthetic data generator
├── requirements.txt                # Python dependencies
├── learning_models.pkl             # Trained models (generated)
└── learning_sessions_data.csv      # Training data (generated)
```

## 📝 Requirements Met

- ✅ Clearly defined problem (study-plan optimisation)
- ✅ Data loaded via API or database (synthetic data, extensible)
- ✅ Data visualisation (Gantt/timeline charts, tables)
- ✅ User interaction (forms, feedback flow)
- ✅ Machine learning (Ridge Regression)
- ✅ Well-documented code
- ✅ Contribution matrix tracked
