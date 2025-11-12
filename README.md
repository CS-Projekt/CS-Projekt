# 📚 AI-gestützter Lernplan Generator

Ein intelligenter Lernplan-Generator, der mit **Ridge Regression** personalisierte Lernpläne erstellt basierend auf Konzentrationsfähigkeit, Tageszeit und individuellem Lernverhalten.

## 🎯 Features

- ✅ **Machine Learning**: Ridge Regression zur Vorhersage optimaler Lernzeiten
- ✅ **Personalisierte Empfehlungen**: Basierend auf Tageszeit, Konzentration und historischen Daten
- ✅ **Interaktive Visualisierung**: Gantt-Charts und Zeitpläne
- ✅ **Feedback-System**: User-Feedback wird gespeichert für zukünftiges Re-Training
- ✅ **Streamlit Web-App**: Einfach zu bedienende Benutzeroberfläche

## 🚀 Installation & Setup

### 1. Repository klonen

```bash
git clone https://github.com/CS-Projekt/CS-Projekt.git
cd CS-Projekt
```

### 2. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 3. Trainingsdaten generieren

```bash
python generate_training_data.py
```

Dies erstellt eine CSV-Datei mit 500 synthetischen Lernsessions.

### 4. ML-Modell trainieren

```bash
python train_model.py
```

Dies trainiert 4 Ridge Regression Modelle und speichert sie in `learning_models.pkl`.

### 5. App starten

```bash
streamlit run app.py
```

Die App öffnet sich automatisch im Browser unter `http://localhost:8501`

## 📊 Wie funktioniert's?

### Machine Learning Komponente

Die App nutzt **4 separate Ridge Regression Modelle**:

1. **Arbeitsblöcke**: Vorhersage der optimalen Anzahl von Lernblöcken
2. **Block-Dauer**: Vorhersage der idealen Länge pro Lernblock
3. **Pausen-Dauer**: Vorhersage der notwendigen Pausenlänge
4. **Nächste Session**: Empfehlung für den Zeitpunkt der nächsten Lernsession

### Input-Features

- Gesamte Session-Dauer (30-240 Minuten)
- Tageszeit (Morgen/Nachmittag/Abend/Nacht)
- Konzentrationslevel (1-10)
- Tage seit letzter Session
- Rating der vorherigen Session

### Output

- Optimierter Zeitplan mit Lern- und Pausenblöcken
- Personalisierte Tipps
- Empfehlung für die nächste Session

## 🧠 Wissenschaftlicher Hintergrund

Die Modelle basieren auf:
- **Pomodoro-Technik**: 25 Minuten Arbeit + 5 Minuten Pause
- **Chronobiologie**: Tageszeit-abhängige Konzentrationsfähigkeit
- **Spacing Effect**: Optimale Abstände zwischen Lernsessions

## 📁 Projektstruktur

```
CS-Projekt/
├── app.py                          # Streamlit Web-App
├── train_model.py                  # ML-Modell Training
├── generate_training_data.py       # Synthetische Daten
├── requirements.txt                # Python Dependencies
├── learning_models.pkl             # Trainierte Modelle (wird erstellt)
└── learning_sessions_data.csv      # Trainingsdaten (wird erstellt)
```

## 👥 Team

Gruppenprojekt für **Fundamentals and Methods of Computer Science** - Universität St. Gallen

## 📝 Anforderungen erfüllt

- ✅ Problem klar definiert (Lernplan-Optimierung)
- ✅ Daten via API geladen (synthetische Daten, erweiterbar)
- ✅ Datenvisualisierung (Gantt-Charts, Tabellen)
- ✅ User-Interaktion (Input-Formulare, Feedback)
- ✅ Machine Learning (Ridge Regression)
- ✅ Gut dokumentierter Code
- ✅ Contribution Matrix vorhanden

## 🔮 Zukünftige Erweiterungen

- Integration mit echten Lern-Daten (Spaced Repetition Systems)
- Re-Training mit User-Feedback
- Export von Lernplänen (iCal, PDF)
- Mobile App Version
- Multi-User Support mit Datenbank

## 📄 Lizenz

Dieses Projekt wurde für akademische Zwecke erstellt.