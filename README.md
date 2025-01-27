# VisCos Projekt 🚀

VisCos ist ein Projekt zur Visualisierung und Analyse von Daten mit Hilfe von Clustering- und Bildanalyse-Methoden. Es ist modular aufgebaut und nutzt Python als Programmiersprache.

## Voraussetzungen ✅

Um das Projekt auszuführen, wird folgende Software benötigt:

1. **Python**: Version 3.10 oder höher 🐍
2. **Poetry**: Ein Paketmanager für Python 📦
3. **Git**: Um das Repository zu klonen, falls noch nicht lokal verfügbar 🌐

Alle Abhängigkeiten sind in der pyproject.toml aufgelistet und werden automatisch mit Poetry installiert. Einige wichtige Bibliotheken sind:
- plotly für Visualisierungen 📈
- scikit-learn für Clustering 🔍
- Weitere hilfreiche Python-Pakete, die das Projekt unterstützt 📜

## Installation ⚙️

1. Repository klonen:

```
git clone https://github.com/nlsrmll/VisCos.git
cd VisCos
```
2. Poetry installieren (falls nicht vorhanden):
```
pip install poetry
```
3. Abhängigkeiten installieren:
```
poetry install
```
## Projektstruktur 🗃️
```
VisCos
├── main.py                   # Haupteinstiegspunkt des Projekts
├── packages                  # Enthält die Python-Module
│   ├── clustering            # Module für Clustering-Methoden
│   ├── image_analyse         # Module für Bildanalyse
│   └── visualization         # Module für Datenvisualisierung
├── data                      # Platz für Daten
├── pyproject.toml            # Poetry-Konfigurationsdatei
└── poetry.lock               # Abhängigkeits-Sperrdatei
```
## Ausführung ▶️

1. **Virtuelle Umgebung aktivieren**:
Poetry erstellt eine virtuelle Umgebung automatisch. Aktivieren Sie sie mit:
```
poetry shell
```
2. **Projekt starten**:
```
python main.py
```
## Hinweise ℹ️

Der Ordner data ist für Eingabedaten vorgesehen. Stellen Sie sicher, dass alle benötigten Dateien dort abgelegt werden.

## Debugging und Tests 🛠️

Linting: Nutzen Sie ruff (bereitgestellt durch Poetry), um den Code zu überprüfen:
```
poetry run ruff check .
```
## Kontakt 📬

Bei Fragen oder Problemen wenden Sie sich bitte an den Projektverantwortlichen.
