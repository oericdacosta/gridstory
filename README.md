# gridstory

> F1 race data pipeline — from raw telemetry to structured JSON events ready for LLM-powered narrative generation.

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-violet)
![Status](https://img.shields.io/badge/status-active%20development-orange)

gridstory is a two-module system for Formula 1 race analysis:

- **Module 1 (complete):** Data engineering + ML pipeline that processes a race and outputs three validated JSON files
- **Module 2 (planned):** LLM layer that turns those JSONs into a journalistic report and an interactive chatbot

```
FastF1 API → Extract → Preprocess → ML → Pydantic → timeline.json
                                                    → race_summary.json
                                                    → driver_profiles.json
                                                          ↓
                                               DSPY → race_report.md
                                               Agno → chat endpoint
                                             FastAPI → REST API
```

---

## Features

**Module 1 — implemented**

- Extracts all race data via FastF1: laps, telemetry, race control, weather, results
- Detects tire degradation cliffs with Ruptures/PELT change point detection
- Classifies lap clusters (push / base / degraded) with K-Means
- Identifies anomalous laps with Isolation Forest, cross-referenced with Safety Car events
- Detects undercut maneuvers from pit timing and position data
- Outputs three Pydantic-validated JSONs — the only interface between ML and LLM:
  - `timeline.json` — chronological race events (safety car, driver error, tire dropoff, undercut, penalty)
  - `race_summary.json` — winner, podium, DNFs, safety car count, weather
  - `driver_profiles.json` — per-driver: compounds, push/base/degraded %, tire cliffs, anomalies
- MLflow tracking for all ML experiments (config-driven)

**Module 2 — planned**

- DSPY journalistic report generation from the three JSONs
- Agno knowledge base chatbot ("Who executed the best undercut?")
- FastAPI endpoints: `GET /race/{year}/{round}/report` and `POST /race/{year}/{round}/chat`

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/seu-usuario/gridstory.git
cd gridstory
uv sync

# Run the full pipeline for 2025 Round 1 (Australian GP)
uv run python cli/pipeline.py 2025 1
```

Output:

```
data/
├── raw/races/2025/round_01/          # Phase 1: raw Parquet files
├── processed/races/2025/round_01/    # Phase 2: engineered features
├── ml/races/2025/round_01/           # Phase 3: ML outputs
└── timelines/races/2025/round_01/    # Phase 4: Pydantic-validated JSONs
    ├── timeline.json
    ├── race_summary.json
    └── driver_profiles.json
```

---

## Installation

**Requirements:** Python 3.12+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/seu-usuario/gridstory.git
cd gridstory
uv sync
```

---

## Usage

### Pipeline (single command)

```bash
# Full pipeline — extract, preprocess, ML, generate JSONs
uv run python cli/pipeline.py 2025 1

# Use polling for very recent races (data may not be available yet)
uv run python cli/pipeline.py 2025 1 --polling

# Print data samples at each phase
uv run python cli/pipeline.py 2025 1 --show-sample
```

### MLflow UI

```bash
# View experiment tracking (clustering metrics, anomaly rates, PELT penalty runs)
uv run mlflow ui
# http://localhost:5000
```

### PELT penalty calibration

```bash
# Run penalty search and compare in MLflow UI
uv run python cli/ruptures_analysis.py --year 2025 --round 1 --penalty-search --mlflow

# After choosing the best value, set ml.degradation.penalty in config.yaml
```

---

## Output JSONs

The three files in `data/timelines/races/YEAR/round_XX/` are the single source of truth for Module 2.

### `timeline.json`

Chronological list of significant race events. Every field is narrative-ready — no internal ML metrics.

```json
[
  { "lap": 1,  "type": "safety_car",       "deployed_on_lap": 1, "duration_laps": 6 },
  { "lap": 37, "type": "driver_error",     "driver": "LEC" },
  { "lap": 38, "type": "tire_dropoff",     "driver": "ALB", "lap_time_drop_seconds": 1.39, "cliff_validated": true },
  { "lap": 43, "type": "penalty",          "driver": "BOR", "reason": "Unsafe Release" },
  { "lap": 47, "type": "undercut",         "winner": "NOR", "loser": "VER", "time_gained_seconds": 0.803 }
]
```

Event types: `safety_car` · `driver_error` · `external_incident` · `tire_dropoff` · `undercut` · `penalty`

### `race_summary.json`

```json
{
  "year": 2025, "round": 1, "total_laps": 57,
  "winner": "NOR", "fastest_lap_driver": "NOR", "fastest_lap_time": 82.167,
  "podium": [{ "position": 1, "driver": "NOR", "team": "McLaren", "gap_to_leader": "0.000" }],
  "dnfs": [{ "driver": "LAW", "on_lap": 47 }],
  "safety_car_count": 3,
  "weather": { "condition": "mixed", "air_temp_avg_c": 15.7, "had_rainfall": true }
}
```

### `driver_profiles.json`

```json
[
  {
    "driver": "NOR", "team": "McLaren",
    "final_position": 1, "grid_position": 1, "positions_gained": 0, "points": 25.0,
    "push_pct": 0.067, "base_pct": 0.844, "degraded_pct": 0.089,
    "compounds_used": [{ "compound": "INTERMEDIATE", "laps": 47 }, { "compound": "HARD", "laps": 10 }],
    "had_tire_cliff": true, "cliff_count": 1, "anomaly_count": 3
  }
]
```

---

## Architecture

```
gridstory/
├── cli/
│   ├── pipeline.py                  # Single entry point
│   ├── pipeline_steps/
│   │   ├── extraction.py            # Phase 1
│   │   ├── preprocessing.py         # Phase 2
│   │   ├── ml.py                    # Phase 3
│   │   ├── events.py                # Phase 4 — generates the 3 JSONs
│   │   └── reporting.py
│   ├── ruptures_analysis.py         # PELT penalty calibration tool
│   └── list_data.py
│
├── src/
│   ├── extraction/                  # FastF1 ETL
│   ├── preprocessing/               # SciPy signal processing + feature engineering
│   ├── ml/
│   │   ├── pipeline.py              # run_race_analysis()
│   │   ├── clustering.py            # K-Means, DBSCAN
│   │   ├── anomaly_detection.py     # Isolation Forest
│   │   ├── anomaly_classification.py# Z-score + race control cross-reference
│   │   ├── change_point.py          # Ruptures/PELT tire cliffs
│   │   ├── strategy.py              # detect_undercuts()
│   │   ├── timeline.py              # build_race_timeline()
│   │   ├── race_summary_builder.py  # build_race_summary()
│   │   ├── driver_profiles_builder.py # build_driver_profiles()
│   │   ├── metrics.py
│   │   └── tracking.py              # MLflow integration
│   ├── models/                      # Pydantic contracts (ML ↔ LLM firewall)
│   │   ├── race_events.py           # RaceTimeline + all event types
│   │   ├── race_summary.py          # RaceSummary, WeatherSummary, PodiumEntry
│   │   └── driver_profile.py        # DriverProfile, CompoundUsage
│   ├── llm/                         # [planned] DSPY + Agno
│   ├── api/                         # [planned] FastAPI
│   └── utils/
│
├── data/                            # gitignored
│   ├── raw/races/
│   ├── processed/races/
│   ├── ml/races/
│   └── timelines/races/
│
├── config.yaml                      # All pipeline parameters
└── pyproject.toml
```

---

## Tech Stack

| Layer | Tools | Status |
|---|---|---|
| Data extraction | FastF1, Pandas, NumPy | ✅ Done |
| Storage | Parquet (PyArrow) | ✅ Done |
| Preprocessing | SciPy (interpolate, signal, stats) | ✅ Done |
| ML — clustering | Scikit-learn K-Means, DBSCAN | ✅ Done |
| ML — anomaly detection | Scikit-learn Isolation Forest | ✅ Done |
| ML — change point | Ruptures/PELT | ✅ Done |
| ML tracking | MLflow | ✅ Done |
| Data contracts | Pydantic v2 | ✅ Done |
| Report generation | DSPY | 📅 Planned |
| Chatbot | Agno | 📅 Planned |
| API | FastAPI | 📅 Planned |

---

## Configuration

All pipeline parameters live in `config.yaml`:

```yaml
mlflow:
  enabled: true          # set to false to skip tracking

ml:
  degradation:
    penalty: 3           # PELT sensitivity — calibrate with ruptures_analysis.py
    min_cliff_magnitude: 0.3
```

See [docs/configuration.md](docs/configuration.md) for all options.

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| 1 — Extraction | FastF1 → Parquet | ✅ Complete |
| 2 — Preprocessing | SciPy feature engineering | ✅ Complete |
| 3 — Machine Learning | Clustering, anomaly detection, tire cliffs | ✅ Complete |
| 4 — Pydantic contracts | Validate ML outputs → 3 JSONs | ✅ Complete |
| 5 — LLM report | DSPY journalistic report | 📅 Next |
| 6 — LLM chatbot | Agno knowledge base Q&A | 📅 Next |
| 7 — API | FastAPI endpoints | 📅 Next |

---

## Documentation

- [USAGE.md](USAGE.md) — detailed usage guide
- [ARCHITECTURE.md](ARCHITECTURE.md) — architecture and data flow
- [CHANGELOG.md](CHANGELOG.md) — version history
- [cli/README.md](cli/README.md) — CLI reference
- [src/ml/README.md](src/ml/README.md) — ML module reference
- [docs/configuration.md](docs/configuration.md) — config.yaml reference

---

## Contributing

Issues and pull requests are welcome.
