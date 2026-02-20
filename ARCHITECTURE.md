# Architecture — gridstory

## Data Flow

```
┌────────────────┐
│  FastF1 API    │
└───────┬────────┘
        │
        ▼
┌────────────────────────┐
│  Phase 1: Extraction   │  cli/pipeline_steps/extraction.py
│  FastF1 + Pandas       │  → data/raw/races/YEAR/round_XX/
└───────┬────────────────┘    laps.parquet
        │                     telemetry/*.parquet
        │                     race_control.parquet
        │                     weather.parquet
        │                     results.parquet
        ▼
┌────────────────────────┐
│  Phase 2: Preprocess   │  cli/pipeline_steps/preprocessing.py
│  SciPy + Scikit-learn  │  → data/processed/races/YEAR/round_XX/
└───────┬────────────────┘    laps_processed.parquet
        │                     race_control_processed.parquet
        │                     weather_processed.parquet
        │                     results_processed.parquet
        ▼
┌────────────────────────┐
│  Phase 3: ML           │  cli/pipeline_steps/ml.py
│  Scikit-learn +        │  → data/ml/races/YEAR/round_XX/
│  Ruptures + MLflow     │    laps_clustered.parquet
└───────┬────────────────┘    laps_anomalies.parquet
        │                     tire_cliffs.parquet
        │                     anomalies_summary.parquet
        │                     tire_cliffs_summary.parquet
        ▼
┌────────────────────────┐
│  Phase 4: Pydantic     │  cli/pipeline_steps/events.py
│  (ML ↔ LLM firewall)  │  → data/timelines/races/YEAR/round_XX/
└───────┬────────────────┘    timeline.json        ← chronological events
        │                     race_summary.json    ← race context
        │                     driver_profiles.json ← per-driver stats
        │
        │  [Module 2 — planned]
        ▼
┌────────────────────────┐
│  Phase 5: LLM Report   │  src/llm/reporter.py  (DSPY)
│  DSPY                  │  → data/timelines/races/YEAR/round_XX/
└───────┬────────────────┘    race_report.md
        │
        ▼
┌────────────────────────┐
│  Phase 6: API          │  src/api/main.py  (FastAPI)
│  FastAPI + Agno        │  GET /race/{year}/{round}/report
└────────────────────────┘  POST /race/{year}/{round}/chat
```

---

## Project Structure

```
gridstory/
│
├── cli/                              # Command-line entry points
│   ├── pipeline.py                   # ✅ Main pipeline (phases 1–4)
│   ├── pipeline_steps/
│   │   ├── extraction.py             # ✅ Phase 1: FastF1 extraction
│   │   ├── preprocessing.py          # ✅ Phase 2: SciPy preprocessing
│   │   ├── ml.py                     # ✅ Phase 3: ML pipeline
│   │   ├── events.py                 # ✅ Phase 4: Pydantic JSON generation
│   │   └── reporting.py              # ✅ Reporter helper class
│   ├── ruptures_analysis.py          # ✅ PELT penalty calibration tool
│   └── list_data.py                  # ✅ List available data
│
├── src/
│   ├── extraction/                   # ✅ FastF1 ETL
│   │   ├── calendar.py
│   │   ├── polling.py
│   │   ├── etl.py
│   │   └── orchestrator.py
│   │
│   ├── preprocessing/                # ✅ SciPy signal processing + features
│   │   ├── interpolation.py
│   │   ├── signal_processing.py
│   │   └── feature_engineering.py
│   │
│   ├── ml/                           # ✅ Unsupervised ML pipeline
│   │   ├── pipeline.py               # run_race_analysis()
│   │   ├── clustering.py             # K-Means, DBSCAN
│   │   ├── anomaly_detection.py      # Isolation Forest
│   │   ├── anomaly_classification.py # Z-score + race control cross-ref
│   │   ├── change_point.py           # Ruptures/PELT — tire cliffs
│   │   ├── strategy.py               # detect_undercuts()
│   │   ├── timeline.py               # build_race_timeline()
│   │   ├── race_summary_builder.py   # build_race_summary()
│   │   ├── driver_profiles_builder.py# build_driver_profiles()
│   │   ├── metrics.py
│   │   └── tracking.py               # MLflow integration
│   │
│   ├── models/                       # ✅ Pydantic data contracts
│   │   ├── race_events.py            # RaceTimeline + 6 event types
│   │   ├── race_summary.py           # RaceSummary, WeatherSummary, PodiumEntry, DnfEntry
│   │   └── driver_profile.py         # DriverProfile, CompoundUsage
│   │
│   ├── llm/                          # 📅 Planned — DSPY + Agno
│   ├── api/                          # 📅 Planned — FastAPI
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── data/                             # gitignored
│   ├── raw/races/
│   ├── processed/races/
│   ├── ml/races/
│   └── timelines/races/
│
├── docs/
├── config.yaml
└── pyproject.toml
```

---

## Module Descriptions

### `src/extraction/`
FastF1 ETL — connects to the F1 API and saves all race data as Parquet files.
Extracts: laps, telemetry (all drivers), race control messages, weather, results.

### `src/preprocessing/`
SciPy-based signal processing and feature engineering for all five data types.
Key operations: telemetry interpolation to common grid, noise removal, Z-score features, degradation slope per driver/compound.

### `src/ml/`
Unsupervised ML pipeline. Three algorithms on lap data:
- **K-Means** — classifies every lap as `push`, `base`, or `degraded`
- **Isolation Forest** — flags statistically anomalous laps
- **Ruptures/PELT** — detects tire cliff change points per stint

Additional modules:
- `anomaly_classification.py` — determines if anomaly is `driver_error` or `external_incident` (SC/flags)
- `strategy.py` — detects undercut maneuvers from pit timing and position data
- `timeline.py` / `race_summary_builder.py` / `driver_profiles_builder.py` — build the Pydantic objects that feed the three JSONs

### `src/models/` — Pydantic firewall
The only interface between ML and LLM. All ML outputs must pass through these models before being serialized. No raw ML metrics (`anomaly_score`, `z_score`) are exposed to downstream consumers.

Event types in `RaceTimeline`:
| Type | Description |
|---|---|
| `driver_error` | Anomalous lap with no external cause |
| `external_incident` | Anomalous lap during SC / yellow flag |
| `tire_dropoff` | Tire cliff detected by PELT |
| `undercut` | Undercut maneuver — winner / loser |
| `safety_car` | SC deployed — duration in laps |
| `penalty` | FIA penalty — driver + reason |

### `src/llm/` (planned)
- `reporter.py` — DSPY `RaceReportSignature` + `RaceReporter` module
- `agent.py` — Agno `JSONKnowledgeBase` + `Agent`

### `src/api/` (planned)
- `main.py` — FastAPI application with report and chat endpoints

---

## Design Principles

1. **LLM receives only semantic data** — Pydantic models remove all internal ML metrics before serialization. The LLM sees `winner`/`loser`, not `anomaly_score`/`z_score`.
2. **Everything deterministic before the LLM** — race summary, driver profiles, event classification are all calculated by code, not inferred by AI.
3. **Config-driven** — all ML hyperparameters (contamination, penalty, k) and MLflow settings live in `config.yaml`.
4. **Fail fast** — `RaceTimeline.model_validate()` raises `ValidationError` immediately if any ML output is malformed.

---

## Development Status

| Phase | Status |
|---|---|
| Phase 1: Extraction | ✅ Complete |
| Phase 2: Preprocessing | ✅ Complete |
| Phase 3: Machine Learning | ✅ Complete |
| Phase 4: Pydantic contracts | ✅ Complete |
| Phase 5: LLM report (DSPY) | 📅 Planned |
| Phase 6: Chatbot (Agno) | 📅 Planned |
| Phase 7: API (FastAPI) | 📅 Planned |
