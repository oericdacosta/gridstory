# gridstory — Usage Guide

## Quick Start

```bash
uv run python cli/pipeline.py 2025 1
```

This single command runs the full pipeline — extraction, preprocessing, ML, and JSON generation — for 2025 Round 1 (Australian GP).

---

## CLI Reference

### `cli/pipeline.py` — main pipeline

```bash
# Full pipeline
uv run python cli/pipeline.py YEAR ROUND

# Wait for data to become available (use for very recent races)
uv run python cli/pipeline.py 2025 1 --polling

# Print data samples at each phase
uv run python cli/pipeline.py 2025 1 --show-sample
```

What each phase does:

| Phase | Description | Output |
|---|---|---|
| 1 — Extraction | Downloads all race data via FastF1 | `data/raw/races/YEAR/round_XX/` |
| 2 — Preprocessing | Engineers features with SciPy | `data/processed/races/YEAR/round_XX/` |
| 3 — ML | Clustering, anomaly detection, tire cliffs | `data/ml/races/YEAR/round_XX/` |
| 4 — Events | Builds and validates the three JSONs with Pydantic | `data/timelines/races/YEAR/round_XX/` |

### `cli/ruptures_analysis.py` — PELT penalty calibration

Use this once to find the right `penalty` value for the PELT algorithm, then set it in `config.yaml`.

```bash
# Test a range of penalties (range defined in config.yaml > ml.degradation.penalty_search_range)
uv run python cli/ruptures_analysis.py --year 2025 --round 1 --penalty-search --mlflow

# Compare past runs in MLflow
uv run python cli/ruptures_analysis.py --compare --experiment "F1_2025_Round_01_Ruptures"

# Inspect one driver with the current config penalty
uv run python cli/ruptures_analysis.py --year 2025 --round 1 --driver VER --show-metrics
```

Calibration workflow:
1. Run `--penalty-search` → MLflow logs one run per penalty tested
2. Open `uv run mlflow ui` → compare `cliff_rate` and `cliff_validated_rate`
3. Pick the best value → set `ml.degradation.penalty` in `config.yaml`
4. Run `pipeline.py` normally from then on

### `cli/list_data.py` — data inventory

```bash
uv run python cli/list_data.py
```

Lists all extracted and processed races with sizes and driver counts.

---

## Output Structure

### Phase 1 — Raw data (`data/raw/`)

```
data/raw/races/2025/round_01/
├── metadata.json
├── laps.parquet
├── race_control.parquet
├── weather.parquet
├── results.parquet
└── telemetry/
    ├── VER.parquet
    ├── NOR.parquet
    └── ...           # one file per driver
```

### Phase 2 — Processed data (`data/processed/`)

```
data/processed/races/2025/round_01/
├── laps_processed.parquet
├── race_control_processed.parquet
├── weather_processed.parquet
├── results_processed.parquet
└── telemetry/
    ├── VER_processed.parquet
    └── ...
```

Key features added to laps:
- `z_score`, `is_outlier` — statistical anomaly flags
- `degradation_slope`, `degradation_r_squared` — per-stint tire degradation rate

Key features added to race_control:
- `is_safety_car`, `is_flag`, `is_penalty` — binary event flags
- `event_severity` — 0=info, 1=warning, 2=critical

### Phase 3 — ML outputs (`data/ml/`)

```
data/ml/races/2025/round_01/
├── laps_clustered.parquet       # cluster_semantic: push / base / degraded
├── laps_anomalies.parquet       # is_anomaly, anomaly_score per lap
├── anomalies_summary.parquet    # anomalies_count, anomaly_rate per driver
├── laps_changepoints.parquet    # stint_regime, is_cliff_lap
├── tire_cliffs.parquet          # has_cliff, cliff_lap, cliff_delta_magnitude per (Driver, Stint)
└── tire_cliffs_summary.parquet  # stints_with_cliff, cliffs_validated per driver
```

### Phase 4 — Timeline JSONs (`data/timelines/`)

```
data/timelines/races/2025/round_01/
├── timeline.json         # chronological events (semantic format)
├── race_summary.json     # winner, podium, DNFs, weather, safety car count
└── driver_profiles.json  # per-driver: pace clusters, compounds, cliffs, anomalies
```

These are the only files consumed by Module 2 (LLM). No raw ML metrics or DataFrames reach the LLM.

---

## Programmatic Usage

### Run the full pipeline

```python
from cli.pipeline import run_complete_pipeline

run_complete_pipeline(year=2025, round_num=1, show_sample=True)
```

### Access processed data

```python
import pandas as pd

laps = pd.read_parquet('data/processed/races/2025/round_01/laps_processed.parquet')
rc   = pd.read_parquet('data/processed/races/2025/round_01/race_control_processed.parquet')

# Safety car periods
sc = rc[rc['is_safety_car'] & rc['Message'].str.contains('DEPLOYED')]

# Tire degradation per driver
deg = laps.groupby('Driver')['degradation_slope'].mean().sort_values()
```

### Access ML outputs

```python
import pandas as pd

clustered = pd.read_parquet('data/ml/races/2025/round_01/laps_clustered.parquet')

# Percentage of laps in each cluster per driver
pct = (
    clustered.groupby('Driver')['cluster_semantic']
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .round(3)
)
```

### Read the output JSONs

```python
import json

timeline = json.load(open('data/timelines/races/2025/round_01/timeline.json'))
summary  = json.load(open('data/timelines/races/2025/round_01/race_summary.json'))
profiles = json.load(open('data/timelines/races/2025/round_01/driver_profiles.json'))

# All undercuts
undercuts = [e for e in timeline if e['type'] == 'undercut']

# Drivers with tire cliffs
cliffs = [p for p in profiles if p['had_tire_cliff']]
```

---

## MLflow

Tracking is config-driven. Enable or disable in `config.yaml`:

```yaml
mlflow:
  enabled: true
```

```bash
uv run mlflow ui
# http://localhost:5000
```

---

## Performance

| Operation | Time (with cache) | Disk |
|---|---|---|
| Extraction | ~30–60s | ~11–15 MB |
| Preprocessing | ~10–20s | ~8–12 MB |
| Machine Learning | ~5–15s | ~2–5 MB |
| JSON generation | <1s | <100 KB |
| **Full pipeline** | **~45–95s** | **~25–35 MB** |

First run is slower (FastF1 downloads and caches data).

---

## Troubleshooting

**Data not available yet**
```bash
uv run python cli/pipeline.py 2025 10 --polling
```

**Corrupted FastF1 cache**
```bash
rm -rf ~/.cache/fastf1/
```

**ruptures_analysis needs ML data first**
```bash
uv run python cli/pipeline.py 2025 1   # run pipeline first
uv run python cli/ruptures_analysis.py --year 2025 --round 1 --penalty-search
```
