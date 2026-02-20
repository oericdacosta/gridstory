# CLI Reference — gridstory

## Commands

### `pipeline.py` — main entry point

Runs the complete pipeline: extraction → preprocessing → ML → JSON generation.

```bash
uv run python cli/pipeline.py YEAR ROUND [--polling] [--show-sample]
```

**Options:**
- `YEAR` — season year (e.g. `2025`)
- `ROUND` — race number (e.g. `1`)
- `--polling` — wait for data availability (use for races within the last 24h)
- `--show-sample` — print data samples at each phase

**Phases executed:**
1. **Extraction** (`pipeline_steps/extraction.py`) — FastF1 → `data/raw/`
2. **Preprocessing** (`pipeline_steps/preprocessing.py`) — SciPy → `data/processed/`
3. **ML** (`pipeline_steps/ml.py`) — Clustering + anomaly detection + PELT → `data/ml/`
4. **Events** (`pipeline_steps/events.py`) — Pydantic validation → `data/timelines/`

**Output:**
```
data/timelines/races/YEAR/round_XX/
├── timeline.json         # chronological events (semantic format)
├── race_summary.json     # winner, podium, DNFs, weather, safety cars
└── driver_profiles.json  # per-driver stats
```

---

### `ruptures_analysis.py` — PELT penalty calibration

One-time calibration tool for the Ruptures/PELT change point detection algorithm.

```bash
# Test a range of penalties and log to MLflow
uv run python cli/ruptures_analysis.py --year 2025 --round 1 --penalty-search --mlflow

# Compare past runs by cliff_rate
uv run python cli/ruptures_analysis.py --compare --experiment "F1_2025_Round_01_Ruptures"

# Inspect one driver with the current config penalty
uv run python cli/ruptures_analysis.py --year 2025 --round 1 --driver VER --show-metrics
```

**Calibration workflow:**
1. Run `--penalty-search` → MLflow logs one run per penalty tested
2. Open `uv run mlflow ui` → compare `cliff_rate` and `cliff_validated_rate`
3. Choose the best value → set `ml.degradation.penalty` in `config.yaml`
4. Done — `pipeline.py` uses that value automatically

**Options:**
- `--year / --round` — identify the race
- `--driver CODE` — analyze a single driver only
- `--penalty-search` — test all penalties in `penalty_search_range` (config.yaml)
- `--mlflow` — log each penalty as a separate MLflow run
- `--experiment NAME` — MLflow experiment name
- `--compare` — compare past runs by cliff_rate
- `--save` — save `laps_changepoints.parquet` and `tire_cliffs.parquet`
- `--show-metrics` — print detailed metrics

---

### `list_data.py` — data inventory

```bash
uv run python cli/list_data.py
```

Lists all extracted and processed races with sizes and driver counts.

---

## Recommended workflow

```bash
# 1. Run the full pipeline
uv run python cli/pipeline.py 2025 1

# 2. (Optional) View MLflow experiments
uv run mlflow ui    # http://localhost:5000

# 3. (One-time) Calibrate PELT penalty
uv run python cli/ruptures_analysis.py --year 2025 --round 1 --penalty-search --mlflow
# → pick best value in MLflow UI → set ml.degradation.penalty in config.yaml

# 4. Check generated data
uv run python cli/list_data.py
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

---

## Troubleshooting

```bash
# Data not yet available
uv run python cli/pipeline.py 2025 10 --polling

# Corrupted FastF1 cache
rm -rf ~/.cache/fastf1/

# ruptures_analysis needs ML data — run pipeline first
uv run python cli/pipeline.py 2025 1
```

---

## See also

- [README.md](../README.md) — project overview
- [USAGE.md](../USAGE.md) — detailed usage guide
- [docs/configuration.md](../docs/configuration.md) — config.yaml reference
- [MLFLOW_SETUP.md](../MLFLOW_SETUP.md) — MLflow setup and metrics
