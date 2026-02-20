# `src/models/` — Pydantic Data Contracts

This module is the **firewall between ML and LLM**. Every ML output must pass through these Pydantic models before being serialized to JSON. If data is invalid, `ValidationError` is raised here — the LLM never receives malformed data.

## Design principle

The models expose only **semantic, narrative-ready fields**. Internal ML metrics (`anomaly_score`, `z_score`, `contamination`) are computed in `src/ml/` but never appear in the final JSON.

## Files

### `race_events.py` — `RaceTimeline`

Discriminated union of six event types, ordered chronologically.

| Type | Fields | Source |
|---|---|---|
| `driver_error` | `lap`, `driver` | Isolation Forest, Z-score < 0, no external cause |
| `external_incident` | `lap`, `driver` | Anomalous lap during SC / yellow flag |
| `tire_dropoff` | `lap`, `driver`, `lap_time_drop_seconds`, `cliff_validated`, `positions_lost` | Ruptures/PELT |
| `undercut` | `lap`, `winner`, `loser`, `time_gained_seconds` | strategy.detect_undercuts() |
| `safety_car` | `lap`, `deployed_on_lap`, `duration_laps` | race_control DEPLOYED → IN THIS LAP |
| `penalty` | `lap`, `driver`, `reason` | race_control is_penalty |

```python
from src.models.race_events import RaceTimeline

timeline = RaceTimeline.model_validate(raw_events_list)
json_str = timeline.to_json()   # serializes without None fields
```

`ConfigDict(extra="forbid", strict=True)` on all models — rejects unknown fields from ML.

### `race_summary.py` — `RaceSummary`

```python
class RaceSummary(BaseModel):
    year: int
    round: int
    total_laps: int
    winner: str                     # "NOR"
    fastest_lap_driver: str
    fastest_lap_time: float
    podium: list[PodiumEntry]       # top 3
    dnfs: list[DnfEntry]
    safety_car_count: int
    weather: WeatherSummary         # condition: "dry" | "wet" | "mixed"
```

Built by `src/ml/race_summary_builder.py` from results + weather + laps + race_control DataFrames.

### `driver_profile.py` — `DriverProfile`

```python
class DriverProfile(BaseModel):
    driver: str                     # "NOR"
    team: str
    final_position: int | None
    grid_position: int | None
    positions_gained: int           # positive = gained, negative = lost
    points: float
    push_pct: float                 # fraction of laps in "push" cluster
    base_pct: float
    degraded_pct: float
    compounds_used: list[CompoundUsage]
    had_tire_cliff: bool
    cliff_count: int
    anomaly_count: int
```

Built by `src/ml/driver_profiles_builder.py` from laps_clustered + anomalies_summary + tire_cliffs_summary + results DataFrames.

## Outputs

All three models serialize to `data/timelines/races/YEAR/round_XX/`:

```
timeline.json         ← RaceTimeline.to_json()
race_summary.json     ← RaceSummary.model_dump_json()
driver_profiles.json  ← [DriverProfile.model_dump() for p in profiles]
```
