# ML Pipeline Documentation

🚧 **Under Development** - ML pipeline not yet implemented

## Overview

The ML pipeline transforms raw race data into structured events using machine learning algorithms.

## Pipeline Stages

### 1. Feature Engineering

Extract relevant features from raw data:
- Lap time deltas
- Tire age
- Gap to car ahead/behind
- Track position
- Weather conditions

### 2. Event Detection

#### Tire Degradation (Ruptures)

Use Change Point Detection (CUSUM) to find exact moment tire performance drops.

**Input**: Lap times per stint
**Output**: Degradation points with confidence scores

#### Anomalies (Isolation Forest)

Detect driver errors and unusual laps.

**Input**: Telemetry features (speed, throttle, brake)
**Output**: Anomaly scores per lap

#### Stint Clustering (K-Means/DBSCAN)

Group stints by pace characteristics.

**Input**: Average lap time, tire compound, fuel load
**Output**: Stint clusters (fast/medium/slow)

### 3. Event Synthesis

Combine ML outputs into structured events:
- Undercuts
- Overcuts
- Tire drops
- Safety car impact
- Weather changes

### 4. Validation (Pydantic)

Ensure output conforms to schema.

## Observability (MLflow)

Track:
- Model parameters
- Detection thresholds
- Performance metrics
- Experiment runs

## Output Format

```json
{
  "race": {
    "year": 2025,
    "round": 1
  },
  "events": [
    {
      "type": "tire_degradation",
      "driver": "VER",
      "lap": 18,
      "confidence": 0.95,
      "delta_time": -0.8
    }
  ]
}
```
