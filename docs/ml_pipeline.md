# ML Pipeline - PitWall AI

Pipeline completo de Machine Learning para análise de corridas de Fórmula 1.

## Visão Geral

O pipeline de ML transforma dados pré-processados em eventos e insights estatísticos usando **aprendizado não supervisionado**. É executado como Fase 3 do pipeline principal:

```bash
uv run python cli/pipeline.py 2025 1
```

## Fluxo do Pipeline

```
laps_processed.parquet (data/processed/races/)
    ↓
[1. Engenharia de Features]
    ├─ Stint detection (via TyreLife resets)
    ├─ LapTime_delta (desvio relativo à mediana do stint)
    ├─ TyreAge_normalized (0→1 dentro do stint)
    └─ Compound_ordinal (SOFT=1, MEDIUM=2, HARD=3)
    ↓
[2. Pré-processamento para ML]
    ├─ Imputação (SimpleImputer, estratégia=median)
    ├─ Encoding (OneHotEncoder para Compound)
    └─ Escalonamento (RobustScaler)
    ↓
[3. Clustering — K-Means por piloto (k=3 fixo)]
    ├─ Features: LapTime_delta, TyreAge_normalized, Sector1Time_seconds, degradation_slope
    ├─ Laps estruturais filtrados (pit stops, safety car > 1.5× mediana)
    ├─ Semântica determinística: push=0, base=1, degraded=2
    └─ DBSCAN (análise complementar)
    ↓
[4. Anomaly Detection — Isolation Forest por piloto]
    ├─ Features: LapTime_delta, Sector1/2/3Time_seconds, degradation_slope, Position
    ├─ Roda em TODOS os laps (incluindo pit/SC — queremos detectá-los)
    └─ Contamination config-driven (perfis: clean=3%, normal=5%, chaotic=10%)
    ↓
[5. Change Point Detection — PELT por stint]
    ├─ Input: laps_anomalies (com LapTime_delta, is_anomaly, Stint)
    ├─ Detecta tire cliffs (mudança de regime de degradação)
    └─ Validação: slope positivo de degradação antes do cliff
    ↓
[6. MLFlow Tracking (se mlflow.enabled=true no config.yaml)]
    ├─ Parâmetros: contamination, kmeans_k, scaler_type, features usadas
    ├─ Métricas: silhouette/DB por piloto, n_anomalies, cliff_rate
    └─ Artefatos CSV: laps_clustered, laps_anomalies, tire_cliffs, per_driver_metrics
    ↓
Outputs (data/ml/races/YEAR/round_XX/)
```

## Outputs Gerados

| Arquivo | Conteúdo |
|---------|----------|
| `laps_clustered.parquet` | K-Means: `cluster_label` (0=push, 1=base, 2=degraded), `cluster_semantic` |
| `laps_anomalies.parquet` | Isolation Forest: `is_anomaly`, `anomaly_score` |
| `anomalies_summary.parquet` | Sumário de anomalias por piloto |
| `laps_changepoints.parquet` | PELT: `stint_regime`, `is_cliff_lap` |
| `tire_cliffs.parquet` | Tire cliffs por (Driver, Stint): `has_cliff`, `cliff_lap`, `cliff_delta_magnitude`, `cliff_validated` |
| `tire_cliffs_summary.parquet` | Sumário de cliffs por piloto |

## Semântica de Clusters (K-Means)

k=3 é fixo — prior físico F1. A numeração do K-Means é arbitrária; a função `normalize_cluster_semantics()` atribui semântica determinística por piloto:

1. **Base** (`cluster_label=1`) → cluster com mais voltas (ritmo dominante da corrida)
2. **Push** (`cluster_label=0`) → menor `LapTime_delta` entre os restantes (mais rápido)
3. **Degraded** (`cluster_label=2`) → maior `LapTime_delta` entre os restantes (mais lento)

Garante `cluster_semantic` consistente entre pilotos e corridas para consumo downstream (LLM, API).

## Configuração

Todos os parâmetros estão em `config.yaml`:

```yaml
ml:
  random_state: 42
  anomaly:
    contamination_profiles:
      clean: 0.03
      normal: 0.05
      chaotic: 0.10
  clustering:
    n_clusters: 3
  degradation:
    penalty: 3           # Calibrar via cli/ruptures_analysis.py --penalty-search
    min_cliff_magnitude: 0.3
mlflow:
  enabled: true
```

## Observabilidade (MLFlow)

Com `mlflow.enabled: true`, cada execução do pipeline cria um run com:

**Parâmetros logados:**
- `analysis_type`, `scaler_type`, `contamination_profile`, `contamination`
- `kmeans_k`, `structural_filter_threshold`
- `cluster_features`, `anomaly_features`

**Métricas logadas:**
- Por piloto: `driver_VER_silhouette`, `driver_HAM_davies_bouldin`, etc.
- Globais: `silhouette_mean_per_driver`, `n_anomalies`, `anomaly_rate`, `cliff_rate`

```bash
uv run mlflow ui   # http://localhost:5000
```

## Referências

- [src/ml/README.md](../src/ml/README.md) - Documentação detalhada dos módulos
- [MLFLOW_SETUP.md](../MLFLOW_SETUP.md) - Guia do MLFlow
- [docs/configuration.md](configuration.md) - Referência completa do config.yaml
- [cli/README.md](../cli/README.md) - Comandos CLI
