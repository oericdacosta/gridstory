"""
Módulo de Machine Learning usando Scikit-learn.

Pipeline não supervisionado para análise de corridas de F1:
- Clusterização: K-Means e DBSCAN para identificar ritmos e padrões
- Detecção de Anomalias: Isolation Forest para eventos raros
- Pipeline: Pré-processamento + ML em um único fluxo
- Métricas: Avaliação completa de clustering e anomaly detection
- Tracking: Integração com MLFlow para experimentação
"""

from .clustering import cluster_laps_kmeans, cluster_laps_dbscan, find_optimal_k
from .anomaly_detection import detect_anomalies_isolation_forest, summarize_anomalies
from .pipeline import (
    create_ml_pipeline,
    run_race_analysis,
    engineer_lap_features,
    detect_stints,
    filter_structural_laps,
    normalize_cluster_semantics,
)
from .metrics import (
    calculate_clustering_metrics,
    calculate_anomaly_metrics,
    calculate_cluster_statistics,
    calculate_per_driver_clustering_metrics,
    evaluate_clustering_quality,
)
from .tracking import (
    setup_mlflow,
    track_clustering_run,
    track_anomaly_detection_run,
    track_pipeline_run,
    get_best_run,
    compare_runs,
)

__all__ = [
    # Clustering
    "cluster_laps_kmeans",
    "cluster_laps_dbscan",
    "find_optimal_k",
    # Anomaly Detection
    "detect_anomalies_isolation_forest",
    "summarize_anomalies",
    # Pipeline
    "create_ml_pipeline",
    "run_race_analysis",
    "engineer_lap_features",
    "detect_stints",
    "filter_structural_laps",
    "normalize_cluster_semantics",
    # Metrics
    "calculate_clustering_metrics",
    "calculate_anomaly_metrics",
    "calculate_cluster_statistics",
    "calculate_per_driver_clustering_metrics",
    "evaluate_clustering_quality",
    # Tracking
    "setup_mlflow",
    "track_clustering_run",
    "track_anomaly_detection_run",
    "track_pipeline_run",
    "get_best_run",
    "compare_runs",
]
