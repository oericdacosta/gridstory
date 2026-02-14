"""
Módulo de Machine Learning usando Scikit-learn.

Pipeline não supervisionado para análise de corridas de F1:
- Clusterização: K-Means e DBSCAN para identificar ritmos e padrões
- Detecção de Anomalias: Isolation Forest para eventos raros
- Pipeline: Pré-processamento + ML em um único fluxo
"""

from .clustering import cluster_laps_kmeans, cluster_laps_dbscan, find_optimal_k
from .anomaly_detection import detect_anomalies_isolation_forest
from .pipeline import create_ml_pipeline, run_race_analysis

__all__ = [
    "cluster_laps_kmeans",
    "cluster_laps_dbscan",
    "find_optimal_k",
    "detect_anomalies_isolation_forest",
    "create_ml_pipeline",
    "run_race_analysis",
]
