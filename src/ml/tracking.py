"""
Integração com MLFlow para tracking de experimentos de Machine Learning.

Fornece funções para rastrear parâmetros, métricas e artefatos de modelos
de clustering e detecção de anomalias.
"""

import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from .metrics import (
    calculate_clustering_metrics,
    calculate_anomaly_metrics,
    evaluate_clustering_quality,
)


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str | None = None,
    enable_autolog: bool = False,
) -> None:
    """
    Configura o MLFlow para tracking de experimentos.

    Args:
        experiment_name: Nome do experimento (ex: "F1_2025_Bahrain_Race")
        tracking_uri: URI do servidor de tracking (None = local "./mlruns")
        enable_autolog: Se True, habilita autolog do scikit-learn.
                        Padrão False — autolog cria centenas de child runs (um por fit())
                        que poluem a UI sem informação adicional útil.

    Example:
        >>> setup_mlflow("F1_2025_Round_01")
        >>> # Agora todos os runs serão organizados sob este experimento
    """
    # Configurar tracking URI (padrão: local)
    if tracking_uri is None:
        tracking_uri = "file:./mlruns"

    mlflow.set_tracking_uri(tracking_uri)

    # Criar/definir experimento
    mlflow.set_experiment(experiment_name)

    # Autolog desabilitado por padrão: cria um child run por sklearn fit(),
    # gerando 10k+ arquivos (model pickle, conda.yaml, requirements.txt por run)
    # sem valor adicional — as métricas relevantes são logadas manualmente.
    mlflow.sklearn.autolog(disable=True, silent=True)


def track_clustering_run(
    run_name: str,
    X: np.ndarray | pd.DataFrame,
    labels: np.ndarray,
    params: dict[str, Any],
    model: Any = None,
    additional_metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """
    Tracka um run de clustering com MLFlow.

    Args:
        run_name: Nome descritivo do run (ex: "KMeans_k3_StandardScaler")
        X: Matriz de features usada no clustering
        labels: Labels resultantes do clustering
        params: Parâmetros do modelo (ex: {"n_clusters": 3, "random_state": 42})
        model: Objeto do modelo treinado (opcional, para salvar)
        additional_metrics: Métricas adicionais para logar
        tags: Tags para organizar runs (ex: {"driver": "VER", "algorithm": "kmeans"})

    Returns:
        ID do run criado

    Example:
        >>> from sklearn.cluster import KMeans
        >>> kmeans = KMeans(n_clusters=3, random_state=42)
        >>> labels = kmeans.fit_predict(X)
        >>>
        >>> run_id = track_clustering_run(
        ...     run_name="Stint_Clustering_VER",
        ...     X=X,
        ...     labels=labels,
        ...     params={"n_clusters": 3, "random_state": 42},
        ...     model=kmeans,
        ...     tags={"driver": "VER", "algorithm": "kmeans"}
        ... )
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Logar parâmetros
        mlflow.log_params(params)

        # Logar tags
        if tags:
            mlflow.set_tags(tags)

        # Calcular e logar métricas de clustering
        clustering_metrics = calculate_clustering_metrics(X, labels, remove_noise=True)
        for metric_name, metric_value in clustering_metrics.items():
            if isinstance(metric_value, (int, float)) and metric_value is not None:
                mlflow.log_metric(metric_name, metric_value)

        # Logar métricas adicionais
        if additional_metrics:
            mlflow.log_metrics(additional_metrics)

        # Avaliar qualidade do clustering
        evaluation = evaluate_clustering_quality(clustering_metrics)
        mlflow.log_param("clustering_quality", evaluation['quality'])
        mlflow.set_tag("quality", evaluation['quality'])

        # Salvar modelo (se fornecido)
        if model is not None:
            mlflow.sklearn.log_model(model, "model")

        return run.info.run_id


def track_anomaly_detection_run(
    run_name: str,
    X: np.ndarray | pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray | None,
    params: dict[str, Any],
    model: Any = None,
    additional_metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """
    Tracka um run de detecção de anomalias com MLFlow.

    Args:
        run_name: Nome descritivo do run (ex: "IsolationForest_cont005")
        X: Matriz de features usada na detecção
        predictions: Predições (1 = normal, -1 = anomalia)
        scores: Scores de anomalia (valores negativos = mais anômalos)
        params: Parâmetros do modelo (ex: {"contamination": 0.05, "n_estimators": 100})
        model: Objeto do modelo treinado (opcional)
        additional_metrics: Métricas adicionais para logar
        tags: Tags para organizar runs

    Returns:
        ID do run criado

    Example:
        >>> from sklearn.ensemble import IsolationForest
        >>> iso_forest = IsolationForest(contamination=0.05, random_state=42)
        >>> predictions = iso_forest.fit_predict(X)
        >>> scores = iso_forest.score_samples(X)
        >>>
        >>> run_id = track_anomaly_detection_run(
        ...     run_name="Outlier_Detection_HAM",
        ...     X=X,
        ...     predictions=predictions,
        ...     scores=scores,
        ...     params={"contamination": 0.05, "n_estimators": 100},
        ...     model=iso_forest,
        ...     tags={"driver": "HAM", "algorithm": "isolation_forest"}
        ... )
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Logar parâmetros
        mlflow.log_params(params)

        # Logar tags
        if tags:
            mlflow.set_tags(tags)

        # Calcular e logar métricas de anomaly detection
        anomaly_metrics = calculate_anomaly_metrics(predictions, scores)
        for metric_name, metric_value in anomaly_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        # Logar métricas adicionais
        if additional_metrics:
            mlflow.log_metrics(additional_metrics)

        # Salvar modelo (se fornecido)
        if model is not None:
            mlflow.sklearn.log_model(model, "model")

        return run.info.run_id


def track_pipeline_run(
    run_name: str,
    year: int,
    round_number: int,
    clustering_results: dict | None = None,
    anomaly_results: dict | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    artifacts: dict[str, pd.DataFrame] | None = None,
) -> str:
    """
    Tracka um run completo do pipeline de ML.

    Útil para trackear uma análise completa de corrida que inclui
    múltiplos modelos (clustering + anomaly detection).

    Args:
        run_name: Nome do run (ex: "F1_2025_Bahrain_FullPipeline")
        year: Ano da corrida
        round_number: Número da rodada
        clustering_results: Resultados de clustering (opcional)
        anomaly_results: Resultados de anomaly detection (opcional)
        params: Parâmetros gerais do pipeline
        tags: Tags para organizar
        artifacts: DataFrames para salvar como artefatos CSV.
                   Chave = nome do arquivo (ex: "laps_clustered.csv")

    Returns:
        ID do run criado

    Example:
        >>> run_id = track_pipeline_run(
        ...     run_name="F1_2025_Round01_Analysis",
        ...     year=2025,
        ...     round_number=1,
        ...     clustering_results={"n_clusters": 3, "silhouette": 0.65},
        ...     anomaly_results={"n_anomalies": 12, "anomaly_rate": 2.5},
        ...     artifacts={"laps_clustered.csv": laps_clustered_df},
        ...     tags={"event": "Bahrain GP", "session": "Race"}
        ... )
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Logar metadados da corrida
        mlflow.log_param("year", year)
        mlflow.log_param("round", round_number)

        # Logar parâmetros gerais
        if params:
            mlflow.log_params(params)

        # Logar tags
        if tags:
            mlflow.set_tags(tags)

        # Logar resultados de clustering (apenas métricas por piloto)
        if clustering_results:
            for key, value in clustering_results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"clustering_{key}", value)

        # Logar resultados de anomaly detection
        if anomaly_results:
            for key, value in anomaly_results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"anomaly_{key}", value)

        # Salvar DataFrames como artefatos CSV (visíveis na aba Artifacts do MLFlow UI)
        if artifacts:
            for filename, df in artifacts.items():
                log_dataframe_artifact(df, filename, artifact_path="results")

        return run.info.run_id


def log_dataframe_artifact(
    df: pd.DataFrame,
    filename: str,
    artifact_path: str | None = None,
) -> None:
    """
    Salva um DataFrame como artefato do MLFlow.

    Args:
        df: DataFrame para salvar
        filename: Nome do arquivo (ex: "anomalies_summary.csv")
        artifact_path: Subdiretório dentro de artifacts (opcional)

    Example:
        >>> with mlflow.start_run():
        ...     log_dataframe_artifact(
        ...         df=anomalies_summary,
        ...         filename="anomalies_summary.csv",
        ...         artifact_path="results"
        ...     )
    """
    # Criar diretório temporário
    temp_dir = Path("./tmp_mlflow_artifacts")
    temp_dir.mkdir(exist_ok=True)

    # Salvar DataFrame
    filepath = temp_dir / filename
    if filename.endswith('.csv'):
        df.to_csv(filepath, index=False)
    elif filename.endswith('.parquet'):
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Formato não suportado: {filename}")

    # Logar como artefato
    mlflow.log_artifact(str(filepath), artifact_path=artifact_path)

    # Limpar arquivo temporário
    filepath.unlink()


def get_best_run(
    experiment_name: str,
    metric_name: str,
    ascending: bool = False,
) -> dict | None:
    """
    Recupera o melhor run de um experimento baseado em uma métrica.

    Args:
        experiment_name: Nome do experimento
        metric_name: Nome da métrica para ordenar (ex: "silhouette_score")
        ascending: Se True, menor valor é melhor (ex: Davies-Bouldin)

    Returns:
        Dicionário com informações do melhor run ou None se não encontrado

    Example:
        >>> best_run = get_best_run(
        ...     experiment_name="F1_2025_Round_01",
        ...     metric_name="silhouette_score",
        ...     ascending=False  # Maior silhouette é melhor
        ... )
        >>> if best_run:
        ...     print(f"Melhor run: {best_run['run_name']}")
        ...     print(f"Silhouette: {best_run['metrics']['silhouette_score']}")
    """
    # Buscar experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    # Buscar runs do experimento
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )

    if runs.empty:
        return None

    # Converter para dicionário
    best_run = runs.iloc[0].to_dict()

    return {
        'run_id': best_run['run_id'],
        'run_name': best_run.get('tags.mlflow.runName', 'N/A'),
        'metrics': {k.replace('metrics.', ''): v for k, v in best_run.items() if k.startswith('metrics.')},
        'params': {k.replace('params.', ''): v for k, v in best_run.items() if k.startswith('params.')},
    }


def compare_runs(
    experiment_name: str,
    metric_names: list[str] | None = None,
    max_runs: int = 10,
) -> pd.DataFrame:
    """
    Compara múltiplos runs de um experimento.

    Args:
        experiment_name: Nome do experimento
        metric_names: Métricas para incluir na comparação (None = todas)
        max_runs: Número máximo de runs para retornar

    Returns:
        DataFrame com comparação de runs

    Example:
        >>> comparison = compare_runs(
        ...     experiment_name="F1_2025_Round_01",
        ...     metric_names=["silhouette_score", "davies_bouldin_score"],
        ...     max_runs=5
        ... )
        >>> print(comparison[["run_name", "silhouette_score", "davies_bouldin_score"]])
    """
    # Buscar experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    # Buscar runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_runs,
    )

    if runs.empty:
        return pd.DataFrame()

    # Filtrar colunas
    cols_to_keep = ['run_id', 'start_time', 'end_time', 'status']

    # Adicionar run_name se existir
    if 'tags.mlflow.runName' in runs.columns:
        runs['run_name'] = runs['tags.mlflow.runName']
        cols_to_keep.append('run_name')

    # Adicionar métricas
    metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
    if metric_names:
        metric_cols = [f"metrics.{name}" for name in metric_names if f"metrics.{name}" in runs.columns]

    cols_to_keep.extend(metric_cols)

    # Adicionar parâmetros principais
    param_cols = [col for col in runs.columns if col.startswith('params.')]
    cols_to_keep.extend(param_cols)

    # Filtrar e renomear
    comparison = runs[cols_to_keep].copy()
    comparison.columns = [col.replace('metrics.', '').replace('params.', 'param_') for col in comparison.columns]

    return comparison
