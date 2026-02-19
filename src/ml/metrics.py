"""
Métricas de avaliação para modelos de Machine Learning.

Fornece métricas completas para clustering e detecção de anomalias,
incluindo métricas intrínsecas (sem labels) e estatísticas descritivas.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def calculate_clustering_metrics(
    X: np.ndarray | pd.DataFrame,
    labels: np.ndarray,
    remove_noise: bool = True,
) -> dict[str, float]:
    """
    Calcula métricas completas de avaliação de clustering.

    Métricas intrínsecas (não requerem ground truth):
    - Silhouette Score: [-1, 1], maior é melhor (coesão vs separação)
    - Davies-Bouldin Index: [0, ∞), menor é melhor (compacidade vs separação)
    - Calinski-Harabasz Score: [0, ∞), maior é melhor (ratio between/within variance)
    - Inertia: Soma das distâncias quadradas ao centroide mais próximo (apenas K-Means)

    Args:
        X: Matriz de features (n_samples, n_features)
        labels: Labels de cluster (n_samples,)
        remove_noise: Se True, remove pontos com label -1 (ruído do DBSCAN)

    Returns:
        Dicionário com métricas calculadas:
        - silhouette_score: Qualidade geral do clustering
        - davies_bouldin_score: Similaridade entre clusters (menor = melhor)
        - calinski_harabasz_score: Dispersão entre clusters (maior = melhor)
        - n_clusters: Número de clusters (excluindo ruído)
        - n_samples: Número de amostras usadas
        - n_noise: Número de pontos classificados como ruído (DBSCAN)

    Rationale:
        - Silhouette: Mede se cada ponto está bem atribuído ao seu cluster
        - Davies-Bouldin: Mede se clusters são compactos e bem separados
        - Calinski-Harabasz: Razão entre dispersão inter-cluster e intra-cluster

    Example:
        >>> from sklearn.cluster import KMeans
        >>> kmeans = KMeans(n_clusters=3)
        >>> labels = kmeans.fit_predict(X)
        >>> metrics = calculate_clustering_metrics(X, labels)
        >>> print(f"Silhouette: {metrics['silhouette_score']:.3f}")
        >>> print(f"Davies-Bouldin: {metrics['davies_bouldin_score']:.3f}")
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    labels = np.array(labels)

    # Filtrar ruído (label -1) se solicitado
    if remove_noise:
        mask = labels != -1
        X_clean = X[mask]
        labels_clean = labels[mask]
        n_noise = np.sum(~mask)
    else:
        X_clean = X
        labels_clean = labels
        n_noise = 0

    # Verificar se há clusters suficientes
    unique_labels = np.unique(labels_clean)
    n_clusters = len(unique_labels)

    metrics = {
        'n_clusters': n_clusters,
        'n_samples': len(X_clean),
        'n_noise': n_noise,
    }

    # Calcular métricas apenas se houver pelo menos 2 clusters
    if n_clusters >= 2 and len(X_clean) >= 2:
        try:
            metrics['silhouette_score'] = silhouette_score(X_clean, labels_clean)
        except Exception as e:
            metrics['silhouette_score'] = None
            metrics['silhouette_error'] = str(e)

        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_clean, labels_clean)
        except Exception as e:
            metrics['davies_bouldin_score'] = None
            metrics['davies_bouldin_error'] = str(e)

        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_clean, labels_clean)
        except Exception as e:
            metrics['calinski_harabasz_score'] = None
            metrics['calinski_harabasz_error'] = str(e)
    else:
        metrics['silhouette_score'] = None
        metrics['davies_bouldin_score'] = None
        metrics['calinski_harabasz_score'] = None
        metrics['error'] = f"Insufficient clusters ({n_clusters}) or samples ({len(X_clean)})"

    return metrics


def calculate_inertia(
    X: np.ndarray | pd.DataFrame,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> float:
    """
    Calcula a inércia (within-cluster sum of squares) para K-Means.

    A inércia mede quão próximos os pontos de um cluster estão do seu centroide.
    Menor inércia = clusters mais compactos.

    Args:
        X: Matriz de features (n_samples, n_features)
        labels: Labels de cluster (n_samples,)
        centroids: Centróides dos clusters (n_clusters, n_features)

    Returns:
        Inércia (soma das distâncias quadradas aos centróides)

    Example:
        >>> from sklearn.cluster import KMeans
        >>> kmeans = KMeans(n_clusters=3)
        >>> kmeans.fit(X)
        >>> # A inércia também está disponível em kmeans.inertia_
        >>> inertia = calculate_inertia(X, kmeans.labels_, kmeans.cluster_centers_)
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    labels = np.array(labels)
    centroids = np.array(centroids)

    inertia = 0.0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            distances = np.sum((cluster_points - centroid) ** 2)
            inertia += distances

    return inertia


def calculate_anomaly_metrics(
    predictions: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Calcula métricas de avaliação para detecção de anomalias.

    Métricas (sem ground truth):
    - Número total de anomalias detectadas
    - Taxa de anomalias (%)
    - Estatísticas dos scores (se disponíveis)

    Args:
        predictions: Predições (1 = normal, -1 = anomalia)
        scores: Scores de anomalia (opcional, valores negativos = anômalos)

    Returns:
        Dicionário com métricas:
        - n_total: Total de amostras
        - n_anomalies: Número de anomalias
        - anomaly_rate: Taxa de anomalias (%)
        - score_mean: Média dos scores (se disponível)
        - score_std: Desvio padrão dos scores (se disponível)
        - score_min: Score mínimo (mais anômalo, se disponível)
        - score_max: Score máximo (menos anômalo, se disponível)

    Example:
        >>> from sklearn.ensemble import IsolationForest
        >>> iso_forest = IsolationForest(contamination=0.05)
        >>> predictions = iso_forest.fit_predict(X)
        >>> scores = iso_forest.score_samples(X)
        >>> metrics = calculate_anomaly_metrics(predictions, scores)
        >>> print(f"Anomalias: {metrics['n_anomalies']} ({metrics['anomaly_rate']:.2f}%)")
    """
    predictions = np.array(predictions)

    n_total = len(predictions)
    n_anomalies = np.sum(predictions == -1)
    anomaly_rate = 100 * n_anomalies / n_total if n_total > 0 else 0

    metrics = {
        'n_total': n_total,
        'n_anomalies': int(n_anomalies),
        'anomaly_rate': anomaly_rate,
    }

    # Adicionar estatísticas dos scores se disponíveis
    if scores is not None:
        scores = np.array(scores)
        metrics['score_mean'] = float(np.mean(scores))
        metrics['score_std'] = float(np.std(scores))
        metrics['score_min'] = float(np.min(scores))
        metrics['score_max'] = float(np.max(scores))

        # Scores das anomalias vs normais
        anomaly_mask = predictions == -1
        if np.any(anomaly_mask):
            metrics['anomaly_score_mean'] = float(np.mean(scores[anomaly_mask]))
        if np.any(~anomaly_mask):
            metrics['normal_score_mean'] = float(np.mean(scores[~anomaly_mask]))

    return metrics


def calculate_cluster_statistics(
    df: pd.DataFrame,
    cluster_column: str = 'cluster_label',
    feature_columns: list[str] | None = None,
    driver_column: str | None = None,
) -> pd.DataFrame:
    """
    Calcula estatísticas descritivas por cluster (e por piloto, se informado).

    Args:
        df: DataFrame com dados e labels de cluster
        cluster_column: Nome da coluna com labels de cluster
        feature_columns: Colunas para calcular estatísticas (se None, usa todas numéricas)
        driver_column: Se fornecido, agrupa por (driver, cluster) e calcula
                       cluster_size_pct como % das voltas do piloto naquele cluster.

    Returns:
        DataFrame com estatísticas por cluster:
        - [driver_column]: Piloto (somente se driver_column fornecido)
        - cluster_label: ID do cluster (0=push, 1=base, 2=degraded)
        - cluster_semantic: Label textual do cluster (se coluna existir no df)
        - size: Número de voltas no cluster
        - cluster_size_pct: % das voltas do piloto/total nesse cluster
        - {feature}_mean: Média de cada feature
        - {feature}_std: Desvio padrão de cada feature

    Example:
        >>> laps_clustered = cluster_laps_kmeans(...)
        >>> stats = calculate_cluster_statistics(
        ...     laps_clustered,
        ...     cluster_column='cluster_label',
        ...     feature_columns=['LapTime_delta', 'TyreAge_normalized'],
        ...     driver_column='Driver',
        ... )
        >>> print(stats)
    """
    if cluster_column not in df.columns:
        raise ValueError(f"Coluna '{cluster_column}' não encontrada no DataFrame")

    has_semantic = 'cluster_semantic' in df.columns
    use_driver = driver_column is not None and driver_column in df.columns

    # Se feature_columns não especificado, usar todas colunas numéricas
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if cluster_column in feature_columns:
            feature_columns.remove(cluster_column)

    def _cluster_stats(cluster_df: pd.DataFrame, n_reference: int) -> dict:
        stats = {
            'size': len(cluster_df),
            'cluster_size_pct': round(100 * len(cluster_df) / n_reference, 2),
        }
        if has_semantic and len(cluster_df) > 0:
            stats['cluster_semantic'] = cluster_df['cluster_semantic'].iloc[0]
        for col in feature_columns:
            if col in cluster_df.columns:
                stats[f'{col}_mean'] = cluster_df[col].mean()
                stats[f'{col}_std'] = cluster_df[col].std()
        return stats

    stats_list = []

    if use_driver:
        for driver, driver_df in df.groupby(driver_column):
            n_driver_laps = len(driver_df)
            for cluster_id in sorted(driver_df[cluster_column].unique()):
                cluster_df = driver_df[driver_df[cluster_column] == cluster_id]
                stats = {driver_column: driver, 'cluster_label': cluster_id}
                stats.update(_cluster_stats(cluster_df, n_driver_laps))
                stats_list.append(stats)
    else:
        n_total = len(df)
        for cluster_id in sorted(df[cluster_column].unique()):
            cluster_df = df[df[cluster_column] == cluster_id]
            stats = {'cluster_label': cluster_id}
            stats.update(_cluster_stats(cluster_df, n_total))
            stats_list.append(stats)

    return pd.DataFrame(stats_list)


def calculate_per_driver_clustering_metrics(
    df: pd.DataFrame,
    feature_columns: list[str],
    labels_column: str = 'cluster_label',
    driver_column: str = 'Driver',
) -> dict:
    """
    Calcula métricas de clustering por piloto e agrega resultados.

    Em vez de calcular o Silhouette Score no dataset inteiro (que mistura pilotos),
    calcula por piloto separadamente e retorna média/std entre pilotos.
    Isso é mais correto porque cada piloto tem seu próprio espaço de clusters.

    Args:
        df: DataFrame com dados clusterizados
        feature_columns: Colunas de features usadas no clustering
        labels_column: Coluna com labels de cluster
        driver_column: Coluna de identificação do piloto

    Returns:
        Dicionário com:
        - silhouette_mean: Média do Silhouette Score entre pilotos
        - silhouette_std: Desvio padrão
        - silhouette_min / silhouette_max: Limites
        - davies_bouldin_mean: Média do Davies-Bouldin entre pilotos
        - calinski_harabasz_mean: Média do Calinski-Harabasz entre pilotos
        - n_drivers_evaluated: Número de pilotos avaliados
        - per_driver: Lista com métricas individuais por piloto

    Example:
        >>> per_driver_metrics = calculate_per_driver_clustering_metrics(
        ...     laps_clustered,
        ...     feature_columns=['LapTime_delta', 'TyreAge_normalized'],
        ... )
        >>> print(f"Silhouette médio: {per_driver_metrics['silhouette_mean']:.3f}")
    """
    if driver_column not in df.columns:
        return {
            'silhouette_mean': None,
            'davies_bouldin_mean': None,
            'calinski_harabasz_mean': None,
            'n_drivers_evaluated': 0,
            'per_driver': [],
        }

    driver_metrics = []

    for driver, driver_df in df.groupby(driver_column):
        # Excluir ruído do DBSCAN (label -1)
        clean_df = driver_df[driver_df[labels_column] != -1]

        available_features = [col for col in feature_columns if col in clean_df.columns]
        if not available_features or len(clean_df) < 4:
            continue

        X = clean_df[available_features].values
        labels = clean_df[labels_column].values
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            continue

        try:
            s = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)

            driver_metrics.append({
                'driver': driver,
                'silhouette_score': float(s),
                'davies_bouldin_score': float(db),
                'calinski_harabasz_score': float(ch),
                'n_clusters': int(len(unique_labels)),
                'n_laps': int(len(X)),
            })
        except Exception:
            pass

    if not driver_metrics:
        return {
            'silhouette_mean': None,
            'davies_bouldin_mean': None,
            'calinski_harabasz_mean': None,
            'n_drivers_evaluated': 0,
            'per_driver': [],
        }

    per_driver_df = pd.DataFrame(driver_metrics)

    return {
        'silhouette_mean': float(per_driver_df['silhouette_score'].mean()),
        'silhouette_std': float(per_driver_df['silhouette_score'].std()),
        'silhouette_min': float(per_driver_df['silhouette_score'].min()),
        'silhouette_max': float(per_driver_df['silhouette_score'].max()),
        'davies_bouldin_mean': float(per_driver_df['davies_bouldin_score'].mean()),
        'davies_bouldin_std': float(per_driver_df['davies_bouldin_score'].std()),
        'calinski_harabasz_mean': float(per_driver_df['calinski_harabasz_score'].mean()),
        'n_drivers_evaluated': len(driver_metrics),
        'per_driver': driver_metrics,
    }


def evaluate_clustering_quality(
    metrics: dict[str, float],
    silhouette_threshold: float = 0.5,
    davies_bouldin_threshold: float = 1.0,
) -> dict[str, str | bool]:
    """
    Avalia a qualidade do clustering com base nas métricas calculadas.

    Args:
        metrics: Dicionário com métricas (output de calculate_clustering_metrics)
        silhouette_threshold: Threshold mínimo para bom silhouette score
        davies_bouldin_threshold: Threshold máximo para bom Davies-Bouldin

    Returns:
        Dicionário com avaliação:
        - quality: 'excellent', 'good', 'fair', 'poor'
        - silhouette_pass: True se silhouette >= threshold
        - davies_bouldin_pass: True se DB <= threshold
        - recommendation: Recomendação textual

    Example:
        >>> metrics = calculate_clustering_metrics(X, labels)
        >>> evaluation = evaluate_clustering_quality(metrics)
        >>> print(f"Qualidade: {evaluation['quality']}")
        >>> print(f"Recomendação: {evaluation['recommendation']}")
    """
    silhouette = metrics.get('silhouette_score')
    davies_bouldin = metrics.get('davies_bouldin_score')

    evaluation = {
        'silhouette_pass': False,
        'davies_bouldin_pass': False,
    }

    # Avaliar silhouette
    if silhouette is not None:
        evaluation['silhouette_pass'] = silhouette >= silhouette_threshold

    # Avaliar Davies-Bouldin
    if davies_bouldin is not None:
        evaluation['davies_bouldin_pass'] = davies_bouldin <= davies_bouldin_threshold

    # Determinar qualidade geral
    if evaluation['silhouette_pass'] and evaluation['davies_bouldin_pass']:
        evaluation['quality'] = 'excellent'
        evaluation['recommendation'] = 'Clustering de alta qualidade. Clusters bem separados e coesos.'
    elif evaluation['silhouette_pass'] or evaluation['davies_bouldin_pass']:
        evaluation['quality'] = 'good'
        evaluation['recommendation'] = 'Clustering aceitável. Considere ajustar hiperparâmetros para melhorar.'
    elif silhouette is not None and silhouette > 0:
        evaluation['quality'] = 'fair'
        evaluation['recommendation'] = 'Clustering fraco. Experimente diferentes valores de k ou use DBSCAN.'
    else:
        evaluation['quality'] = 'poor'
        evaluation['recommendation'] = 'Clustering de baixa qualidade. Revise features e algoritmo.'

    return evaluation
