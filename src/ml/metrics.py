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


def calculate_changepoint_metrics(
    changepoints_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Calcula métricas agregadas de detecção de change points.

    Args:
        changepoints_df: DataFrame com resultados por (Driver, Stint),
                         output de detect_tire_changepoints()

    Returns:
        Dicionário com métricas:
        - n_stints_analyzed: Total de stints analisados
        - n_cliffs_detected: Número de stints com cliff detectado
        - cliff_rate: Taxa de stints com cliff (%)
        - cliff_validated_rate: Taxa de cliffs validados pelo anomaly_score (%)
        - mean_cliff_magnitude: Magnitude média do cliff (LapTime_delta)
        - mean_laps_before_cliff: Média de voltas antes do primeiro cliff

    Example:
        >>> metrics = calculate_changepoint_metrics(changepoints_df)
        >>> print(f"Cliff rate: {metrics['cliff_rate']:.1f}%")
    """
    if changepoints_df.empty:
        return {
            'n_stints_analyzed': 0,
            'n_cliffs_detected': 0,
            'cliff_rate': 0.0,
            'cliff_validated_rate': 0.0,
            'mean_cliff_magnitude': 0.0,
            'mean_laps_before_cliff': 0.0,
        }

    n_stints = len(changepoints_df)
    cliffs = changepoints_df[changepoints_df['has_cliff'] == True]
    n_cliffs = len(cliffs)
    cliff_rate = 100.0 * n_cliffs / n_stints if n_stints > 0 else 0.0

    # Validação: cliffs validados pelo anomaly_score
    if 'cliff_validated' in cliffs.columns:
        n_validated = int(cliffs['cliff_validated'].sum())
        cliff_validated_rate = 100.0 * n_validated / n_cliffs if n_cliffs > 0 else 0.0
    else:
        cliff_validated_rate = 0.0

    # Magnitude média do cliff
    if 'cliff_delta_magnitude' in cliffs.columns:
        mean_magnitude = float(cliffs['cliff_delta_magnitude'].abs().mean()) if n_cliffs > 0 else 0.0
    else:
        mean_magnitude = 0.0

    # Média de voltas antes do primeiro cliff
    if 'cliff_lap' in changepoints_df.columns and 'n_laps_in_stint' in changepoints_df.columns:
        # cliff_lap é o LapNumber absoluto; usar laps_before_cliff se disponível
        if 'laps_before_cliff' in changepoints_df.columns:
            mean_laps_before = float(cliffs['laps_before_cliff'].mean()) if n_cliffs > 0 else 0.0
        else:
            mean_laps_before = 0.0
    else:
        mean_laps_before = 0.0

    return {
        'n_stints_analyzed': int(n_stints),
        'n_cliffs_detected': int(n_cliffs),
        'cliff_rate': float(cliff_rate),
        'cliff_validated_rate': float(cliff_validated_rate),
        'mean_cliff_magnitude': float(mean_magnitude),
        'mean_laps_before_cliff': float(mean_laps_before),
    }


def compute_driver_quality_score(
    driver: str,
    silhouette_score_val: float | None,
    n_laps_analyzed: int,
    anomaly_rate: float,
    n_stints: int,
) -> float:
    """
    Score composto 0-1 que representa a confiabilidade dos dados de um piloto.

    Score baixo indica que o piloto tem poucos dados limpos (DNF precoce, muitas anomalias)
    e o relatório LLM deve confiar menos em narrativas detalhadas desse piloto.

    Args:
        driver: Código do piloto (apenas para logging)
        silhouette_score_val: Silhouette score do clustering do piloto (None se não calculado)
        n_laps_analyzed: Número de voltas analisadas (após filtros estruturais)
        anomaly_rate: Taxa de anomalias do piloto (0.0 a 1.0)
        n_stints: Número de stints completados pelo piloto

    Returns:
        Score entre 0 (dados ruins) e 1 (dados excelentes)

    Composição:
        - 40% clustering quality (silhouette): clusters bem separados = dados ricos
        - 30% quantidade de dados: mínimo 30 laps para score completo
        - 20% taxa de anomalias saudável: >15% anomalias indica dados problemáticos
        - 10% número de stints: mínimo 2 stints para análise estratégica completa
    """
    score = 0.0

    # 40% — qualidade do clustering
    if silhouette_score_val is not None and not np.isnan(silhouette_score_val):
        score += min(1.0, max(0.0, float(silhouette_score_val))) * 0.40
    else:
        score += 0.10  # Score mínimo se não foi possível calcular

    # 30% — quantidade de dados limpos
    score += min(1.0, n_laps_analyzed / 30.0) * 0.30

    # 20% — taxa de anomalias dentro do esperado (>15% = dados problemáticos)
    if anomaly_rate <= 0.15:
        score += (1.0 - anomaly_rate / 0.15) * 0.20
    # Acima de 15% = penalidade total nesse componente

    # 10% — número de stints (piloto com 1 stint = menos contexto estratégico)
    score += min(1.0, n_stints / 2.0) * 0.10

    return round(score, 3)


def compute_all_driver_quality_scores(
    laps_anomalies: pd.DataFrame,
    per_driver_clustering: dict,
    tire_cliffs: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    Calcula o score de qualidade de dados para todos os pilotos.

    Args:
        laps_anomalies: DataFrame com is_anomaly e Driver
        per_driver_clustering: Output de calculate_per_driver_clustering_metrics()
        tire_cliffs: DataFrame de cliffs por (Driver, Stint), opcional

    Returns:
        Dict {driver_code: quality_score}
    """
    scores: dict[str, float] = {}

    if laps_anomalies.empty or "Driver" not in laps_anomalies.columns:
        return scores

    # Mapear silhouette por piloto
    silhouette_map: dict[str, float] = {}
    for entry in per_driver_clustering.get("per_driver", []):
        silhouette_map[entry["driver"]] = entry.get("silhouette_score", 0.0)

    # Mapear stints por piloto (se disponível)
    stints_map: dict[str, int] = {}
    if tire_cliffs is not None and not tire_cliffs.empty and "Driver" in tire_cliffs.columns:
        stints_map = tire_cliffs.groupby("Driver").size().to_dict()

    for driver, driver_df in laps_anomalies.groupby("Driver"):
        n_laps = len(driver_df)
        n_anomalies = int(driver_df["is_anomaly"].sum()) if "is_anomaly" in driver_df.columns else 0
        anomaly_rate = n_anomalies / n_laps if n_laps > 0 else 0.0

        scores[str(driver)] = compute_driver_quality_score(
            driver=str(driver),
            silhouette_score_val=silhouette_map.get(str(driver)),
            n_laps_analyzed=n_laps,
            anomaly_rate=anomaly_rate,
            n_stints=stints_map.get(str(driver), 1),
        )

    return scores


def evaluate_clustering_quality(
    metrics: dict[str, float],
    silhouette_threshold: float | None = None,
    davies_bouldin_threshold: float | None = None,
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
    # Ler thresholds do config se não fornecidos explicitamente
    if silhouette_threshold is None or davies_bouldin_threshold is None:
        from src.utils.config import get_config as _get_config
        _cfg = _get_config()
        if silhouette_threshold is None:
            silhouette_threshold = _cfg.get_silhouette_threshold()
        if davies_bouldin_threshold is None:
            davies_bouldin_threshold = _cfg.get_davies_bouldin_threshold()

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
