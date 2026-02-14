"""
Clusterização não supervisionada para análise de ritmo de corrida.

Implementa K-Means e DBSCAN para agrupar voltas semelhantes e identificar
o "ritmo de corrida" real versus voltas de economia ou tráfego.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def find_optimal_k(
    X: np.ndarray | pd.DataFrame,
    k_range: range = range(2, 6),
    method: str = "silhouette",
) -> int:
    """
    Encontra o número ótimo de clusters (k) usando método do cotovelo ou silhueta.

    Args:
        X: Matriz de features (já escalonada)
        k_range: Range de valores de k para testar (padrão: 2 a 5)
        method: Método de seleção:
               - 'silhouette': Usa silhouette score (maior é melhor)
               - 'elbow': Usa método do cotovelo (inércia)

    Returns:
        Valor ótimo de k

    Example:
        >>> from src.preprocessing.feature_engineering import scale_features
        >>> laps_scaled = scale_features(laps_df, ['LapTime_seconds', 'Sector1Time_seconds'])
        >>> optimal_k = find_optimal_k(laps_scaled[['LapTime_seconds', 'Sector1Time_seconds']])
        >>> print(f"Número ótimo de clusters: {optimal_k}")
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    if method == "silhouette":
        # Usar silhouette score (maior = melhor)
        best_k = 2
        best_score = -1

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)

            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    else:  # elbow method
        # Usar inércia (menor = melhor, mas buscar "cotovelo")
        inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Calcular "cotovelo" (mudança na taxa de decréscimo)
        # Simplificado: escolher k onde a redução de inércia diminui
        if len(inertias) >= 3:
            deltas = np.diff(inertias)
            delta_deltas = np.diff(deltas)
            # Ponto onde a aceleração muda mais
            elbow_idx = np.argmax(np.abs(delta_deltas))
            return list(k_range)[elbow_idx + 1]
        else:
            return list(k_range)[0]


def cluster_laps_kmeans(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_clusters: int | None = None,
    group_by: str | None = None,
) -> pd.DataFrame:
    """
    Agrupa voltas usando K-Means para identificar diferentes ritmos de pilotagem.

    Classifica as voltas de um piloto em grupos como "Ritmo Puro", "Gestão de Pneus"
    e "Tráfego". O centroide de cada cluster representa o ritmo médio daquele modo.

    Args:
        df: DataFrame com dados de voltas (já pré-processado e escalonado)
        feature_columns: Colunas para usar no clustering (ex: ['LapTime_seconds', 'Sector1Time_seconds'])
        n_clusters: Número de clusters. Se None, encontra automaticamente usando silhouette
        group_by: Coluna para agrupar antes do clustering (ex: 'Driver' para analisar cada piloto)

    Returns:
        DataFrame com coluna adicional:
        - cluster_label: Label do cluster (0, 1, 2, ...)
        - cluster_centroid_distance: Distância ao centroide do cluster

    Rationale:
        - K-Means é rápido e eficiente para grupos esféricos
        - Bom para categorizar ritmos quando você sabe quantos existem
        - Centroide representa o "ritmo típico" daquele modo

    Example:
        >>> # Pré-processar e escalonar
        >>> laps_processed = enrich_dataframe_with_stats(laps_df, ...)
        >>> laps_scaled = scale_features(laps_processed, ['LapTime_seconds'])
        >>> 
        >>> # Clustering por piloto
        >>> laps_clustered = cluster_laps_kmeans(
        ...     laps_scaled,
        ...     feature_columns=['LapTime_seconds', 'Sector1Time_seconds'],
        ...     n_clusters=3,
        ...     group_by='Driver'
        ... )
        >>> 
        >>> # Analisar clusters
        >>> for driver in laps_clustered['Driver'].unique():
        ...     driver_laps = laps_clustered[laps_clustered['Driver'] == driver]
        ...     print(f"{driver}: {driver_laps['cluster_label'].value_counts()}")
    """
    df = df.copy()

    # Verificar se features existem
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas não encontradas: {missing_cols}")

    # Se agrupar, processar cada grupo separadamente
    if group_by:
        # Inicializar colunas de resultado
        df['cluster_label'] = -1
        df['cluster_centroid_distance'] = np.nan

        for group_name, group_df in df.groupby(group_by):
            if len(group_df) < 2:
                # Grupo muito pequeno, pular
                continue

            X = group_df[feature_columns].values

            # Encontrar k ótimo se não especificado
            k = n_clusters
            if k is None:
                max_k = min(5, len(group_df) - 1)
                if max_k >= 2:
                    k = find_optimal_k(X, k_range=range(2, max_k + 1))
                else:
                    k = 2

            # Aplicar K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            # Calcular distância ao centroide
            distances = np.sqrt(((X - kmeans.cluster_centers_[labels]) ** 2).sum(axis=1))

            # Atribuir resultados
            df.loc[group_df.index, 'cluster_label'] = labels
            df.loc[group_df.index, 'cluster_centroid_distance'] = distances

    else:
        # Processar todo o dataset como um único grupo
        X = df[feature_columns].values

        # Encontrar k ótimo se não especificado
        k = n_clusters
        if k is None:
            max_k = min(5, len(df) - 1)
            if max_k >= 2:
                k = find_optimal_k(X, k_range=range(2, max_k + 1))
            else:
                k = 2

        # Aplicar K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df['cluster_label'] = kmeans.fit_predict(X)

        # Calcular distância ao centroide
        df['cluster_centroid_distance'] = np.sqrt(
            ((X - kmeans.cluster_centers_[df['cluster_label']]) ** 2).sum(axis=1)
        )

    return df


def cluster_laps_dbscan(
    df: pd.DataFrame,
    feature_columns: list[str],
    eps: float | None = None,
    min_samples: int = 3,
    group_by: str | None = None,
) -> pd.DataFrame:
    """
    Agrupa voltas usando DBSCAN para identificar ritmo consistente e ruído.

    DBSCAN identifica o "pelotão" de voltas consistentes e trata o resto como ruído.
    É superior ao K-Means se os clusters tiverem formas irregulares ou se você não
    souber quantos "ritmos" existem.

    Args:
        df: DataFrame com dados de voltas (já pré-processado e escalonado)
        feature_columns: Colunas para usar no clustering
        eps: Distância máxima entre voltas para serem consideradas vizinhas.
            Se None, estima automaticamente usando média das distâncias
        min_samples: Mínimo de voltas para formar um ritmo consistente (padrão: 3)
        group_by: Coluna para agrupar antes do clustering (ex: 'Driver')

    Returns:
        DataFrame com colunas adicionais:
        - cluster_label: Label do cluster (-1 = ruído/outlier, 0+ = cluster)
        - is_noise: Flag binária indicando se é ruído (True) ou cluster (False)

    Rationale:
        - DBSCAN detecta clusters de forma arbitrária (não assume esferas)
        - Identifica automaticamente outliers como "ruído" (label -1)
        - Não precisa especificar número de clusters antecipadamente
        - Ideal para limpar análise focando apenas no ritmo real

    Example:
        >>> # Pré-processar e escalonar
        >>> laps_scaled = scale_features(laps_df, ['LapTime_seconds', 'Sector1Time_seconds'])
        >>> 
        >>> # Clustering DBSCAN
        >>> laps_clustered = cluster_laps_dbscan(
        ...     laps_scaled,
        ...     feature_columns=['LapTime_seconds', 'Sector1Time_seconds'],
        ...     min_samples=5,
        ...     group_by='Driver'
        ... )
        >>> 
        >>> # Filtrar ruído
        >>> clean_laps = laps_clustered[~laps_clustered['is_noise']]
        >>> print(f"Voltas válidas: {len(clean_laps)}, Ruído: {laps_clustered['is_noise'].sum()}")
    """
    df = df.copy()

    # Verificar se features existem
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas não encontradas: {missing_cols}")

    # Se agrupar, processar cada grupo separadamente
    if group_by:
        # Inicializar colunas de resultado
        df['cluster_label'] = -1
        df['is_noise'] = True

        for group_name, group_df in df.groupby(group_by):
            if len(group_df) < min_samples:
                # Grupo muito pequeno, tudo é ruído
                continue

            X = group_df[feature_columns].values

            # Estimar eps se não especificado
            eps_value = eps
            if eps_value is None:
                # Usar 10% da distância média entre pontos como heurística
                from sklearn.neighbors import NearestNeighbors
                neighbors = NearestNeighbors(n_neighbors=min_samples)
                neighbors.fit(X)
                distances, _ = neighbors.kneighbors(X)
                eps_value = np.percentile(distances[:, -1], 90)

            # Aplicar DBSCAN
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            # Atribuir resultados
            df.loc[group_df.index, 'cluster_label'] = labels
            df.loc[group_df.index, 'is_noise'] = labels == -1

    else:
        # Processar todo o dataset
        X = df[feature_columns].values

        # Estimar eps se não especificado
        eps_value = eps
        if eps_value is None:
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors.fit(X)
            distances, _ = neighbors.kneighbors(X)
            eps_value = np.percentile(distances[:, -1], 90)

        # Aplicar DBSCAN
        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
        df['cluster_label'] = dbscan.fit_predict(X)
        df['is_noise'] = df['cluster_label'] == -1

    return df
