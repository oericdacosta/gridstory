"""
Pipeline completo de Machine Learning para análise de corridas.

Integra pré-processamento (imputação, encoding, escalonamento) com algoritmos
de ML (clustering, detecção de anomalias) em um fluxo unificado.

Inclui tracking com MLFlow para monitorar experimentos e métricas.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest

from ..preprocessing.feature_engineering import (
    impute_missing_values,
    encode_categorical_variables,
    scale_features,
)
from .clustering import cluster_laps_kmeans, cluster_laps_dbscan
from .anomaly_detection import detect_anomalies_isolation_forest
from .change_point import detect_tire_changepoints, summarize_cliffs
from .metrics import (
    calculate_anomaly_metrics,
    calculate_changepoint_metrics,
    calculate_cluster_statistics,
    calculate_per_driver_clustering_metrics,
)
from .tracking import setup_mlflow, track_pipeline_run


def create_ml_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str] | None = None,
    scaler_type: str = "standard",
) -> ColumnTransformer:
    """
    Cria um pipeline de pré-processamento usando ColumnTransformer.

    O ColumnTransformer aplica transformações diferentes para colunas numéricas
    e categóricas, evitando vazamento de dados (data leakage).

    Args:
        numeric_columns: Colunas numéricas para escalonar
        categorical_columns: Colunas categóricas para codificar (OneHot)
        scaler_type: Tipo de scaler ('standard' ou 'robust')

    Returns:
        ColumnTransformer configurado

    Example:
        >>> from sklearn.ensemble import IsolationForest
        >>> 
        >>> # Criar pipeline de pré-processamento
        >>> preprocessor = create_ml_pipeline(
        ...     numeric_columns=['LapTime_seconds', 'TyreLife'],
        ...     categorical_columns=['Compound'],
        ...     scaler_type='robust'
        ... )
        >>> 
        >>> # Criar pipeline completo
        >>> full_pipeline = Pipeline([
        ...     ('preprocessor', preprocessor),
        ...     ('detector', IsolationForest(contamination=0.05))
        ... ])
        >>> 
        >>> # Treinar e predizer
        >>> predictions = full_pipeline.fit_predict(laps_df)
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    transformers = []

    # Transformador para colunas numéricas
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    transformers.append(('num', scaler, numeric_columns))

    # Transformador para colunas categóricas (se houver)
    if categorical_columns:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        transformers.append(('cat', encoder, categorical_columns))

    # Criar ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Descartar colunas não especificadas
    )

    return preprocessor


def detect_stints(
    df: pd.DataFrame,
    driver_column: str = 'Driver',
    lap_column: str = 'LapNumber',
    tyre_life_column: str = 'TyreLife',
) -> pd.DataFrame:
    """
    Detecta número do stint por piloto a partir de resets no TyreLife.

    Quando o TyreLife diminui de uma volta para outra, indica pit stop (troca de pneu).
    """
    df = df.copy()
    df['Stint'] = 1

    if tyre_life_column not in df.columns:
        return df

    if driver_column in df.columns:
        for driver, driver_df in df.groupby(driver_column):
            if lap_column in df.columns:
                driver_df = driver_df.sort_values(lap_column)

            tyre_life = driver_df[tyre_life_column].ffill().values
            stints = np.ones(len(tyre_life), dtype=int)
            current_stint = 1

            for i in range(1, len(tyre_life)):
                if tyre_life[i] < tyre_life[i - 1]:
                    current_stint += 1
                stints[i] = current_stint

            df.loc[driver_df.index, 'Stint'] = stints

    return df


def engineer_lap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features engenheiradas para melhorar qualidade do clustering.

    Features adicionadas:
    - Stint: Número do stint por piloto (detectado por resets no TyreLife)
    - LapTime_delta: Desvio do tempo de volta em relação à mediana do stint do piloto
      (captura variação de ritmo relativa, eliminando diferença absoluta entre pilotos)
    - TyreAge_normalized: Vida do pneu normalizada 0→1 por stint
      (captura fase de degradação do pneu independente da estratégia)
    - Compound_ordinal: Compound como ordinal (SOFT=1, MEDIUM=2, HARD=3)
    """
    df = df.copy()

    # Compound ordinal: captura dureza do composto como feature contínua
    compound_order = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3, 'INTERMEDIATE': 4, 'WET': 5}
    if 'Compound' in df.columns:
        df['Compound_ordinal'] = df['Compound'].map(compound_order).fillna(2).astype(float)

    # Detectar stints
    df = detect_stints(df)

    # Features por piloto × stint
    if 'Driver' in df.columns and 'LapTime_seconds' in df.columns:
        df['LapTime_delta'] = 0.0
        df['TyreAge_normalized'] = 0.0

        for driver, driver_df in df.groupby('Driver'):
            for stint, stint_df in driver_df.groupby('Stint'):
                idx = stint_df.index

                # LapTime_delta: desvio relativo à mediana do stint
                median_lt = df.loc[idx, 'LapTime_seconds'].median()
                df.loc[idx, 'LapTime_delta'] = df.loc[idx, 'LapTime_seconds'] - median_lt

                # TyreAge_normalized: 0→1 dentro do stint
                if 'TyreLife' in df.columns:
                    tyre_vals = df.loc[idx, 'TyreLife']
                    max_tyre = tyre_vals.max()
                    df.loc[idx, 'TyreAge_normalized'] = (
                        tyre_vals / max_tyre if max_tyre > 0 else 0.0
                    )

    return df


def normalize_cluster_semantics(
    df: pd.DataFrame,
    driver_column: str = 'Driver',
    delta_column: str = 'LapTime_delta',
) -> pd.DataFrame:
    """
    Atribui semântica determinística aos clusters por piloto.

    Lógica por piloto (dois critérios combinados — tamanho e delta):
    1. Base pace  → cluster com mais voltas (ritmo dominante da corrida)
    2. Push       → entre os restantes, cluster com menor delta (mais rápido)
    3. Degraded   → entre os restantes, cluster com maior delta (mais lento)

    Desempate de tamanho: menor abs(delta_mean) → base (ritmo mais próximo
    da mediana do stint, evitando promover cluster de push a base).

    Renumera cluster_label: push=0, base=1, degraded=2 (consistente entre
    pilotos). Adiciona coluna cluster_semantic com os labels textuais.

    Args:
        df: DataFrame com coluna 'cluster_label' e feature de delta
        driver_column: Coluna de identificação do piloto
        delta_column: Coluna de delta de tempo de volta

    Returns:
        DataFrame com 'cluster_label' renumerados e coluna 'cluster_semantic'
    """
    df = df.copy()

    if delta_column not in df.columns or 'cluster_label' not in df.columns:
        return df

    def _build_semantic_mapping(group_df: pd.DataFrame) -> tuple[dict[int, int], bool]:
        """Retorna (mapeamento {old_label → new_label}, is_clean) para um grupo (piloto).

        is_clean=True quando push_delta < base_delta < degraded_delta, ou seja,
        a separação semântica é válida. is_clean=False indica caso degenerado:
        o piloto não tem um cluster de push genuinamente mais rápido que o base,
        ou não tem degradação genuinamente mais lenta (ex: todos os clusters
        residuais estão do mesmo lado do base no eixo de delta).
        """
        stats = (
            group_df.groupby('cluster_label')[delta_column]
            .agg(['count', 'mean'])
            .rename(columns={'count': 'size', 'mean': 'delta_mean'})
        )
        stats['abs_delta'] = stats['delta_mean'].abs()

        # Base: maior tamanho; desempate por menor abs(delta) — ritmo mais central
        stats_sorted = stats.sort_values(['size', 'abs_delta'], ascending=[False, True])
        labels_ordered = stats_sorted.index.tolist()
        base_label = labels_ordered[0]
        remaining = labels_ordered[1:]
        base_delta_val = stats.loc[base_label, 'delta_mean']

        if len(remaining) >= 2:
            # Classificar relativamente ao delta do base:
            # push = cluster com menor delta (abaixo ou o menos acima do base)
            # degraded = cluster com maior delta (acima ou o menos abaixo do base)
            remaining_by_delta = stats.loc[remaining, 'delta_mean'].sort_values()
            push_label = remaining_by_delta.index[0]
            degraded_label = remaining_by_delta.index[-1]
            push_delta = remaining_by_delta.iloc[0]
            degraded_delta = remaining_by_delta.iloc[-1]
            # Caso limpo: push genuinamente mais rápido E degraded genuinamente mais lento
            is_clean = (push_delta < base_delta_val) and (degraded_delta > base_delta_val)
            return {push_label: 0, base_label: 1, degraded_label: 2}, is_clean
        elif len(remaining) == 1:
            other = remaining[0]
            other_delta = stats.loc[other, 'delta_mean']
            if other_delta < base_delta_val:
                return {other: 0, base_label: 1}, True    # push existe, sem degraded
            else:
                return {base_label: 1, other: 2}, True    # degraded existe, sem push
        else:
            return {base_label: 1}, True

    if driver_column in df.columns:
        for driver, driver_df in df.groupby(driver_column):
            mapping, is_clean = _build_semantic_mapping(driver_df)
            df.loc[driver_df.index, 'cluster_label'] = driver_df['cluster_label'].map(mapping)
            df.loc[driver_df.index, 'cluster_semantic_clean'] = is_clean
    else:
        mapping, is_clean = _build_semantic_mapping(df)
        df['cluster_label'] = df['cluster_label'].map(mapping)
        df['cluster_semantic_clean'] = is_clean

    # Adicionar coluna semântica textual (contrato para downstream)
    semantic_map = {0: 'push', 1: 'base', 2: 'degraded'}
    df['cluster_semantic'] = df['cluster_label'].map(semantic_map)

    return df


def filter_structural_laps(
    df: pd.DataFrame,
    threshold: float = 1.5,
    driver_column: str = 'Driver',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove voltas estruturalmente lentas (pit stops, safety car) antes do clustering.

    Estas voltas são anomalias conhecidas e não representam ritmo de corrida.
    São removidas apenas para o clustering; o anomaly detection ainda as processa.

    Args:
        df: DataFrame com dados de voltas
        threshold: Fator multiplicador da mediana para considerar volta como outlier estrutural
                  (1.5 = voltas >50% mais lentas que a mediana do piloto são filtradas)
        driver_column: Coluna de identificação do piloto

    Returns:
        Tuple (clean_df, filtered_df) onde filtered_df contém as voltas removidas
    """
    if 'LapTime_seconds' not in df.columns:
        return df.copy(), pd.DataFrame()

    clean_mask = pd.Series(True, index=df.index)

    if driver_column in df.columns:
        for driver, driver_df in df.groupby(driver_column):
            median_lt = driver_df['LapTime_seconds'].median()
            slow_idx = driver_df.index[driver_df['LapTime_seconds'] > threshold * median_lt]
            clean_mask.loc[slow_idx] = False
    else:
        median_lt = df['LapTime_seconds'].median()
        clean_mask = df['LapTime_seconds'] <= threshold * median_lt

    return df[clean_mask].copy(), df[~clean_mask].copy()


def run_race_analysis(
    laps_df: pd.DataFrame,
    analysis_type: str = "all",
    driver: str | None = None,
    enable_mlflow: bool | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    year: int | None = None,
    round_number: int | None = None,
    contamination_profile: str = 'normal',
) -> dict[str, pd.DataFrame]:
    """
    Executa análise completa de ML em dados de voltas de uma corrida.

    Pipeline completo:
    1. Pré-processamento: Imputação + Encoding + Escalonamento
    2. Clustering: K-Means ou DBSCAN para identificar ritmos
    3. Detecção de Anomalias: Isolation Forest para eventos raros
    4. (Opcional) Tracking com MLFlow

    Args:
        laps_df: DataFrame com dados de voltas (raw ou pré-processado)
        analysis_type: Tipo de análise:
                      - 'clustering': Apenas clustering (K-Means)
                      - 'anomaly': Apenas detecção de anomalias
                      - 'all': Ambos (padrão)
        driver: Filtrar por piloto específico (opcional)
        enable_mlflow: Se True, tracka com MLFlow. None = lê de config.yaml (padrão)
        experiment_name: Nome do experimento MLFlow (auto-gerado se None)
        run_name: Nome do run MLFlow (auto-gerado se None)
        year: Ano da corrida (usado no nome do experimento e nos params do MLFlow)
        round_number: Número da rodada (idem)
        contamination_profile: Perfil de contaminação para Isolation Forest.
                               'clean' (3%), 'normal' (5%), 'chaotic' (10%).
                               Lido de config.yaml — não hardcoded.

    Returns:
        Dicionário com DataFrames de resultados:
        - 'laps_processed': Dados pré-processados
        - 'laps_clustered': Dados com clusters (se analysis_type='clustering' ou 'all')
        - 'laps_anomalies': Dados com anomalias (se analysis_type='anomaly' ou 'all')
        - 'summary': Sumário da análise
        - 'clustering_metrics': Métricas de clustering (se clustering executado)
        - 'anomaly_metrics': Métricas de anomaly detection (se executado)
        - 'laps_changepoints': Dados com regimes de degradação (stint_regime, is_cliff_lap)
        - 'tire_cliffs': Sumário de cliffs por (Driver, Stint)
        - 'tire_cliffs_summary': Sumário por piloto
        - 'changepoint_metrics': Métricas de change point detection
        - 'mlflow_run_id': ID do run MLFlow (se MLFlow habilitado)

    Example:
        >>> import pandas as pd
        >>>
        >>> # Carregar dados brutos
        >>> laps_df = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')
        >>>
        >>> # Executar análise completa COM tracking MLFlow
        >>> results = run_race_analysis(
        ...     laps_df,
        ...     analysis_type='all',
        ...     enable_mlflow=True,
        ...     experiment_name='F1_2025_Round_01',
        ...     run_name='Full_Analysis'
        ... )
        >>>
        >>> # Ver resultados
        >>> print(results['summary'])
        >>> print(f"MLFlow Run ID: {results.get('mlflow_run_id')}")
        >>> anomalies = results['laps_anomalies'][results['laps_anomalies']['is_anomaly']]
        >>> print(f"Anomalias detectadas: {len(anomalies)}")
    """
    # Resolver enable_mlflow: None = lê do config.yaml
    if enable_mlflow is None:
        from src.utils.config import get_config as _get_config
        enable_mlflow = _get_config().get_mlflow_enabled()

    # Resolver year/round a partir do DataFrame se não passados explicitamente
    _year = year or (int(laps_df['Year'].iloc[0]) if 'Year' in laps_df.columns else 0)
    _round = round_number or (int(laps_df['Round'].iloc[0]) if 'Round' in laps_df.columns else 0)

    # Setup MLFlow se habilitado
    if enable_mlflow:
        from src.utils.config import get_config as _get_config
        _cfg = _get_config()
        if experiment_name is None:
            prefix = _cfg.get_mlflow_experiment_prefix()
            experiment_name = f"{prefix}_{_year}_Round_{_round:02d}"
        setup_mlflow(experiment_name, tracking_uri=_cfg.get_mlflow_tracking_uri())

    # Filtrar por piloto se especificado
    if driver:
        laps_df = laps_df[laps_df['Driver'] == driver].copy()

    # Remover voltas com LapTime=0 (volta de formação, out-lap sem telemetria completa).
    # Essas voltas são "fáceis demais" para Isolation Forest e diluem a discriminação
    # de anomalias reais (erros de piloto, problemas mecânicos).
    if 'LapTime_seconds' in laps_df.columns:
        n_zero_laps = int((laps_df['LapTime_seconds'] == 0).sum())
        laps_df = laps_df[laps_df['LapTime_seconds'] > 0].copy()
    else:
        n_zero_laps = 0

    # Etapa 1: Pré-processamento
    # 1.1 Imputação de valores faltantes
    # Inclui features do pré-processamento: degradation_slope (taxa de desgaste do pneu)
    # e Position (posição na corrida) — disponíveis em laps_processed.parquet
    base_numeric = ['LapTime_seconds', 'Sector1Time_seconds', 'Sector2Time_seconds',
                    'Sector3Time_seconds', 'TyreLife']
    extra_numeric = ['degradation_slope', 'degradation_r_squared', 'Position']
    numeric_cols = base_numeric + [c for c in extra_numeric if c in laps_df.columns]
    laps_imputed = impute_missing_values(
        laps_df,
        numeric_columns=numeric_cols,
        strategy='median',
        use_knn=False
    )

    # 1.2 Engenharia de features (antes do escalonamento, usando valores reais)
    # Adiciona: Stint, LapTime_delta, TyreAge_normalized, Compound_ordinal
    laps_engineered = engineer_lap_features(laps_imputed)

    # 1.3 Filtrar voltas estruturais (pit stops, SC) nos dados BRUTOS, antes de escalar
    # O filtro usa LapTime_seconds real (ex: >1.5x a mediana do piloto ≈ pit stop laps)
    # Clustering usa apenas laps limpos; anomaly detection usa todos os laps
    laps_for_clustering_raw, laps_structural = filter_structural_laps(laps_engineered)
    n_structural = len(laps_structural)

    # 1.4 Encoding de variáveis categóricas em ambos os subsets
    engineered_numeric = ['LapTime_delta', 'TyreAge_normalized', 'Compound_ordinal', 'Stint']

    def _encode_and_scale(df_in):
        if 'Compound' in df_in.columns:
            df_enc = encode_categorical_variables(
                df_in, categorical_columns=['Compound'], drop_first=True
            )
        else:
            df_enc = df_in
        scale_cols = [col for col in numeric_cols + engineered_numeric if col in df_enc.columns]
        return scale_features(df_enc, numeric_columns=scale_cols, scaler_type='robust')

    # Dataset completo escalado (para anomaly detection — inclui pit/SC laps)
    laps_scaled = _encode_and_scale(laps_engineered)

    # Subset de clustering escalado (laps estruturais removidos antes do scale)
    laps_scaled_cluster = _encode_and_scale(laps_for_clustering_raw)

    results = {'laps_processed': laps_scaled}

    clustering_metrics_dict = None
    anomaly_metrics_dict = None

    # Etapa 2: Clustering (se solicitado)
    if analysis_type in ['clustering', 'all']:
        # Features para clustering — preferência por features engenheiradas + degradation_slope
        # degradation_slope: taxa de desgaste do pneu por volta (diferencia stints de push vs gestão)
        # TyreAge_normalized: fase do pneu dentro do stint (0=novo, 1=fim)
        # LapTime_delta: desvio do tempo relativo à mediana do stint (captura variação de ritmo)
        # Sector1Time_seconds: discrimina comportamento por setor (frenagem/aceleração)
        cluster_feature_cols = [
            col for col in [
                'LapTime_delta', 'TyreAge_normalized',
                'Sector1Time_seconds', 'degradation_slope',
            ]
            if col in laps_scaled_cluster.columns
        ]
        if not cluster_feature_cols:
            cluster_feature_cols = ['LapTime_seconds', 'Sector1Time_seconds']

        laps_for_clustering = laps_scaled_cluster  # já filtrado e escalado

        has_driver_col = 'Driver' in laps_for_clustering.columns

        # 2.1 K-Means por piloto com k=3 FIXO (prior físico F1: push | base | degraded)
        # k=3 não é limitação técnica — é escolha de domínio.
        # O downstream (LLM/Agno) precisa que cluster_semantic seja consistente entre
        # pilotos e corridas. find_optimal_k() existe como ferramenta de pesquisa
        # mas nunca deve ser chamada aqui. Ver: config.yaml > ml.clustering
        laps_clustered = cluster_laps_kmeans(
            laps_for_clustering,
            feature_columns=cluster_feature_cols,
            n_clusters=3,
            group_by='Driver' if has_driver_col else None,
        )

        # Normalizar semântica: cluster 0 = push (menor LapTime_delta)
        #                        cluster 1 = base pace
        #                        cluster 2 = degraded/blocked (maior LapTime_delta)
        # Garante labels consistentes entre pilotos para geração de eventos downstream.
        if 'LapTime_delta' in laps_clustered.columns:
            laps_clustered = normalize_cluster_semantics(
                laps_clustered,
                driver_column='Driver' if has_driver_col else '__none__',
            )

        results['laps_clustered'] = laps_clustered

        # Métricas por piloto (pooled não fazem sentido — espaços absolutos de tempo diferentes)
        per_driver_clustering = calculate_per_driver_clustering_metrics(
            laps_clustered,
            feature_columns=cluster_feature_cols,
            labels_column='cluster_label',
            driver_column='Driver' if has_driver_col else '__none__',
        )
        results['per_driver_clustering_metrics'] = per_driver_clustering

        # Dict de métricas: somente agregados por piloto + n_structural_filtered
        clustering_metrics_dict = {'n_structural_filtered': n_structural}
        for k, v in per_driver_clustering.items():
            if isinstance(v, (int, float)) and v is not None:
                clustering_metrics_dict[k] = v

        # Métricas individuais por piloto (para MLFlow)
        for driver_m in per_driver_clustering.get('per_driver', []):
            d = driver_m['driver']
            clustering_metrics_dict[f'driver_{d}_silhouette'] = driver_m['silhouette_score']
            clustering_metrics_dict[f'driver_{d}_davies_bouldin'] = driver_m['davies_bouldin_score']

        results['clustering_metrics'] = pd.DataFrame([{
            k: v for k, v in clustering_metrics_dict.items()
            if not isinstance(v, list)
        }])

        # Estatísticas por cluster (com cluster_semantic e cluster_size_pct por piloto)
        cluster_stats = calculate_cluster_statistics(
            laps_clustered,
            cluster_column='cluster_label',
            feature_columns=cluster_feature_cols,
            driver_column='Driver' if has_driver_col else None,
        )
        results['cluster_statistics'] = cluster_stats

        # 2.2 DBSCAN como análise complementar
        laps_dbscan = cluster_laps_dbscan(
            laps_for_clustering,
            feature_columns=cluster_feature_cols,
            group_by='Driver' if has_driver_col else None,
        )
        results['laps_dbscan'] = laps_dbscan

        per_driver_dbscan = calculate_per_driver_clustering_metrics(
            laps_dbscan[laps_dbscan['cluster_label'] != -1],
            feature_columns=cluster_feature_cols,
            labels_column='cluster_label',
            driver_column='Driver' if has_driver_col else '__none__',
        )
        n_dbscan_noise = int(laps_dbscan['is_noise'].sum())
        dbscan_metrics_dict = {
            'dbscan_n_noise': n_dbscan_noise,
            'dbscan_noise_rate': round(100 * n_dbscan_noise / max(len(laps_dbscan), 1), 2),
        }
        for k, v in per_driver_dbscan.items():
            if isinstance(v, (int, float)) and v is not None:
                dbscan_metrics_dict[f'dbscan_{k}'] = v

        clustering_metrics_dict.update(dbscan_metrics_dict)
        results['dbscan_metrics'] = pd.DataFrame([dbscan_metrics_dict])

    # Etapa 3: Detecção de Anomalias (se solicitado)
    # Roda em TODOS os dados (incluindo pit/SC laps — queremos detectá-los)
    if analysis_type in ['anomaly', 'all']:
        # LapTime_delta: desvio relativo à mediana do piloto no stint — mais discriminativo
        #   que LapTime_seconds absoluto porque captura desvios do padrão do próprio piloto.
        #   Produz faixa de anomaly_score mais ampla (spikes mais claros para o Ruptures).
        # degradation_slope: taxa de desgaste anormal indica falha mecânica ou erro
        # Sector1/2/3Time_seconds: localiza em qual setor a anomalia ocorreu
        # Position: quedas súbitas de posição indicam problemas (colisão, breakdown)
        anomaly_feature_cols = [
            col for col in [
                'LapTime_delta',
                'Sector1Time_seconds', 'Sector2Time_seconds', 'Sector3Time_seconds',
                'degradation_slope', 'Position',
            ]
            if col in laps_scaled.columns
        ]
        # contamination lido do config via perfil (clean/normal/chaotic) — não hardcoded
        from src.utils.config import get_config as _get_config
        _contamination = _get_config().get_contamination(contamination_profile)
        laps_anomalies = detect_anomalies_isolation_forest(
            laps_scaled,
            feature_columns=anomaly_feature_cols,
            contamination=_contamination,
            group_by='Driver' if 'Driver' in laps_scaled.columns else None,
            return_scores=True
        )
        results['laps_anomalies'] = laps_anomalies

        predictions = laps_anomalies['is_anomaly'].map({True: -1, False: 1}).values
        scores = (
            laps_anomalies['anomaly_score'].values
            if 'anomaly_score' in laps_anomalies.columns else None
        )
        anomaly_metrics_dict = calculate_anomaly_metrics(predictions, scores)
        results['anomaly_metrics'] = pd.DataFrame([anomaly_metrics_dict])

    # Etapa 4: Detecção de Tire Cliffs (Ruptures/PELT)
    # Roda apenas no modo 'all' e requer a saída do anomaly detection
    changepoint_metrics_dict = None
    if analysis_type == 'all' and 'laps_anomalies' in results:
        laps_for_cp = results['laps_anomalies'].copy()

        # LapTime_delta já deve existir (calculado em engineer_lap_features)
        # mas garantimos como fallback
        if 'LapTime_delta' not in laps_for_cp.columns and 'LapTime_seconds' in laps_for_cp.columns:
            laps_for_cp['LapTime_delta'] = laps_for_cp.groupby('Driver')['LapTime_seconds'].transform(
                lambda x: x - x.median()
            )

        required_cp = ['is_anomaly', 'Stint', 'LapNumber', 'LapTime_delta']
        if all(c in laps_for_cp.columns for c in required_cp):
            laps_changepoints, changepoints_df = detect_tire_changepoints(laps_for_cp)
            cliffs_summary = summarize_cliffs(changepoints_df)
            changepoint_metrics_dict = calculate_changepoint_metrics(changepoints_df)

            results['laps_changepoints'] = laps_changepoints
            results['tire_cliffs'] = changepoints_df
            results['tire_cliffs_summary'] = cliffs_summary
            results['changepoint_metrics'] = pd.DataFrame([changepoint_metrics_dict])

    # Criar sumário executivo
    summary = {
        'total_laps': len(laps_scaled),
        'n_zero_laps_removed': n_zero_laps,
        'drivers': laps_scaled['Driver'].nunique() if 'Driver' in laps_scaled.columns else 1,
        'analysis_type': analysis_type,
    }

    if 'laps_clustered' in results:
        summary['laps_for_clustering'] = len(results['laps_clustered'])
        summary['n_structural_filtered'] = clustering_metrics_dict.get('n_structural_filtered', 0)
        summary['silhouette_mean_per_driver'] = per_driver_clustering.get('silhouette_mean')
        summary['davies_bouldin_mean_per_driver'] = per_driver_clustering.get('davies_bouldin_mean')

    if 'laps_anomalies' in results:
        n_anomalies = int(results['laps_anomalies']['is_anomaly'].sum())
        anomaly_rate = 100 * n_anomalies / len(laps_scaled)
        summary['n_anomalies'] = n_anomalies
        summary['anomaly_rate'] = f"{anomaly_rate:.2f}%"

    results['summary'] = pd.DataFrame([summary])

    # Tracking com MLFlow (se habilitado)
    if enable_mlflow:
        if run_name is None:
            run_name = f"Pipeline_{_year}_R{_round:02d}"
            if driver:
                run_name += f"_{driver}"

        # Artefatos: DataFrames principais salvos como CSV
        mlflow_artifacts = {}
        if 'laps_clustered' in results:
            mlflow_artifacts['laps_clustered.csv'] = results['laps_clustered']
        if 'laps_anomalies' in results:
            mlflow_artifacts['laps_anomalies.csv'] = results['laps_anomalies']
        if 'cluster_statistics' in results:
            mlflow_artifacts['cluster_statistics.csv'] = results['cluster_statistics']
        if 'per_driver_clustering_metrics' in results:
            per_driver_list = results['per_driver_clustering_metrics'].get('per_driver', [])
            if per_driver_list:
                mlflow_artifacts['per_driver_metrics.csv'] = pd.DataFrame(per_driver_list)
        if 'laps_changepoints' in results:
            mlflow_artifacts['laps_changepoints.csv'] = results['laps_changepoints']
        if 'tire_cliffs' in results:
            mlflow_artifacts['tire_cliffs.csv'] = results['tire_cliffs']
        if 'tire_cliffs_summary' in results:
            mlflow_artifacts['tire_cliffs_summary.csv'] = results['tire_cliffs_summary']

        mlflow_run_id = track_pipeline_run(
            run_name=run_name,
            year=_year,
            round_number=_round,
            clustering_results=clustering_metrics_dict,
            anomaly_results=anomaly_metrics_dict,
            changepoint_results=changepoint_metrics_dict,
            params={
                'analysis_type': analysis_type,
                'driver': driver if driver else 'all',
                'scaler_type': 'robust',
                'contamination_profile': contamination_profile,
                'contamination': _contamination if 'laps_anomalies' in results else None,
                'kmeans_k': 3,
                'structural_filter_threshold': 1.5,
                'cluster_semantics': 'size_first_then_delta',
                'n_zero_laps_removed': n_zero_laps,
                'cluster_features': ','.join(cluster_feature_cols) if analysis_type in ['clustering', 'all'] else '',
                'anomaly_features': ','.join(anomaly_feature_cols) if analysis_type in ['anomaly', 'all'] else '',
            },
            tags={
                'driver': driver if driver else 'all',
                'features': 'engineered',
            },
            artifacts=mlflow_artifacts,
        )

        results['mlflow_run_id'] = mlflow_run_id

    return results
