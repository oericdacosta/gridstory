"""
Pipeline completo de Machine Learning para análise de corridas.

Integra pré-processamento (imputação, encoding, escalonamento) com algoritmos
de ML (clustering, detecção de anomalias) em um fluxo unificado.
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


def run_race_analysis(
    laps_df: pd.DataFrame,
    analysis_type: str = "all",
    driver: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Executa análise completa de ML em dados de voltas de uma corrida.

    Pipeline completo:
    1. Pré-processamento: Imputação + Encoding + Escalonamento
    2. Clustering: K-Means ou DBSCAN para identificar ritmos
    3. Detecção de Anomalias: Isolation Forest para eventos raros

    Args:
        laps_df: DataFrame com dados de voltas (raw ou pré-processado)
        analysis_type: Tipo de análise:
                      - 'clustering': Apenas clustering (K-Means)
                      - 'anomaly': Apenas detecção de anomalias
                      - 'all': Ambos (padrão)
        driver: Filtrar por piloto específico (opcional)

    Returns:
        Dicionário com DataFrames de resultados:
        - 'laps_processed': Dados pré-processados
        - 'laps_clustered': Dados com clusters (se analysis_type='clustering' ou 'all')
        - 'laps_anomalies': Dados com anomalias (se analysis_type='anomaly' ou 'all')
        - 'summary': Sumário da análise

    Example:
        >>> import pandas as pd
        >>> 
        >>> # Carregar dados brutos
        >>> laps_df = pd.read_parquet('data/raw/races/2025/round_01/laps.parquet')
        >>> 
        >>> # Executar análise completa
        >>> results = run_race_analysis(laps_df, analysis_type='all')
        >>> 
        >>> # Ver resultados
        >>> print(results['summary'])
        >>> anomalies = results['laps_anomalies'][results['laps_anomalies']['is_anomaly']]
        >>> print(f"Anomalias detectadas: {len(anomalies)}")
    """
    # Filtrar por piloto se especificado
    if driver:
        laps_df = laps_df[laps_df['Driver'] == driver].copy()

    # Etapa 1: Pré-processamento
    # 1.1 Imputação de valores faltantes
    numeric_cols = ['LapTime_seconds', 'Sector1Time_seconds', 'Sector2Time_seconds', 
                   'Sector3Time_seconds', 'TyreLife']
    laps_imputed = impute_missing_values(
        laps_df,
        numeric_columns=numeric_cols,
        strategy='median',
        use_knn=False
    )

    # 1.2 Encoding de variáveis categóricas
    if 'Compound' in laps_imputed.columns:
        laps_encoded = encode_categorical_variables(
            laps_imputed,
            categorical_columns=['Compound'],
            drop_first=True
        )
    else:
        laps_encoded = laps_imputed

    # 1.3 Escalonamento
    # Selecionar apenas colunas numéricas relevantes (excluir binárias do encoding)
    scale_cols = [col for col in numeric_cols if col in laps_encoded.columns]
    laps_scaled = scale_features(
        laps_encoded,
        numeric_columns=scale_cols,
        scaler_type='robust'
    )

    results = {'laps_processed': laps_scaled}

    # Etapa 2: Clustering (se solicitado)
    if analysis_type in ['clustering', 'all']:
        feature_cols = ['LapTime_seconds', 'Sector1Time_seconds']
        laps_clustered = cluster_laps_kmeans(
            laps_scaled,
            feature_columns=feature_cols,
            n_clusters=None,  # Auto-detect
            group_by='Driver' if 'Driver' in laps_scaled.columns else None
        )
        results['laps_clustered'] = laps_clustered

    # Etapa 3: Detecção de Anomalias (se solicitado)
    if analysis_type in ['anomaly', 'all']:
        feature_cols = ['LapTime_seconds', 'Sector1Time_seconds']
        laps_anomalies = detect_anomalies_isolation_forest(
            laps_scaled,
            feature_columns=feature_cols,
            contamination=0.05,
            group_by='Driver' if 'Driver' in laps_scaled.columns else None,
            return_scores=True
        )
        results['laps_anomalies'] = laps_anomalies

    # Criar sumário
    summary = {
        'total_laps': len(laps_scaled),
        'drivers': laps_scaled['Driver'].nunique() if 'Driver' in laps_scaled.columns else 1,
        'analysis_type': analysis_type,
    }

    if 'laps_clustered' in results:
        n_clusters = results['laps_clustered']['cluster_label'].nunique()
        summary['n_clusters'] = n_clusters

    if 'laps_anomalies' in results:
        n_anomalies = results['laps_anomalies']['is_anomaly'].sum()
        anomaly_rate = 100 * n_anomalies / len(laps_scaled)
        summary['n_anomalies'] = n_anomalies
        summary['anomaly_rate'] = f"{anomaly_rate:.2f}%"

    results['summary'] = pd.DataFrame([summary])

    return results
