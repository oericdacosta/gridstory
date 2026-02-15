"""
Detecção de anomalias usando Isolation Forest.

Detecta eventos pontuais raros (erros, falhas mecânicas, voltas excepcionalmente
rápidas) para alimentar o sistema de geração de notícias.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.utils.config import get_config


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    feature_columns: list[str],
    contamination: float | None = None,
    contamination_profile: str = "normal",
    group_by: str | None = None,
    return_scores: bool = False,
) -> pd.DataFrame:
    """
    Detecta voltas anômalas usando Isolation Forest.

    O Isolation Forest isola observações selecionando aleatoriamente uma característica
    e um valor de divisão. Anomalias são isoladas mais rapidamente (caminhos mais curtos
    na árvore de decisão) do que voltas normais.

    Args:
        df: DataFrame com dados de voltas (já pré-processado e escalonado)
        feature_columns: Colunas para usar na detecção (ex: ['LapTime_seconds', 'Sector1Time_seconds'])
        contamination: Proporção esperada de anomalias no dataset (padrão: carrega de config.yaml)
                      Se especificado, sobrescreve contamination_profile
        contamination_profile: Perfil de corrida para contamination - "clean", "normal", ou "chaotic"
                             Ignorado se contamination for especificado explicitamente
        group_by: Coluna para agrupar antes da detecção (ex: 'Driver' para analisar cada piloto)
        return_scores: Se True, retorna também o anomaly score (quanto mais negativo, mais anômalo)

    Returns:
        DataFrame com colunas adicionais:
        - is_anomaly: Flag binária indicando anomalia (True) ou normal (False)
        - anomaly_score: Score de anomalia (apenas se return_scores=True)
                        Valores negativos = anômalos, positivos = normais

    Rationale:
        - Isolation Forest é eficiente e escalável
        - Não assume distribuição dos dados
        - Bom para detectar outliers multidimensionais (não apenas em uma variável)
        - Contamination deve ser ajustado baseado no conhecimento do domínio
          (ex: 5% das voltas podem ser anômalas devido a tráfego, erros, etc.)

    Example:
        >>> # Pré-processar e escalonar
        >>> from src.preprocessing.feature_engineering import enrich_dataframe_with_stats, scale_features
        >>> laps_processed = enrich_dataframe_with_stats(laps_df, ...)
        >>> laps_scaled = scale_features(laps_processed, ['LapTime_seconds', 'Sector1Time_seconds'])
        >>>
        >>> # Detectar anomalias
        >>> laps_anomalies = detect_anomalies_isolation_forest(
        ...     laps_scaled,
        ...     feature_columns=['LapTime_seconds', 'Sector1Time_seconds'],
        ...     group_by='Driver',
        ...     return_scores=True
        ... )
        >>>
        >>> # Analisar anomalias
        >>> anomalies = laps_anomalies[laps_anomalies['is_anomaly']]
        >>> print(f"Anomalias detectadas: {len(anomalies)}")
        >>> print(anomalies[['Driver', 'LapNumber', 'LapTime_seconds', 'anomaly_score']])
    """
    config = get_config()

    if contamination is None:
        contamination = config.get_contamination(profile=contamination_profile)

    random_state = config.get_random_state()
    n_estimators = config.get_n_estimators()
    df = df.copy()

    # Verificar se features existem
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas não encontradas: {missing_cols}")

    # Se agrupar, processar cada grupo separadamente
    if group_by:
        # Inicializar colunas de resultado
        df['is_anomaly'] = False
        if return_scores:
            df['anomaly_score'] = 0.0

        for group_name, group_df in df.groupby(group_by):
            if len(group_df) < 10:
                # Grupo muito pequeno, não detectar anomalias
                # (Isolation Forest precisa de dados suficientes)
                continue

            X = group_df[feature_columns].values

            # Criar e treinar Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=n_estimators
            )

            # Predição: 1 = normal, -1 = anomalia
            predictions = iso_forest.fit_predict(X)

            # Converter para boolean (True = anomalia)
            is_anomaly = predictions == -1

            # Atribuir resultados
            df.loc[group_df.index, 'is_anomaly'] = is_anomaly

            if return_scores:
                # Scores: valores negativos = anômalos
                scores = iso_forest.score_samples(X)
                df.loc[group_df.index, 'anomaly_score'] = scores

    else:
        # Processar todo o dataset
        X = df[feature_columns].values

        # Criar e treinar Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

        # Predição
        predictions = iso_forest.fit_predict(X)
        df['is_anomaly'] = predictions == -1

        if return_scores:
            df['anomaly_score'] = iso_forest.score_samples(X)

    return df


def summarize_anomalies(
    df: pd.DataFrame,
    group_by: str | None = None,
) -> pd.DataFrame:
    """
    Sumariza anomalias detectadas para análise e geração de eventos.

    Args:
        df: DataFrame com coluna 'is_anomaly' (resultado de detect_anomalies_isolation_forest)
        group_by: Coluna para agrupar sumário (ex: 'Driver')

    Returns:
        DataFrame com estatísticas de anomalias:
        - total_laps: Total de voltas
        - anomalies_count: Número de anomalias
        - anomaly_rate: Taxa de anomalias (%)
        - anomaly_laps: Lista de números de voltas anômalas

    Example:
        >>> laps_anomalies = detect_anomalies_isolation_forest(...)
        >>> summary = summarize_anomalies(laps_anomalies, group_by='Driver')
        >>> print(summary)
    """
    if 'is_anomaly' not in df.columns:
        raise ValueError("DataFrame deve ter coluna 'is_anomaly'")

    if group_by:
        # Sumário por grupo
        summary = df.groupby(group_by).agg({
            'is_anomaly': [
                ('total_laps', 'count'),
                ('anomalies_count', 'sum'),
                ('anomaly_rate', lambda x: 100 * x.sum() / len(x))
            ]
        })
        summary.columns = ['total_laps', 'anomalies_count', 'anomaly_rate']
        summary = summary.reset_index()

        # Adicionar lista de voltas anômalas
        if 'LapNumber' in df.columns:
            anomaly_laps = df[df['is_anomaly']].groupby(group_by)['LapNumber'].apply(list)
            summary = summary.merge(
                anomaly_laps.rename('anomaly_laps'),
                left_on=group_by,
                right_index=True,
                how='left'
            )
            summary['anomaly_laps'] = summary['anomaly_laps'].apply(
                lambda x: x if isinstance(x, list) else []
            )

    else:
        # Sumário global
        total = len(df)
        anomalies = df['is_anomaly'].sum()
        summary = pd.DataFrame({
            'total_laps': [total],
            'anomalies_count': [anomalies],
            'anomaly_rate': [100 * anomalies / total if total > 0 else 0]
        })

        if 'LapNumber' in df.columns:
            summary['anomaly_laps'] = [df[df['is_anomaly']]['LapNumber'].tolist()]

    return summary
