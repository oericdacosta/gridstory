"""
Features estatísticas e análise de degradação.

Calcula métricas estatísticas, identifica outliers e analisa degradação de pneus
usando scipy.stats para criar features para modelos de ML.
"""

import numpy as np
import pandas as pd
from scipy import stats


def calculate_statistical_features(
    df: pd.DataFrame,
    value_column: str = "LapTime",
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calcula features estatísticas para tempos de volta ou outras métricas.

    Computa Z-scores, marca outliers e adiciona estatísticas descritivas
    como features para processamento ML downstream.

    Args:
        df: DataFrame de entrada com dados de voltas
        value_column: Coluna para analisar (padrão: 'LapTime')
        group_by: Colunas para agrupar (ex: ['Driver', 'Compound'])
                 Se None, analisa todo o conjunto de dados como um grupo

    Returns:
        DataFrame com colunas de features estatísticas adicionadas:
        - z_score: Score padronizado relativo à média do grupo
        - is_outlier: Flag binária para outliers estatísticos (|z| > 3)
        - group_mean: Valor médio dentro do grupo
        - group_std: Desvio padrão dentro do grupo

    Example:
        >>> laps_df = pd.DataFrame({
        ...     'Driver': ['VER', 'VER', 'VER'],
        ...     'Compound': ['SOFT', 'SOFT', 'SOFT'],
        ...     'LapTime': [90.5, 91.2, 95.8]  # Última é outlier
        ... })
        >>> enriched = calculate_statistical_features(
        ...     laps_df,
        ...     value_column='LapTime',
        ...     group_by=['Driver', 'Compound']
        ... )
        >>> print(enriched['is_outlier'])
    """
    df = df.copy()

    if value_column not in df.columns:
        raise ValueError(f"Coluna '{value_column}' não encontrada no DataFrame")

    # Converter para numérico se necessário
    if df[value_column].dtype == "object":
        # Lidar com timedelta se for LapTime
        if "LapTime" in value_column or "Time" in value_column:
            df[f"{value_column}_converted"] = pd.to_timedelta(
                df[value_column]
            ).dt.total_seconds()
            value_column = f"{value_column}_converted"
        else:
            df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    # Se já for numérico, mas for timedelta64, converter para segundos
    elif pd.api.types.is_timedelta64_dtype(df[value_column]):
        df[f"{value_column}_converted"] = df[value_column].dt.total_seconds()
        value_column = f"{value_column}_converted"

    if group_by:
        # Calcular dentro dos grupos
        df["z_score"] = df.groupby(group_by)[value_column].transform(
            lambda x: stats.zscore(x, nan_policy="omit")
        )
        df["group_mean"] = df.groupby(group_by)[value_column].transform("mean")
        df["group_std"] = df.groupby(group_by)[value_column].transform("std")
    else:
        # Calcular para todo o conjunto de dados
        valid_values = df[value_column].dropna()
        if len(valid_values) > 0:
            z_scores = stats.zscore(valid_values)
            df.loc[valid_values.index, "z_score"] = z_scores
        else:
            df["z_score"] = np.nan

        df["group_mean"] = df[value_column].mean()
        df["group_std"] = df[value_column].std()

    # Marcar outliers (|z| > 3 é threshold comum)
    df["is_outlier"] = np.abs(df["z_score"]) > 3
    df["is_outlier"] = df["is_outlier"].fillna(False)

    return df


def calculate_degradation_rate(
    df: pd.DataFrame,
    lap_column: str = "LapNumber",
    time_column: str | None = None,
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calcula taxa de degradação de pneu usando regressão linear.

    Computa a inclinação do tempo de volta vs número da volta para quantificar
    quão rapidamente os pneus estão degradando dentro de um stint.

    Args:
        df: DataFrame de entrada com dados de voltas
        lap_column: Coluna contendo números de volta
        time_column: Coluna contendo tempos de volta
        group_by: Colunas para agrupar (ex: ['Driver', 'Stint'])

    Returns:
        DataFrame com features de degradação adicionadas:
        - degradation_slope: Taxa de aumento do tempo de volta (segundos por volta)
        - degradation_r_squared: Valor R² do ajuste linear (qualidade do ajuste)
        - degradation_intercept: Intercepto Y (tempo estimado da primeira volta)

    Example:
        >>> stint_data = pd.DataFrame({
        ...     'Driver': ['VER'] * 10,
        ...     'Stint': [1] * 10,
        ...     'LapNumber': range(1, 11),
        ...     'LapTime': [90.0, 90.2, 90.5, 90.7, 91.0, 91.3, 91.5, 91.8, 92.0, 92.3]
        ... })
        >>> with_degradation = calculate_degradation_rate(
        ...     stint_data,
        ...     group_by=['Driver', 'Stint']
        ... )
        >>> print(with_degradation['degradation_slope'].iloc[0])  # ~0.25 seg/volta
    """
    df = df.copy()

    # Auto-detectar coluna de tempo se não fornecida
    if time_column is None:
        if 'LapTime_seconds' in df.columns:
            time_column = 'LapTime_seconds'
        elif 'LapTime' in df.columns:
            time_column = 'LapTime'
        else:
            raise ValueError("Nenhuma coluna de tempo encontrada (LapTime ou LapTime_seconds)")

    # Converter tempo para segundos se necessário
    if pd.api.types.is_timedelta64_dtype(df[time_column]):
        time_col_seconds = f"{time_column}_converted"
        df[time_col_seconds] = df[time_column].dt.total_seconds()
    elif df[time_column].dtype == "object":
        time_col_seconds = f"{time_column}_converted"
        df[time_col_seconds] = pd.to_timedelta(df[time_column]).dt.total_seconds()
    else:
        # Já está em segundos
        time_col_seconds = time_column

    def _calculate_regression(group):
        """Calcular regressão linear para um grupo."""
        # Remover valores NaN
        valid_mask = group[lap_column].notna() & group[time_col_seconds].notna()
        x = group.loc[valid_mask, lap_column].values
        y = group.loc[valid_mask, time_col_seconds].values

        if len(x) < 2:
            # Pontos insuficientes para regressão
            return pd.DataFrame(
                {
                    "degradation_slope": [np.nan] * len(group),
                    "degradation_r_squared": [np.nan] * len(group),
                    "degradation_intercept": [np.nan] * len(group),
                },
                index=group.index,
            )

        # Realizar regressão linear
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Adicionar resultados a todas as linhas do grupo
        result = pd.DataFrame(
            {
                "degradation_slope": [slope] * len(group),
                "degradation_r_squared": [r_value**2] * len(group),
                "degradation_intercept": [intercept] * len(group),
            },
            index=group.index,
        )

        return result

    if group_by:
        # Calcular para cada grupo
        degradation_features = df.groupby(group_by, group_keys=False).apply(
            _calculate_regression, include_groups=False
        )
    else:
        # Calcular para todo o conjunto de dados
        degradation_features = _calculate_regression(df)

    # Mesclar de volta ao dataframe original
    df = pd.concat([df, degradation_features], axis=1)

    return df


def calculate_descriptive_statistics(
    values: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Calcula estatísticas descritivas abrangentes para um conjunto de dados.

    Args:
        values: Array ou Series de valores numéricos

    Returns:
        Dicionário contendo:
        - nobs: Número de observações
        - mean: Valor médio
        - variance: Variância
        - skewness: Assimetria (assimetria da distribuição)
        - kurtosis: Curtose (caudas da distribuição)
        - min: Valor mínimo
        - max: Valor máximo

    Example:
        >>> lap_times = np.array([90.5, 90.7, 90.9, 91.1, 91.3])
        >>> stats = calculate_descriptive_statistics(lap_times)
        >>> print(f"Média: {stats['mean']:.2f}s, Std: {stats['variance']**0.5:.2f}s")
    """
    if isinstance(values, pd.Series):
        values = values.values

    # Remover valores NaN
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return {
            "nobs": 0,
            "mean": np.nan,
            "variance": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    # Usar scipy.stats.describe para estatísticas abrangentes
    description = stats.describe(values)

    return {
        "nobs": description.nobs,
        "mean": description.mean,
        "variance": description.variance,
        "skewness": description.skewness,
        "kurtosis": description.kurtosis,
        "min": description.minmax[0],
        "max": description.minmax[1],
    }


def enrich_dataframe_with_stats(
    df: pd.DataFrame,
    value_column: str = "LapTime",
    group_by: list[str] | None = None,
    include_degradation: bool = True,
) -> pd.DataFrame:
    """
    Pipeline abrangente de engenharia de features combinando todas as features estatísticas.

    Este é o ponto de entrada principal para preparar dados para modelos de ML. Combina:
    - Z-scores e detecção de outliers
    - Cálculo de taxa de degradação
    - Estatísticas descritivas

    Args:
        df: DataFrame de entrada
        value_column: Coluna para analisar
        group_by: Colunas de agrupamento
        include_degradation: Se deve calcular taxa de degradação

    Returns:
        DataFrame enriquecido pronto para Scikit-learn e Ruptures

    Example:
        >>> session = fastf1.get_session(2024, 'Monaco', 'R')
        >>> session.load()
        >>> laps = session.laps
        >>> enriched = enrich_dataframe_with_stats(
        ...     laps,
        ...     value_column='LapTime',
        ...     group_by=['Driver', 'Compound'],
        ...     include_degradation=True
        ... )
        >>> # Agora pronto para processamento ML
        >>> clean_laps = enriched[~enriched['is_outlier']]
    """
    # Etapa 1: Calcular Z-scores e outliers
    df = calculate_statistical_features(
        df, value_column=value_column, group_by=group_by
    )

    # Etapa 2: Calcular taxa de degradação se solicitado
    if include_degradation:
        # Usar a mesma coluna que foi usada para features estatísticas
        df = calculate_degradation_rate(df, time_column=value_column, group_by=group_by)

    # Etapa 3: Adicionar estatísticas descritivas por grupo
    if group_by:
        # Usar a coluna que foi realmente processada (pode ter sido convertida)
        # Procurar a coluna original ou convertida
        actual_column = value_column
        if f"{value_column}_converted" in df.columns:
            actual_column = f"{value_column}_converted"

        for name, group in df.groupby(group_by):
            stats_dict = calculate_descriptive_statistics(group[actual_column])

            # Adicionar como colunas com prefixo
            for stat_name, stat_value in stats_dict.items():
                df.loc[group.index, f"group_{stat_name}"] = stat_value

    return df
