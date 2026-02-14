"""
Engenharia de features estatísticas usando scipy.stats.

Calcula métricas estatísticas e identifica outliers para criar
features para modelos de ML e filtrar anomalias óbvias.
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


def preprocess_race_control(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pré-processa dados de controle de corrida (safety car, bandeiras, penalidades).

    Transforma mensagens de texto em features estruturadas para análise ML:
    - Normaliza timestamps
    - Cria indicadores binários para eventos importantes
    - Categoriza tipos de mensagens
    - Extrai informações relevantes de cada categoria

    Args:
        df: DataFrame com race control messages (colunas: Time, Category, Message, etc.)

    Returns:
        DataFrame enriquecido com features categóricas e indicadores binários:
        - time_seconds: Tempo normalizado em segundos
        - is_safety_car: Flag para safety car (SC/VSC)
        - is_flag: Flag para bandeiras (amarela, vermelha, etc.)
        - is_penalty: Flag para penalidades
        - category_encoded: Categoria codificada numericamente
        - event_severity: Severidade do evento (0=info, 1=warning, 2=critical)

    Example:
        >>> race_control = pd.read_parquet('data/raw/races/2025/round_01/race_control.parquet')
        >>> processed = preprocess_race_control(race_control)
        >>> safety_car_events = processed[processed['is_safety_car']]
    """
    df = df.copy()

    # Normalizar timestamps para segundos se necessário
    if 'Time_seconds' not in df.columns and 'Time' in df.columns:
        if pd.api.types.is_timedelta64_dtype(df['Time']):
            df['time_seconds'] = df['Time'].dt.total_seconds()
        elif hasattr(df['Time'].iloc[0], 'total_seconds'):
            df['time_seconds'] = df['Time'].apply(lambda x: x.total_seconds())
        else:
            df['time_seconds'] = df['Time']
    else:
        df['time_seconds'] = df.get('Time_seconds', df.get('Time', 0))

    # Extrair categoria se existir
    if 'Category' in df.columns:
        df['category'] = df['Category'].fillna('UNKNOWN')
    else:
        df['category'] = 'UNKNOWN'

    # Extrair mensagem
    if 'Message' in df.columns:
        df['message'] = df['Message'].fillna('')
    else:
        df['message'] = ''

    # Criar indicadores binários para eventos importantes
    df['is_safety_car'] = df.apply(
        lambda row: any(keyword in str(row.get('message', '')).upper() + str(row.get('category', '')).upper()
                       for keyword in ['SAFETY CAR', 'VIRTUAL SAFETY CAR', 'VSC', 'SC DEPLOYED']),
        axis=1
    )

    df['is_flag'] = df.apply(
        lambda row: any(keyword in str(row.get('message', '')).upper() + str(row.get('category', '')).upper()
                       for keyword in ['YELLOW FLAG', 'RED FLAG', 'GREEN FLAG', 'BLUE FLAG', 'FLAG']),
        axis=1
    )

    df['is_penalty'] = df.apply(
        lambda row: any(keyword in str(row.get('message', '')).upper() + str(row.get('category', '')).upper()
                       for keyword in ['PENALTY', 'PENALISED', 'TIME PENALTY', 'DRIVE THROUGH', 'STOP AND GO']),
        axis=1
    )

    df['is_drs'] = df.apply(
        lambda row: 'DRS' in str(row.get('message', '')).upper() + str(row.get('category', '')).upper(),
        axis=1
    )

    # Codificar categorias
    category_map = {
        'SafetyCar': 1,
        'Flag': 2,
        'Drs': 3,
        'CarEvent': 4,
        'Other': 0,
        'UNKNOWN': 0
    }
    df['category_encoded'] = df['category'].map(category_map).fillna(0).astype(int)

    # Calcular severidade do evento (0=info, 1=warning, 2=critical)
    def calculate_severity(row):
        msg_upper = str(row.get('message', '')).upper() + str(row.get('category', '')).upper()

        # Eventos críticos
        if any(keyword in msg_upper for keyword in ['RED FLAG', 'SAFETY CAR DEPLOYED', 'RACE SUSPENDED']):
            return 2

        # Eventos de atenção
        if any(keyword in msg_upper for keyword in ['YELLOW FLAG', 'VSC', 'PENALTY', 'INVESTIGATION']):
            return 1

        # Informações
        return 0

    df['event_severity'] = df.apply(calculate_severity, axis=1)

    # Ordenar por tempo
    df = df.sort_values('time_seconds').reset_index(drop=True)

    return df


def preprocess_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pré-processa dados meteorológicos (temperatura, chuva, vento).

    Transforma dados brutos de clima em features úteis para análise:
    - Normaliza timestamps
    - Interpola valores faltantes
    - Calcula tendências de temperatura
    - Detecta mudanças bruscas de condições

    Args:
        df: DataFrame com weather data (colunas: Time, AirTemp, TrackTemp, Rainfall, etc.)

    Returns:
        DataFrame enriquecido com features meteorológicas:
        - time_seconds: Tempo normalizado
        - air_temp_normalized: Temperatura do ar normalizada
        - track_temp_normalized: Temperatura da pista normalizada
        - temp_delta: Diferença pista-ar
        - rainfall_indicator: Indicador binário de chuva
        - temp_trend: Tendência de temperatura (subindo/descendo)
        - weather_change: Flag para mudança brusca de condições

    Example:
        >>> weather = pd.read_parquet('data/raw/races/2025/round_01/weather.parquet')
        >>> processed = preprocess_weather(weather)
        >>> rain_periods = processed[processed['rainfall_indicator']]
    """
    df = df.copy()

    # Normalizar timestamps
    if 'Time_seconds' not in df.columns and 'Time' in df.columns:
        if pd.api.types.is_timedelta64_dtype(df['Time']):
            df['time_seconds'] = df['Time'].dt.total_seconds()
        elif hasattr(df['Time'].iloc[0], 'total_seconds'):
            df['time_seconds'] = df['Time'].apply(lambda x: x.total_seconds())
        else:
            df['time_seconds'] = df['Time']
    else:
        df['time_seconds'] = df.get('Time_seconds', df.get('Time', 0))

    # Interpolar valores faltantes de temperatura
    for col in ['AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed']:
        if col in df.columns:
            # Interpolar valores NaN
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            # Preencher qualquer NaN restante com média
            df[col] = df[col].fillna(df[col].mean())

    # Normalizar temperaturas (se existirem)
    if 'AirTemp' in df.columns:
        air_mean = df['AirTemp'].mean()
        air_std = df['AirTemp'].std()
        if air_std > 0:
            df['air_temp_normalized'] = (df['AirTemp'] - air_mean) / air_std
        else:
            df['air_temp_normalized'] = 0

    if 'TrackTemp' in df.columns:
        track_mean = df['TrackTemp'].mean()
        track_std = df['TrackTemp'].std()
        if track_std > 0:
            df['track_temp_normalized'] = (df['TrackTemp'] - track_mean) / track_std
        else:
            df['track_temp_normalized'] = 0

    # Calcular delta temperatura (pista - ar)
    if 'TrackTemp' in df.columns and 'AirTemp' in df.columns:
        df['temp_delta'] = df['TrackTemp'] - df['AirTemp']
    else:
        df['temp_delta'] = 0

    # Indicador de chuva
    if 'Rainfall' in df.columns:
        df['rainfall_indicator'] = (df['Rainfall'] > 0).astype(int)
    else:
        df['rainfall_indicator'] = 0

    # Calcular tendência de temperatura (usando diferença entre pontos consecutivos)
    if 'TrackTemp' in df.columns:
        df['temp_trend'] = df['TrackTemp'].diff().fillna(0)
        # Classificar: 1=subindo, -1=descendo, 0=estável
        df['temp_trend_direction'] = np.where(df['temp_trend'] > 0.5, 1,
                                               np.where(df['temp_trend'] < -0.5, -1, 0))
    else:
        df['temp_trend'] = 0
        df['temp_trend_direction'] = 0

    # Detectar mudanças bruscas (temperatura ou chuva)
    df['weather_change'] = 0
    if 'temp_trend' in df.columns:
        # Mudança brusca = variação > 2 desvios padrão
        temp_std = df['temp_trend'].std()
        if temp_std > 0:
            df['weather_change'] = (np.abs(df['temp_trend']) > 2 * temp_std).astype(int)

    # Se começar a chover, também é mudança brusca
    if 'Rainfall' in df.columns:
        rain_change = df['Rainfall'].diff().fillna(0)
        df['weather_change'] = np.maximum(df['weather_change'], (rain_change > 0).astype(int))

    # Ordenar por tempo
    df = df.sort_values('time_seconds').reset_index(drop=True)

    return df


def preprocess_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pré-processa resultados finais da corrida.

    Transforma dados de classificação em features analíticas:
    - Normaliza status de finalização
    - Calcula diferenças de posições
    - Cria features de desempenho relativo
    - Categoriza resultados

    Args:
        df: DataFrame com results (colunas: Position, GridPosition, Points, Status, etc.)

    Returns:
        DataFrame enriquecido com features de resultado:
        - position_change: Diferença entre posição final e grid
        - position_gain: Ganhou posições (1) ou perdeu (0)
        - finish_status: Status normalizado (1=finished, 0=DNF)
        - points_normalized: Pontos normalizados
        - performance_score: Score de desempenho relativo

    Example:
        >>> results = pd.read_parquet('data/raw/races/2025/round_01/results.parquet')
        >>> processed = preprocess_results(results)
        >>> best_performers = processed.nlargest(5, 'position_change')
    """
    df = df.copy()

    # Garantir que posições são numéricas
    if 'Position' in df.columns:
        df['final_position'] = pd.to_numeric(df['Position'], errors='coerce')
    else:
        df['final_position'] = np.nan

    if 'GridPosition' in df.columns:
        df['grid_position'] = pd.to_numeric(df['GridPosition'], errors='coerce')
    else:
        df['grid_position'] = np.nan

    # Calcular mudança de posição (negativo = ganhou posições)
    if 'grid_position' in df.columns and 'final_position' in df.columns:
        df['position_change'] = df['final_position'] - df['grid_position']
        df['position_gain'] = (df['position_change'] < 0).astype(int)
    else:
        df['position_change'] = 0
        df['position_gain'] = 0

    # Normalizar status de finalização
    if 'Status' in df.columns:
        df['status'] = df['Status'].fillna('Unknown')

        # Criar flag de finalização
        df['finish_status'] = df['status'].apply(
            lambda x: 1 if 'Finished' in str(x) or '+' in str(x) else 0
        )

        # Categorizar tipo de DNF
        def categorize_dnf(status):
            status_upper = str(status).upper()
            if 'COLLISION' in status_upper or 'ACCIDENT' in status_upper:
                return 'collision'
            elif 'MECHANICAL' in status_upper or 'ENGINE' in status_upper or 'GEARBOX' in status_upper:
                return 'mechanical'
            elif 'ELECTRICAL' in status_upper or 'POWER' in status_upper:
                return 'electrical'
            elif 'FINISHED' in status_upper or '+' in status_upper:
                return 'finished'
            else:
                return 'other'

        df['dnf_category'] = df['status'].apply(categorize_dnf)
    else:
        df['finish_status'] = 1
        df['dnf_category'] = 'finished'

    # Normalizar pontos
    if 'Points' in df.columns:
        df['points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
        max_points = df['points'].max()
        if max_points > 0:
            df['points_normalized'] = df['points'] / max_points
        else:
            df['points_normalized'] = 0
    else:
        df['points'] = 0
        df['points_normalized'] = 0

    # Calcular score de desempenho
    # Combina: posição final (invertida), mudança de posição, pontos
    df['performance_score'] = 0.0

    if 'final_position' in df.columns:
        # Inverter posição (1º lugar = score alto)
        max_pos = df['final_position'].max()
        if not np.isnan(max_pos) and max_pos > 0:
            df['performance_score'] += (max_pos + 1 - df['final_position'].fillna(max_pos + 1)) / max_pos

    # Adicionar bônus por ganho de posições
    if 'position_change' in df.columns:
        df['performance_score'] += df['position_change'].fillna(0) * -0.1  # Negativo porque ganho é negativo

    # Adicionar pontos normalizados
    if 'points_normalized' in df.columns:
        df['performance_score'] += df['points_normalized']

    # Normalizar performance score para [0, 1]
    if df['performance_score'].max() > 0:
        df['performance_score'] = (df['performance_score'] - df['performance_score'].min()) / \
                                  (df['performance_score'].max() - df['performance_score'].min())

    # Ordenar por posição final
    df = df.sort_values('final_position').reset_index(drop=True)

    return df
