"""
Pré-processamento específico do domínio F1.

Transforma dados brutos de corrida (race control, weather, results) em features
estruturadas para análise e ML.
"""

import numpy as np
import pandas as pd


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
