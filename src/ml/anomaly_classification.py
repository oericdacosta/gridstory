"""
Classificação de causa de anomalias detectadas pelo Isolation Forest.

O Isolation Forest detecta *que* uma volta é anômala, mas não *por quê*.
Este módulo determina a causa cruzando com dados de controle de corrida:

- Causa externa (SC/VSC/bandeira): is_driver_error = False
- Causa interna (erro/mecânica):   is_driver_error = True

Também calcula o Z-score via SciPy para confirmar significância estatística,
alinhado com a stack do projeto (SciPy para análise estatística rigorosa).
"""

import numpy as np
import pandas as pd
from scipy import stats


def classify_anomaly_cause(
    laps_anomalies: pd.DataFrame,
    race_control: pd.DataFrame,
    laps_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Adiciona colunas 'is_driver_error' e 'z_score' ao DataFrame de anomalias.

    Lógica de classificação:
        1. Identifica intervalos de tempo com SC/VSC ou bandeira amarela/vermelha
           a partir de race_control.
        2. Para cada volta anômala, verifica se ela ocorreu durante um desses eventos.
        3. Se sim → causa externa → is_driver_error = False.
        4. Se não → sem causa externa identificável → is_driver_error = True.

    O Z-score é calculado sobre LapTime_delta por piloto: mede quantos desvios
    padrão a volta está afastada da mediana do piloto (confirmação estatística).

    Args:
        laps_anomalies: DataFrame com is_anomaly, anomaly_score, Driver, LapNumber.
                        Deve conter LapTime_delta (calculado no pipeline ML).
        race_control:   DataFrame processado por preprocess_race_control(),
                        com colunas time_seconds, is_safety_car, is_flag.
        laps_df:        DataFrame original de laps com LapTime (timestamps).
                        Se None, usa LapNumber como proxy de timing.

    Returns:
        DataFrame com colunas adicionais:
        - z_score (float): Z-score do LapTime_delta por piloto
        - is_driver_error (bool): True = sem causa externa identificada
    """
    df = laps_anomalies.copy()

    # --- 1. Calcular Z-score por piloto sobre LapTime_delta ---
    if "LapTime_delta" in df.columns:
        df["z_score"] = df.groupby("Driver")["LapTime_delta"].transform(
            lambda x: stats.zscore(x.fillna(0), nan_policy="omit")
        )
    elif "LapTime_seconds" in df.columns:
        # Fallback: Z-score sobre LapTime_seconds se delta não disponível
        df["z_score"] = df.groupby("Driver")["LapTime_seconds"].transform(
            lambda x: stats.zscore(x.fillna(x.median()), nan_policy="omit")
        )
    else:
        df["z_score"] = 0.0

    # --- 2. Classificar por z_score antes de verificar causas externas ---
    # Z-score negativo = lap mais rápida que a mediana do piloto.
    # Uma lap anormalmente rápida é push tático / pneu novo — nunca um erro do piloto.
    df["is_driver_error"] = True  # default: sem causa externa identificada

    # Voltas mais rápidas que a mediana (z_score < 0) não são erros — são performance
    fast_lap_mask = df["z_score"] < 0
    df.loc[fast_lap_mask, "is_driver_error"] = False

    if race_control is None or race_control.empty:
        return df

    # Eventos externos: safety car ou bandeira amarela/vermelha
    external_events = race_control[
        race_control.get("is_safety_car", pd.Series(False, index=race_control.index)) |
        race_control.get("is_flag", pd.Series(False, index=race_control.index))
    ].copy() if "is_safety_car" in race_control.columns or "is_flag" in race_control.columns else pd.DataFrame()

    if external_events.empty:
        return df

    # Construir intervalos de evento externo (início → próximo evento ou fim da corrida)
    # race_control está ordenado por time_seconds
    external_times = sorted(external_events["time_seconds"].dropna().tolist())
    if not external_times:
        return df

    # --- 3. Marcar voltas durante eventos externos ---
    # Se temos timestamps reais das voltas, usar; caso contrário, usar LapNumber como proxy
    if laps_df is not None and "LapTime" in laps_df.columns and "LapNumber" in laps_df.columns:
        # Tentar obter tempo em segundos de cada volta
        lap_time_col = _get_lap_timing(laps_df)
        if lap_time_col is not None:
            _mark_by_timing(df, lap_time_col, external_times)
            return df

    # Fallback: usar LapNumber como proxy — se houve SC na metade da corrida,
    # estima qual volta corresponde ao evento (aproximação por fração de voltas)
    _mark_by_lap_number(df, race_control, external_events)

    return df


def _get_lap_timing(laps_df: pd.DataFrame) -> pd.Series | None:
    """Extrai série de tempo em segundos por LapNumber, se possível."""
    if "Time_seconds" in laps_df.columns:
        return laps_df.set_index("LapNumber")["Time_seconds"]
    if "Time" in laps_df.columns:
        try:
            col = laps_df["Time"]
            if pd.api.types.is_timedelta64_dtype(col):
                return laps_df.set_index("LapNumber")[col.dt.total_seconds().name]
        except Exception:
            pass
    return None


def _mark_by_timing(
    df: pd.DataFrame,
    lap_timing: pd.Series,
    external_times: list[float],
    window_seconds: float = 120.0,
) -> None:
    """
    Marca is_driver_error=False para voltas que ocorreram durante evento externo.

    Usa uma janela de 'window_seconds' após o início do evento (SC dura ~2 min mínimo).
    """
    for event_time in external_times:
        affected_laps = lap_timing[
            (lap_timing >= event_time) &
            (lap_timing <= event_time + window_seconds)
        ].index.tolist()

        if affected_laps:
            mask = df["LapNumber"].isin(affected_laps) & df["is_anomaly"]
            df.loc[mask, "is_driver_error"] = False


def _mark_by_lap_number(
    df: pd.DataFrame,
    race_control: pd.DataFrame,
    external_events: pd.DataFrame,
) -> None:
    """
    Fallback: estima voltas afetadas por eventos externos via LapNumber.

    Usa 'RacingNumber' ou 'Lap' se disponível em race_control,
    ou estima por proporção de tempo da corrida.
    """
    # Se race_control tem coluna de volta direta (alguns formatos FastF1)
    if "Lap" in external_events.columns:
        affected_laps = external_events["Lap"].dropna().astype(int).tolist()
        if affected_laps:
            # Marca a volta do evento + 1 volta seguinte (SC normalmente dura mais de 1 volta)
            all_affected = set()
            for lap in affected_laps:
                all_affected.update([lap, lap + 1, lap + 2])
            mask = df["LapNumber"].isin(all_affected) & df["is_anomaly"]
            df.loc[mask, "is_driver_error"] = False
