"""
Classificação de causa de anomalias detectadas pelo Isolation Forest.

O Isolation Forest detecta *que* uma volta é anômala, mas não *por quê*.
Este módulo determina a causa cruzando com dados de controle de corrida.

ML-A: anomaly_cause semântico (7 valores) substitui o boolean is_driver_error:
  - safety_car:         Volta sob Safety Car completo
  - virtual_safety_car: Volta sob VSC
  - yellow_flag:        Bandeira amarela local (não SC)
  - pit_lap:            In-lap ou out-lap (pit stop)
  - restart_fast:       Volta rápida pós-relargada (z_score < 0)
  - driver_error:       Causa interna sem evento externo identificado
  - external_incident:  Causa externa (evento externo não-SC)

NARRATIVE_CAUSES: causas que representam incidentes reais para a timeline.

ML-D: removido o caminho morto de classificação por tempo absoluto (guard
"LapTime" nunca era verdadeiro — a coluna chama LapTime_seconds; além disso
time_seconds no race_control é datetime64, não float).
"""

import numpy as np
import pandas as pd
from scipy import stats


# ML-A: Causas narrativamente relevantes — vão para a timeline
NARRATIVE_CAUSES = {"driver_error", "external_incident"}


def classify_anomaly_cause(
    laps_anomalies: pd.DataFrame,
    race_control: pd.DataFrame,
    laps_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Adiciona colunas 'anomaly_cause' e 'z_score' ao DataFrame de anomalias.

    ML-A: anomaly_cause substitui o boolean is_driver_error com 7 categorias
    semânticas, permitindo filtragem precisa em timeline.py sem heurísticas
    retroativas para pit laps, fast laps, etc.

    is_driver_error é mantido por compatibilidade retroativa, mas derivado
    de anomaly_cause: True apenas para driver_error.

    Lógica de classificação (prioridade decrescente):
        1. z_score < 0 → restart_fast (volta rápida = performance, não incidente)
        2. Pit in-lap ou out-lap (PitInTime/PitOutTime no laps_df) → pit_lap
        3. Laps dentro de intervalo SC (DEPLOYED→IN THIS LAP) → safety_car
        4. Laps dentro de VSC → virtual_safety_car
        5. Laps marcados por bandeira/eventos externos → yellow_flag
        6. Sem causa externa → driver_error

    Args:
        laps_anomalies: DataFrame com is_anomaly, anomaly_score, Driver, LapNumber.
                        Deve conter LapTime_delta (calculado no pipeline ML).
        race_control:   DataFrame processado por preprocess_race_control(),
                        com colunas Lap, Message, is_safety_car, is_flag.
        laps_df:        DataFrame original de laps com PitInTime_seconds e
                        PitOutTime_seconds (para detectar pit laps).

    Returns:
        DataFrame com colunas adicionais:
        - z_score (float): Z-score do LapTime_delta por piloto
        - anomaly_cause (str): Causa semântica (ML-A)
        - is_driver_error (bool): True apenas para driver_error (retrocompat.)
    """
    df = laps_anomalies.copy()

    # --- 1. Calcular Z-score por piloto sobre LapTime_delta ---
    if "LapTime_delta" in df.columns:
        df["z_score"] = df.groupby("Driver")["LapTime_delta"].transform(
            lambda x: stats.zscore(x.fillna(0), nan_policy="omit")
        )
    elif "LapTime_seconds" in df.columns:
        df["z_score"] = df.groupby("Driver")["LapTime_seconds"].transform(
            lambda x: stats.zscore(x.fillna(x.median()), nan_policy="omit")
        )
    else:
        df["z_score"] = 0.0

    # --- 2. Inicializar anomaly_cause: default = driver_error ---
    df["anomaly_cause"] = "driver_error"

    # --- 3. restart_fast: z_score < 0 = volta rápida (performance, não incidente) ---
    fast_lap_mask = df["z_score"] < 0
    df.loc[fast_lap_mask, "anomaly_cause"] = "restart_fast"

    if race_control is None or race_control.empty:
        df["is_driver_error"] = df["anomaly_cause"] == "driver_error"
        return df

    # --- 4. pit_lap: cruzar com laps_df para detectar in-laps e out-laps ---
    if laps_df is not None and not laps_df.empty:
        pit_laps_set = _extract_pit_laps(laps_df)
        if pit_laps_set:
            pit_mask = df.apply(
                lambda row: (str(row.get("Driver", "")), int(row.get("LapNumber", 0))) in pit_laps_set,
                axis=1,
            )
            df.loc[pit_mask & df["is_anomaly"], "anomaly_cause"] = "pit_lap"

    # --- 5. ML-08: separar SC completo de VSC ---
    has_sc_col = "is_safety_car" in race_control.columns
    has_vsc_col = "is_virtual_safety_car" in race_control.columns
    has_flag_col = "is_flag" in race_control.columns

    sc_only_mask = pd.Series(False, index=race_control.index)
    vsc_only_mask = pd.Series(False, index=race_control.index)

    if has_vsc_col:
        vsc_only_mask = race_control["is_virtual_safety_car"].fillna(False)
        if has_sc_col:
            sc_only_mask = race_control["is_safety_car"].fillna(False) & ~vsc_only_mask
        flag_mask = race_control["is_flag"].fillna(False) if has_flag_col else pd.Series(False, index=race_control.index)
        external_mask = sc_only_mask | vsc_only_mask | flag_mask
    elif has_sc_col:
        sc_only_mask = race_control["is_safety_car"].fillna(False)
        flag_mask = race_control["is_flag"].fillna(False) if has_flag_col else pd.Series(False, index=race_control.index)
        external_mask = sc_only_mask | flag_mask
    else:
        external_mask = pd.Series(False, index=race_control.index)

    external_events = race_control[external_mask].copy() if external_mask.any() else pd.DataFrame()

    if external_events.empty:
        df["is_driver_error"] = df["anomaly_cause"] == "driver_error"
        return df

    # --- 6. Marcar voltas com Safety Car (DEPLOYED→IN THIS LAP) → safety_car ---
    sc_intervals = _extract_sc_lap_intervals(race_control)
    sc_laps_set: set[int] = set()
    for deploy_lap, end_lap in sc_intervals:
        sc_laps_set.update(range(max(1, deploy_lap - 1), end_lap + 2))

    if sc_laps_set:
        sc_lap_mask = df["LapNumber"].isin(sc_laps_set) & df["is_anomaly"]
        # Não sobrescrever restart_fast ou pit_lap
        overwritable = df["anomaly_cause"] == "driver_error"
        df.loc[sc_lap_mask & overwritable, "anomaly_cause"] = "safety_car"

    # --- 7. Marcar VSC e flags → virtual_safety_car / yellow_flag ---
    if has_vsc_col and "Lap" in external_events.columns:
        vsc_events = external_events[vsc_only_mask.reindex(external_events.index, fill_value=False)]
        vsc_laps: set[int] = set()
        for lap in vsc_events["Lap"].dropna().astype(int):
            vsc_laps.update([lap, lap + 1, lap + 2])
        if vsc_laps:
            vsc_mask = df["LapNumber"].isin(vsc_laps) & df["is_anomaly"]
            overwritable = df["anomaly_cause"] == "driver_error"
            df.loc[vsc_mask & overwritable, "anomaly_cause"] = "virtual_safety_car"

    if has_flag_col and "Lap" in external_events.columns:
        flag_events = external_events[flag_mask.reindex(external_events.index, fill_value=False)] if has_flag_col else pd.DataFrame()
        flag_laps: set[int] = set()
        for lap in flag_events["Lap"].dropna().astype(int):
            flag_laps.update([lap, lap + 1, lap + 2])
        if flag_laps:
            flag_mask2 = df["LapNumber"].isin(flag_laps) & df["is_anomaly"]
            overwritable = df["anomaly_cause"] == "driver_error"
            df.loc[flag_mask2 & overwritable, "anomaly_cause"] = "yellow_flag"

    # --- 8. Retrocompatibilidade: is_driver_error derivado de anomaly_cause ---
    df["is_driver_error"] = df["anomaly_cause"] == "driver_error"

    return df


def _extract_pit_laps(laps_df: pd.DataFrame) -> set[tuple[str, int]]:
    """
    ML-A: Extrai o conjunto de (Driver, LapNumber) que são pit in-laps ou out-laps.

    Usa PitInTime_seconds (in-lap) e PitOutTime_seconds (out-lap) quando disponíveis.
    Como fallback, detecta por reset de TyreLife (novo stint = out-lap).

    Returns:
        Conjunto de tuplas (driver, lap_number) para cruzamento vetorial.
    """
    pit_laps: set[tuple[str, int]] = set()

    has_pit_in = "PitInTime_seconds" in laps_df.columns
    has_pit_out = "PitOutTime_seconds" in laps_df.columns

    if has_pit_in or has_pit_out:
        for _, row in laps_df.iterrows():
            driver = str(row.get("Driver", ""))
            lap = int(row.get("LapNumber", 0))
            if has_pit_in and pd.notna(row.get("PitInTime_seconds")):
                pit_laps.add((driver, lap))
            if has_pit_out and pd.notna(row.get("PitOutTime_seconds")):
                pit_laps.add((driver, lap))
    elif "Stint" in laps_df.columns and "TyreLife" in laps_df.columns:
        # Fallback: primeiro lap de cada stint = out-lap
        for driver, ddf in laps_df.groupby("Driver"):
            ddf = ddf.sort_values("LapNumber")
            stint_changes = ddf.index[ddf["Stint"].diff() > 0]
            for idx in stint_changes:
                lap = int(ddf.loc[idx, "LapNumber"])
                pit_laps.add((str(driver), lap))
                if lap > 1:
                    pit_laps.add((str(driver), lap - 1))  # in-lap estimado

    return pit_laps


def _extract_sc_lap_intervals(race_control: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Extrai intervalos reais de Safety Car como pares (deploy_lap, end_lap).

    Pareia 'SAFETY CAR DEPLOYED' com 'SAFETY CAR IN THIS LAP' para determinar
    a duração exata de cada período de SC em número de voltas.
    """
    if "Message" not in race_control.columns or "Lap" not in race_control.columns:
        return []

    deployed_laps = sorted(
        race_control[race_control["Message"].str.contains("SAFETY CAR DEPLOYED", na=False)]["Lap"]
        .dropna()
        .astype(int)
        .tolist()
    )
    retracted_laps = sorted(
        race_control[race_control["Message"].str.contains("SAFETY CAR IN THIS LAP", na=False)]["Lap"]
        .dropna()
        .astype(int)
        .tolist()
    )

    intervals: list[tuple[int, int]] = []
    for deploy_lap in deployed_laps:
        end_candidates = [l for l in retracted_laps if l >= deploy_lap]
        end_lap = end_candidates[0] if end_candidates else deploy_lap + 4
        intervals.append((deploy_lap, end_lap))

    return intervals


def _mark_by_lap_number(
    df: pd.DataFrame,
    race_control: pd.DataFrame,
    external_events: pd.DataFrame,
) -> None:
    """
    Marca is_driver_error=False para anomalias durante eventos externos via LapNumber.

    Mantido para compatibilidade; o pipeline principal usa anomaly_cause agora.
    """
    if "Lap" not in external_events.columns and "Lap" not in race_control.columns:
        return

    all_affected: set[int] = set()

    sc_intervals = _extract_sc_lap_intervals(race_control)
    for deploy_lap, end_lap in sc_intervals:
        all_affected.update(range(max(1, deploy_lap - 1), end_lap + 2))

    if "Lap" in external_events.columns:
        for lap in external_events["Lap"].dropna().astype(int).tolist():
            all_affected.update([lap, lap + 1, lap + 2])

    if all_affected:
        mask = df["LapNumber"].isin(all_affected) & df["is_anomaly"]
        df.loc[mask, "is_driver_error"] = False
