"""
Builder da RaceTimeline — ponto de encontro entre ML e Pydantic.

Recebe os DataFrames brutos do pipeline de ML, constrói os dicionários de eventos
e os passa pelo RaceTimeline.model_validate(). Qualquer campo inválido (driver com
mais de 3 chars, magnitude negativa, etc.) levanta ValidationError aqui — antes
de qualquer contato com DSPY/Agno.

Formato semântico: a LLM recebe apenas campos narrativos —
  winner/loser (undercut), type (driver_error/external_incident/tire_dropoff/
  safety_car/penalty), positions_lost — sem anomaly_score, z_score.
"""

import logging
import re

import pandas as pd
from pydantic import ValidationError

from src.models.race_events import RaceTimeline

logger = logging.getLogger(__name__)


def build_race_timeline(
    laps_anomalies: pd.DataFrame,
    tire_cliffs: pd.DataFrame,
    undercuts: pd.DataFrame,
    race_control: pd.DataFrame | None = None,
    laps_df: pd.DataFrame | None = None,
) -> RaceTimeline:
    """
    Constrói a RaceTimeline validada pelo Pydantic.

    Mapeamento de campos:
        laps_anomalies (is_driver_error=True):
            Driver    → driver
            LapNumber → lap
            → type: "driver_error"

        laps_anomalies (is_driver_error=False):
            Driver    → driver
            LapNumber → lap
            → type: "external_incident"

        tire_cliffs:
            Driver               → driver
            cliff_lap            → lap
            cliff_delta_magnitude → lap_time_drop_seconds
            cliff_validated      → cliff_validated
            (calculado)          → positions_lost

        undercuts:
            driver              → winner
            target_driver       → loser
            lap                 → lap
            time_gained_seconds → time_gained_seconds

        race_control (SAFETY CAR DEPLOYED):
            Lap         → lap + deployed_on_lap
            (calculado) → duration_laps

        race_control (is_penalty):
            Lap     → lap
            Message → driver (regex) + reason

    Args:
        laps_anomalies: Output de anomaly_classification.classify_anomaly_cause().
        tire_cliffs:    Output de change_point.detect_tire_changepoints().
        undercuts:      Output de strategy.detect_undercuts().
        race_control:   race_control_processed.parquet (opcional — para SC e penalty).
        laps_df:        laps_processed.parquet (opcional — para positions_lost).

    Returns:
        RaceTimeline: lista validada e ordenada cronologicamente por volta.

    Raises:
        ValidationError: Se qualquer evento contém dados inválidos oriundos do ML.
    """
    raw_events = []

    # --- 1. Anomalias do Isolation Forest → driver_error / external_incident ---
    required_anomaly = {"is_anomaly", "Driver", "LapNumber", "is_driver_error"}
    if not laps_anomalies.empty and required_anomaly.issubset(laps_anomalies.columns):
        # Filtrar anomalias sem valor narrativo:
        # - LapNumber == 1: formação/SC inicial — todos os pilotos são lentos
        # - IsAccurate == False + LapNumber <= 5: artefatos do SC inicial
        narrative_mask = laps_anomalies["is_anomaly"]
        narrative_mask = narrative_mask & (laps_anomalies["LapNumber"] > 1)
        if "IsAccurate" in laps_anomalies.columns:
            narrative_mask = narrative_mask & (
                laps_anomalies["IsAccurate"] | (laps_anomalies["LapNumber"] > 5)
            )
        anomalous = laps_anomalies[narrative_mask]
        for _, row in anomalous.iterrows():
            event_type = "driver_error" if bool(row["is_driver_error"]) else "external_incident"
            raw_events.append({
                "type": event_type,
                "lap": int(row["LapNumber"]),
                "driver": str(row["Driver"])[:3],
            })
    else:
        missing = required_anomaly - set(laps_anomalies.columns)
        if missing:
            logger.warning("laps_anomalies: colunas ausentes %s — anomalias ignoradas", missing)

    # --- 2. Tire Cliffs do Ruptures/PELT → tire_dropoff ---
    required_cliffs = {"has_cliff", "Driver", "cliff_lap", "cliff_delta_magnitude", "cliff_validated"}
    if not tire_cliffs.empty and required_cliffs.issubset(tire_cliffs.columns):
        cliffs = tire_cliffs[tire_cliffs["has_cliff"]]
        for _, row in cliffs.iterrows():
            magnitude = float(row["cliff_delta_magnitude"])
            if magnitude <= 0.0:
                logger.debug(
                    "Tire cliff de %s na volta %s ignorado: magnitude %.3f <= 0",
                    row["Driver"], row["cliff_lap"], magnitude,
                )
                continue
            cliff_lap = int(row["cliff_lap"])
            driver = str(row["Driver"])[:3]
            positions_lost = _calc_positions_lost(laps_df, driver, cliff_lap)
            event = {
                "type": "tire_dropoff",
                "lap": cliff_lap,
                "driver": driver,
                "lap_time_drop_seconds": magnitude,
                "cliff_validated": bool(row["cliff_validated"]),
            }
            if positions_lost is not None:
                event["positions_lost"] = positions_lost
            raw_events.append(event)
    else:
        missing = required_cliffs - set(tire_cliffs.columns)
        if missing:
            logger.warning("tire_cliffs: colunas ausentes %s — cliffs ignorados", missing)

    # --- 3. Undercuts da estratégia → undercut (winner/loser) ---
    required_undercuts = {"driver", "target_driver", "lap", "time_gained_seconds"}
    if not undercuts.empty and required_undercuts.issubset(undercuts.columns):
        for _, row in undercuts.iterrows():
            raw_events.append({
                "type": "undercut",
                "lap": int(row["lap"]),
                "winner": str(row["driver"])[:3],
                "loser": str(row["target_driver"])[:3],
                "time_gained_seconds": float(row["time_gained_seconds"]),
            })

    # --- 4. Safety Cars do race_control → safety_car ---
    if race_control is not None and not race_control.empty:
        sc_events = _extract_safety_car_events(race_control)
        raw_events.extend(sc_events)

        pen_events = _extract_penalty_events(race_control)
        raw_events.extend(pen_events)

    # --- 5. Validação pelo Pydantic — Muralha de Fogo ---
    try:
        timeline = RaceTimeline.model_validate(raw_events)
    except ValidationError as exc:
        logger.error("Falha de validação Pydantic — dados do ML contêm valores inválidos:\n%s", exc)
        raise

    type_counts: dict[str, int] = {}
    for e in timeline.root:
        t = e.type
        type_counts[t] = type_counts.get(t, 0) + 1

    logger.info(
        "RaceTimeline construída: %d eventos — %s",
        len(timeline.root),
        ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())),
    )

    return timeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calc_positions_lost(
    laps_df: pd.DataFrame | None,
    driver: str,
    cliff_lap: int,
    window: int = 3,
) -> int | None:
    """
    Calcula posições perdidas pelo piloto ao redor da volta do cliff.

    Compara posição median(cliff_lap-window .. cliff_lap-1)
    com posição median(cliff_lap+1 .. cliff_lap+window).

    Returns:
        Número de posições perdidas (positivo = perdeu posições),
        ou None se não calculável.
    """
    if laps_df is None or "Position" not in laps_df.columns:
        return None

    driver_laps = laps_df[laps_df["Driver"] == driver].copy()
    if driver_laps.empty:
        return None

    before = driver_laps[
        (driver_laps["LapNumber"] >= cliff_lap - window) &
        (driver_laps["LapNumber"] < cliff_lap)
    ]["Position"].dropna()

    after = driver_laps[
        (driver_laps["LapNumber"] > cliff_lap) &
        (driver_laps["LapNumber"] <= cliff_lap + window)
    ]["Position"].dropna()

    if before.empty or after.empty:
        return None

    pos_before = before.median()
    pos_after = after.median()
    lost = int(round(pos_after - pos_before))
    return lost if lost > 0 else None


def _extract_safety_car_events(race_control: pd.DataFrame) -> list[dict]:
    """
    Extrai eventos de Safety Car do race_control.

    Emparelha cada 'SAFETY CAR DEPLOYED' com o próximo 'SAFETY CAR IN THIS LAP'
    para calcular duration_laps.
    """
    events = []
    if "Message" not in race_control.columns or "Lap" not in race_control.columns:
        return events

    deployed = race_control[
        race_control["Message"].str.contains("SAFETY CAR DEPLOYED", na=False)
    ].sort_values("Lap")

    retracted = race_control[
        race_control["Message"].str.contains("SAFETY CAR IN THIS LAP", na=False)
    ].sort_values("Lap")

    retracted_laps = retracted["Lap"].dropna().astype(int).tolist()

    for _, row in deployed.iterrows():
        start_lap = int(row["Lap"]) if pd.notna(row["Lap"]) else None
        if start_lap is None:
            continue

        # Próxima recolhida após o deploy
        end_laps = [l for l in retracted_laps if l >= start_lap]
        end_lap = end_laps[0] if end_laps else start_lap
        duration = max(1, end_lap - start_lap)

        events.append({
            "type": "safety_car",
            "lap": start_lap,
            "deployed_on_lap": start_lap,
            "duration_laps": duration,
        })

    return events


def _extract_penalty_events(race_control: pd.DataFrame) -> list[dict]:
    """
    Extrai penalidades do race_control.

    Filtra apenas penalidades efetivas (exclui SERVED, UNDER INVESTIGATION,
    NO FURTHER ACTION, NOTED — mantém apenas a penalidade principal).
    """
    events = []
    if "is_penalty" not in race_control.columns or "Message" not in race_control.columns:
        return events

    # Filtrar apenas a penalidade principal (não "PENALTY SERVED" nem investigações)
    exclude_pattern = r"SERVED|UNDER INVESTIGATION|NO FURTHER|NOTED"
    penalties = race_control[
        race_control["is_penalty"] &
        ~race_control["Message"].str.contains(exclude_pattern, na=False, case=False)
    ]

    for _, row in penalties.iterrows():
        lap = int(row["Lap"]) if pd.notna(row.get("Lap")) else None
        if lap is None:
            continue

        message = str(row["Message"])

        # Extrair sigla do piloto: "FOR CAR 5 (BOR)" → "BOR"
        match = re.search(r"\(([A-Z]{2,3})\)", message)
        if not match:
            logger.debug("Não foi possível extrair piloto de penalidade: %s", message)
            continue
        driver = match.group(1)[:3]

        # Extrair motivo: tudo após o último " - "
        reason_match = re.search(r" - (.+)$", message)
        reason = reason_match.group(1).title() if reason_match else "Infraction"

        events.append({
            "type": "penalty",
            "lap": lap,
            "driver": driver,
            "reason": reason,
        })

    return events
