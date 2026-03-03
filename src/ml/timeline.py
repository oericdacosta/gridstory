"""
Builder da RaceTimeline — ponto de encontro entre ML e Pydantic.

Recebe os DataFrames brutos do pipeline de ML, constrói os dicionários de eventos
e os passa pelo RaceTimeline.model_validate(). Qualquer campo inválido (driver com
mais de 3 chars, magnitude negativa, etc.) levanta ValidationError aqui — antes
de qualquer contato com DSPY/Agno.

Melhorias v2:
  - SafetyCarEvent inclui `cause` extraída do race_control (mensagens próximas ao SC).
  - RetirementEvent adicionado para todo piloto nos DNFs sem retirement na timeline,
    garantindo que abandonos como o de ALO sejam sempre narrados.
"""

import logging
import re

import pandas as pd
from pydantic import ValidationError

from src.ml.anomaly_classification import NARRATIVE_CAUSES
from src.models.race_events import RaceTimeline

logger = logging.getLogger(__name__)

# Padrões de mensagem que indicam a causa de um safety car
_SC_CAUSE_PATTERNS = re.compile(
    r"(INCIDENT|RETIRED|COLLISION|ACCIDENT|CRASH|SPIN|OFF TRACK|DEBRIS|CONTACT)",
    re.IGNORECASE,
)


def build_race_timeline(
    laps_anomalies: pd.DataFrame,
    tire_cliffs: pd.DataFrame,
    undercuts: pd.DataFrame,
    race_control: pd.DataFrame | None = None,
    laps_df: pd.DataFrame | None = None,
    results: pd.DataFrame | None = None,
    overcuts: pd.DataFrame | None = None,
) -> RaceTimeline:
    """
    Constrói a RaceTimeline validada pelo Pydantic.

    Args:
        laps_anomalies: Output de anomaly_classification.classify_anomaly_cause().
        tire_cliffs:    Output de change_point.detect_tire_changepoints().
        undercuts:      Output de strategy.detect_undercuts().
        race_control:   race_control_processed.parquet (opcional).
        laps_df:        laps_processed.parquet (opcional).
        results:        results_processed.parquet (opcional — para eventos de abandono).
        overcuts:       Output de strategy.detect_overcuts() (ML-06, opcional).

    Returns:
        RaceTimeline: lista validada e ordenada cronologicamente por volta.

    Raises:
        ValidationError: Se qualquer evento contém dados inválidos oriundos do ML.
    """
    raw_events = []

    # --- 1. Anomalias do Isolation Forest → driver_error / external_incident ---
    # ML-B: usa anomaly_cause (ML-A) quando disponível para filtragem semântica precisa.
    # Fallback: usa is_driver_error + filtros retroativos (compatibilidade com runs antigos).
    required_anomaly_new = {"is_anomaly", "Driver", "LapNumber", "anomaly_cause"}
    required_anomaly_old = {"is_anomaly", "Driver", "LapNumber", "is_driver_error"}

    if not laps_anomalies.empty and required_anomaly_new.issubset(laps_anomalies.columns):
        # Caminho ML-B: filtro semântico via anomaly_cause
        narrative_mask = laps_anomalies["is_anomaly"]
        narrative_mask = narrative_mask & (laps_anomalies["LapNumber"] > 1)
        if "IsAccurate" in laps_anomalies.columns:
            narrative_mask = narrative_mask & (
                laps_anomalies["IsAccurate"] | (laps_anomalies["LapNumber"] > 5)
            )
        # ML-B: só causas narrativamente relevantes (driver_error, external_incident)
        narrative_mask = narrative_mask & laps_anomalies["anomaly_cause"].isin(NARRATIVE_CAUSES)
        anomalous = laps_anomalies[narrative_mask]
        for _, row in anomalous.iterrows():
            event_type = str(row["anomaly_cause"])  # "driver_error" ou "external_incident"
            raw_events.append({
                "type": event_type,
                "lap": int(row["LapNumber"]),
                "driver": str(row["Driver"])[:3],
            })
    elif not laps_anomalies.empty and required_anomaly_old.issubset(laps_anomalies.columns):
        # Fallback: filtros retroativos para runs sem anomaly_cause
        narrative_mask = laps_anomalies["is_anomaly"]
        narrative_mask = narrative_mask & (laps_anomalies["LapNumber"] > 1)
        if "IsAccurate" in laps_anomalies.columns:
            narrative_mask = narrative_mask & (
                laps_anomalies["IsAccurate"] | (laps_anomalies["LapNumber"] > 5)
            )
        if "PitInTime_seconds" in laps_anomalies.columns:
            narrative_mask = narrative_mask & laps_anomalies["PitInTime_seconds"].isna()
        if "PitOutTime_seconds" in laps_anomalies.columns:
            narrative_mask = narrative_mask & laps_anomalies["PitOutTime_seconds"].isna()
        if "z_score" in laps_anomalies.columns:
            narrative_mask = narrative_mask & (laps_anomalies["z_score"] >= 0)
        anomalous = laps_anomalies[narrative_mask]
        for _, row in anomalous.iterrows():
            event_type = "driver_error" if bool(row["is_driver_error"]) else "external_incident"
            raw_events.append({
                "type": event_type,
                "lap": int(row["LapNumber"]),
                "driver": str(row["Driver"])[:3],
            })
    else:
        missing = (required_anomaly_new | required_anomaly_old) - set(laps_anomalies.columns)
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

    # --- 3b. Overcuts da estratégia → overcut (winner/loser) — ML-06 ---
    required_overcuts = {"driver", "target_driver", "lap", "time_saved_seconds"}
    if overcuts is not None and not overcuts.empty and required_overcuts.issubset(overcuts.columns):
        for _, row in overcuts.iterrows():
            raw_events.append({
                "type": "overcut",
                "lap": int(row["lap"]),
                "winner": str(row["driver"])[:3],
                "loser": str(row["target_driver"])[:3],
                "time_saved_seconds": float(row["time_saved_seconds"]),
            })

    # --- 4. Safety Cars e Penalidades do race_control ---
    if race_control is not None and not race_control.empty:
        sc_events = _extract_safety_car_events(race_control)
        raw_events.extend(sc_events)

        pen_events = _extract_penalty_events(race_control)
        raw_events.extend(pen_events)

    # --- 5. Abandonos (RetirementEvent) para todos os pilotos DNF ---
    if results is not None and not results.empty:
        retirement_events = _extract_retirement_events(results, raw_events, race_control, laps_df)
        raw_events.extend(retirement_events)

    # --- 6. Validação pelo Pydantic — Muralha de Fogo ---
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
# Helpers — Cálculo de posições perdidas
# ---------------------------------------------------------------------------

def _calc_positions_lost(
    laps_df: pd.DataFrame | None,
    driver: str,
    cliff_lap: int,
    window: int = 3,
) -> int | None:
    """
    Calcula posições perdidas pelo piloto ao redor da volta do cliff.
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


# ---------------------------------------------------------------------------
# Helpers — Safety Car com causa
# ---------------------------------------------------------------------------

def _find_sc_cause(race_control: pd.DataFrame, sc_lap: int) -> str | None:
    """
    Busca a causa do safety car nas mensagens do race_control próximas ao SC.

    Examina 2 voltas antes e a própria volta do SC, excluindo mensagens de
    safety car/DRS, para encontrar a mensagem de incidente/abandono que o causou.
    """
    if "Message" not in race_control.columns or "Lap" not in race_control.columns:
        return None

    window = race_control[
        (race_control["Lap"] >= sc_lap - 2) &
        (race_control["Lap"] <= sc_lap) &
        ~race_control["Message"].str.contains(
            "SAFETY CAR|VIRTUAL|DRS|PIT LANE|TRACK CLEAR", na=False, case=False
        )
    ].sort_values("Lap")

    for _, row in window.iterrows():
        msg = str(row.get("Message", ""))
        if _SC_CAUSE_PATTERNS.search(msg):
            return msg.strip()

    return None


def _extract_safety_car_events(race_control: pd.DataFrame) -> list[dict]:
    """
    Extrai eventos de Safety Car do race_control, incluindo a causa se disponível.

    Emparelha cada 'SAFETY CAR DEPLOYED' com o próximo 'SAFETY CAR IN THIS LAP'
    para calcular duration_laps. Busca a causa nas mensagens anteriores ao SC.
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

        end_laps = [l for l in retracted_laps if l >= start_lap]
        end_lap = end_laps[0] if end_laps else start_lap
        duration = max(1, end_lap - start_lap)

        cause = _find_sc_cause(race_control, start_lap)

        event: dict = {
            "type": "safety_car",
            "lap": start_lap,
            "deployed_on_lap": start_lap,
            "duration_laps": duration,
        }
        if cause:
            event["cause"] = cause

        events.append(event)

    return events


# ---------------------------------------------------------------------------
# Helpers — Penalidades
# ---------------------------------------------------------------------------

def _extract_penalty_events(race_control: pd.DataFrame) -> list[dict]:
    """
    Extrai penalidades do race_control.
    """
    events = []
    if "is_penalty" not in race_control.columns or "Message" not in race_control.columns:
        return events

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
        match = re.search(r"\(([A-Z]{2,3})\)", message)
        if not match:
            logger.debug("Não foi possível extrair piloto de penalidade: %s", message)
            continue
        driver = match.group(1)[:3]

        reason_match = re.search(r" - (.+)$", message)
        reason = reason_match.group(1).title() if reason_match else "Infraction"

        events.append({"type": "penalty", "lap": lap, "driver": driver, "reason": reason})

    return events


# ---------------------------------------------------------------------------
# Helpers — Abandonos (RetirementEvent)
# ---------------------------------------------------------------------------

def _extract_retirement_events(
    results: pd.DataFrame,
    existing_events: list[dict],
    race_control: pd.DataFrame | None,
    laps_df: pd.DataFrame | None,
) -> list[dict]:
    """
    Adiciona RetirementEvent para todo piloto DNF.

    Garante que nenhum abandono passe em branco na timeline,
    independentemente de o ML ter detectado anomalias para esse piloto.
    """
    if "finish_status" not in results.columns or "Abbreviation" not in results.columns:
        return []

    # Drivers que já têm retirement registrado
    existing_retirements = {
        e["driver"] for e in existing_events if e.get("type") == "retirement"
    }

    retirement_events = []
    dnf_rows = results[results["finish_status"] != 1]

    # Total de voltas da corrida (para distinguir DNF de finalizador completo)
    total_race_laps: int | None = None
    if laps_df is not None and not laps_df.empty and "LapNumber" in laps_df.columns:
        total_race_laps = int(laps_df["LapNumber"].max())

    for _, row in dnf_rows.iterrows():
        driver = str(row["Abbreviation"])[:3]

        if driver in existing_retirements:
            continue

        # Fix off-by-one: o piloto abandona *durante* a volta seguinte à última completada.
        # last_completed = última volta registrada no parquet (volta concluída).
        # on_lap = volta em que o abandono ocorreu = last_completed + 1,
        # a não ser que last_completed já seja a última volta da corrida (piloto completou).
        on_lap = None
        if laps_df is not None and not laps_df.empty:
            driver_laps = laps_df[laps_df["Driver"] == driver]
            if not driver_laps.empty:
                last_completed = int(driver_laps["LapNumber"].max())
                if total_race_laps is not None and last_completed < total_race_laps:
                    on_lap = last_completed + 1   # Abandonou durante a próxima volta
                else:
                    on_lap = last_completed        # Completou ou última volta disponível

        if on_lap is None:
            logger.debug("Abandono de %s sem volta detectada — ignorado", driver)
            continue

        cause = _find_retirement_cause(race_control, driver, on_lap)

        event: dict = {"type": "retirement", "lap": on_lap, "driver": driver}
        if cause:
            event["cause"] = cause

        retirement_events.append(event)
        logger.info("RetirementEvent adicionado: %s na volta %d", driver, on_lap)

    return retirement_events


def _find_retirement_cause(
    race_control: pd.DataFrame | None,
    driver: str,
    on_lap: int,
) -> str | None:
    """
    Busca no race_control uma mensagem de causa de abandono para o piloto
    na janela de ±1 volta ao redor de on_lap.
    """
    if race_control is None or race_control.empty:
        return None
    if "Message" not in race_control.columns or "Lap" not in race_control.columns:
        return None

    window = race_control[
        (race_control["Lap"] >= on_lap - 1) &
        (race_control["Lap"] <= on_lap + 1)
    ]

    for _, row in window.iterrows():
        msg = str(row.get("Message", ""))
        if driver in msg or re.search(r"RETIRED|RETIREMENT|WITHDREW", msg, re.IGNORECASE):
            return msg.strip()

    return None
