"""
Builder determinístico para RaceSummary.

Calcula 100% por código (Pandas/NumPy) a partir dos parquets processados —
sem qualquer inferência de LLM. O output é validado pelo Pydantic antes de
ser serializado para race_summary.json.
"""

import logging

import pandas as pd

from src.models.race_summary import DnfEntry, PodiumEntry, RaceSummary, WeatherSummary

logger = logging.getLogger(__name__)


def build_race_summary(
    results: pd.DataFrame,
    weather: pd.DataFrame,
    laps: pd.DataFrame,
    race_control: pd.DataFrame,
    year: int,
    round_num: int,
) -> RaceSummary:
    """
    Constrói o RaceSummary a partir dos parquets processados.

    Args:
        results:      results_processed.parquet — posição final, grid, status, pontos.
        weather:      weather_processed.parquet — temperatura, chuva.
        laps:         laps_processed.parquet — voltas com LapTime_seconds.
        race_control: race_control_processed.parquet — safety cars.
        year:         Ano da temporada.
        round_num:    Número da rodada.

    Returns:
        RaceSummary validado pelo Pydantic.
    """
    # --- Vencedor ---
    winner_row = results[results["final_position"] == 1]
    winner = str(winner_row["Abbreviation"].iloc[0]) if not winner_row.empty else "UNK"

    # --- Total de voltas ---
    total_laps = int(laps["LapNumber"].max()) if not laps.empty else 0

    # --- Volta mais rápida ---
    valid_laps = laps[laps["LapTime_seconds"] > 0].copy()
    if not valid_laps.empty:
        fl_idx = valid_laps["LapTime_seconds"].idxmin()
        fastest_lap_driver = str(valid_laps.loc[fl_idx, "Driver"])[:3]
        fastest_lap_time = float(valid_laps.loc[fl_idx, "LapTime_seconds"])
    else:
        fastest_lap_driver = winner
        fastest_lap_time = 0.0

    # --- Pódio (top 3) ---
    podium_rows = results[results["final_position"].isin([1, 2, 3])].sort_values("final_position")
    podium = []
    for _, row in podium_rows.iterrows():
        pos = int(row["final_position"])
        if pos == 1:
            gap = "0.000"
        else:
            gap = f"+{pos - 1} lap(s)" if row.get("Status", "").strip() == "Lapped" else f"+{pos:.3f}s"

        podium.append(
            PodiumEntry(
                position=pos,
                driver=str(row["Abbreviation"])[:3],
                team=str(row["TeamName"]),
                gap_to_leader=gap,
            )
        )

    # --- DNFs ---
    dnf_rows = results[results["finish_status"] != 1].copy()
    dnfs = []
    for _, row in dnf_rows.iterrows():
        driver_abbr = str(row["Abbreviation"])[:3]
        # Estimar volta de abandono: última volta completada pelo piloto
        driver_laps = laps[laps["Driver"] == driver_abbr]
        on_lap = int(driver_laps["LapNumber"].max()) if not driver_laps.empty else None
        dnfs.append(DnfEntry(driver=driver_abbr, on_lap=on_lap))

    # --- Contagem de Safety Cars ---
    safety_car_count = 0
    if not race_control.empty and "Message" in race_control.columns:
        safety_car_count = int(
            race_control["Message"].str.contains("SAFETY CAR DEPLOYED", na=False).sum()
        )

    # --- Resumo do tempo ---
    weather_summary = _build_weather_summary(weather)

    summary = RaceSummary(
        year=year,
        round=round_num,
        total_laps=total_laps,
        winner=winner[:3],
        fastest_lap_driver=fastest_lap_driver[:3],
        fastest_lap_time=round(fastest_lap_time, 3),
        podium=podium,
        dnfs=dnfs,
        safety_car_count=safety_car_count,
        weather=weather_summary,
    )

    logger.info(
        "RaceSummary construído: vencedor=%s, %d voltas, %d DNFs, %d SCs",
        winner,
        total_laps,
        len(dnfs),
        safety_car_count,
    )

    return summary


def _build_weather_summary(weather: pd.DataFrame) -> WeatherSummary:
    """Calcula o resumo meteorológico a partir do DataFrame de weather."""
    if weather.empty:
        return WeatherSummary(
            condition="dry",
            air_temp_avg_c=0.0,
            track_temp_avg_c=0.0,
            had_rainfall=False,
        )

    had_rainfall = bool(weather["Rainfall"].any()) if "Rainfall" in weather.columns else False

    air_temp_avg = float(weather["AirTemp"].mean()) if "AirTemp" in weather.columns else 0.0
    track_temp_avg = float(weather["TrackTemp"].mean()) if "TrackTemp" in weather.columns else 0.0

    if had_rainfall:
        # Se teve chuva em algum momento mas não todo o tempo → mixed
        rainfall_frac = float(weather["Rainfall"].mean()) if "Rainfall" in weather.columns else 0.0
        condition = "wet" if rainfall_frac > 0.5 else "mixed"
    else:
        condition = "dry"

    return WeatherSummary(
        condition=condition,
        air_temp_avg_c=round(air_temp_avg, 1),
        track_temp_avg_c=round(track_temp_avg, 1),
        had_rainfall=had_rainfall,
    )
