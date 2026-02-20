"""
Builder determinístico para DriverProfile (lista de perfis por piloto).

Calcula 100% por código (Pandas/NumPy) a partir dos parquets processados —
sem qualquer inferência de LLM. O output é validado pelo Pydantic antes de
ser serializado para driver_profiles.json.
"""

import logging

import pandas as pd

from src.models.driver_profile import CompoundUsage, DriverProfile

logger = logging.getLogger(__name__)


def build_driver_profiles(
    results: pd.DataFrame,
    laps_clustered: pd.DataFrame,
    anomalies_summary: pd.DataFrame,
    tire_cliffs_summary: pd.DataFrame,
    laps: pd.DataFrame,
) -> list[DriverProfile]:
    """
    Constrói a lista de DriverProfile a partir dos parquets processados.

    Args:
        results:             results_processed.parquet — posição final, grid, status, pontos.
        laps_clustered:      laps_clustered.parquet — cluster_semantic por volta.
        anomalies_summary:   anomalies_summary.parquet — contagem de anomalias por piloto.
        tire_cliffs_summary: tire_cliffs_summary.parquet — cliffs por piloto.
        laps:                laps_processed.parquet — compostos e voltas brutas.

    Returns:
        Lista de DriverProfile validados pelo Pydantic, um por piloto.
    """
    profiles = []

    # --- Calcular percentuais de cluster por piloto ---
    cluster_pct = pd.DataFrame()
    if not laps_clustered.empty and "cluster_semantic" in laps_clustered.columns:
        cluster_pct = (
            laps_clustered.groupby("Driver")["cluster_semantic"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
        )

    # --- Calcular uso de compostos por piloto ---
    compound_by_driver: dict[str, list[CompoundUsage]] = {}
    if not laps.empty and "Compound" in laps.columns:
        compound_counts = (
            laps.groupby(["Driver", "Compound"])["LapNumber"]
            .count()
            .reset_index()
            .rename(columns={"LapNumber": "laps"})
        )
        for driver, grp in compound_counts.groupby("Driver"):
            compound_by_driver[str(driver)] = [
                CompoundUsage(compound=str(row["Compound"]), laps=int(row["laps"]))
                for _, row in grp.iterrows()
            ]

    # --- Indexar anomalies_summary por driver ---
    anomaly_index: dict[str, int] = {}
    if not anomalies_summary.empty and "anomalies_count" in anomalies_summary.columns:
        for _, row in anomalies_summary.iterrows():
            anomaly_index[str(row["Driver"])] = int(row["anomalies_count"])

    # --- Indexar tire_cliffs_summary por driver ---
    cliff_index: dict[str, dict] = {}
    if not tire_cliffs_summary.empty:
        for _, row in tire_cliffs_summary.iterrows():
            cliff_index[str(row["Driver"])] = {
                "had_cliff": int(row.get("stints_with_cliff", 0)) > 0,
                "cliff_count": int(row.get("stints_with_cliff", 0)),
            }

    # --- Construir perfil por piloto (a partir dos results para garantir todos os pilotos) ---
    for _, res_row in results.iterrows():
        driver = str(res_row["Abbreviation"])[:3]
        team = str(res_row["TeamName"])

        final_pos = res_row.get("final_position")
        final_position = int(final_pos) if pd.notna(final_pos) else None

        grid_pos = res_row.get("grid_position")
        grid_position = int(grid_pos) if pd.notna(grid_pos) else None

        if final_position is not None and grid_position is not None:
            positions_gained = grid_position - final_position
        else:
            positions_gained = 0

        points = float(res_row.get("points", 0.0) or 0.0)

        # Percentuais de cluster
        push_pct = float(cluster_pct.loc[driver, "push"]) if driver in cluster_pct.index and "push" in cluster_pct.columns else 0.0
        base_pct = float(cluster_pct.loc[driver, "base"]) if driver in cluster_pct.index and "base" in cluster_pct.columns else 0.0
        degraded_pct = float(cluster_pct.loc[driver, "degraded"]) if driver in cluster_pct.index and "degraded" in cluster_pct.columns else 0.0

        # Compostos usados
        compounds = compound_by_driver.get(driver, [])
        if not compounds:
            # fallback: ao menos um composto desconhecido para satisfazer min_length=1
            compounds = [CompoundUsage(compound="UNKNOWN", laps=1)]

        # Tire cliffs
        cliff_data = cliff_index.get(driver, {"had_cliff": False, "cliff_count": 0})
        had_tire_cliff = cliff_data["had_cliff"]
        cliff_count = cliff_data["cliff_count"]

        # Anomalias
        anomaly_count = anomaly_index.get(driver, 0)

        profile = DriverProfile(
            driver=driver,
            team=team,
            final_position=final_position,
            grid_position=grid_position,
            positions_gained=positions_gained,
            points=points,
            push_pct=round(push_pct, 3),
            base_pct=round(base_pct, 3),
            degraded_pct=round(degraded_pct, 3),
            compounds_used=compounds,
            had_tire_cliff=had_tire_cliff,
            cliff_count=cliff_count,
            anomaly_count=anomaly_count,
        )
        profiles.append(profile)

    logger.info(
        "DriverProfiles construídos: %d pilotos",
        len(profiles),
    )

    return profiles
