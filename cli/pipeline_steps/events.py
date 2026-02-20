"""
Fase 4: Eventos Estruturados — Pydantic como Muralha de Fogo.

Lê os outputs da Fase 3 (ML), classifica causas de anomalias,
detecta undercuts e gera os 3 JSONs validados pelo Pydantic:

    timeline.json        — eventos cronológicos (formato semântico)
    race_summary.json    — resumo geral da corrida
    driver_profiles.json — perfil de desempenho por piloto

Output: data/timelines/races/YEAR/round_XX/
"""

import json
import logging
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.ml.anomaly_classification import classify_anomaly_cause
from src.ml.driver_profiles_builder import build_driver_profiles
from src.ml.race_summary_builder import build_race_summary
from src.ml.strategy import detect_undercuts
from src.ml.timeline import build_race_timeline
from .reporting import Reporter

logger = logging.getLogger(__name__)


def run_events_phase(
    ml_dir: Path,
    processed_dir: Path,
    year: int,
    round_num: int,
) -> Path:
    """
    Executa a Fase 4: classificação de causas, detecção de undercuts e geração dos 3 JSONs.

    Args:
        ml_dir:        Diretório com outputs da Fase 3 (laps_anomalies.parquet, etc.).
        processed_dir: Diretório com dados pré-processados.
        year:          Ano da temporada.
        round_num:     Número da rodada.

    Returns:
        Path para o diretório onde os JSONs foram salvos.
    """
    reporter = Reporter("FASE 4: EVENTOS ESTRUTURADOS (PYDANTIC)")
    reporter.header()

    timeline_dir = Path("data/timelines/races") / f"{year}" / f"round_{round_num:02d}"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    # --- Carregar outputs da Fase 3 ---
    laps_anomalies_file = ml_dir / "laps_anomalies.parquet"
    tire_cliffs_file = ml_dir / "tire_cliffs.parquet"
    laps_clustered_file = ml_dir / "laps_clustered.parquet"
    anomalies_summary_file = ml_dir / "anomalies_summary.parquet"
    tire_cliffs_summary_file = ml_dir / "tire_cliffs_summary.parquet"

    if not laps_anomalies_file.exists():
        reporter.info("⚠️  laps_anomalies.parquet não encontrado. Pulando Fase 4.", indent=0)
        return timeline_dir

    reporter.section("4.1", "Carregando outputs do ML")
    laps_anomalies = pd.read_parquet(laps_anomalies_file)
    reporter.info(f"{len(laps_anomalies)} voltas | {int(laps_anomalies['is_anomaly'].sum())} anomalias")

    tire_cliffs = pd.DataFrame()
    if tire_cliffs_file.exists():
        tire_cliffs = pd.read_parquet(tire_cliffs_file)
        n_cliffs = int(tire_cliffs["has_cliff"].sum()) if "has_cliff" in tire_cliffs.columns else 0
        reporter.info(f"{n_cliffs} tire cliffs detectados")

    laps_clustered = pd.DataFrame()
    if laps_clustered_file.exists():
        laps_clustered = pd.read_parquet(laps_clustered_file)

    anomalies_summary = pd.DataFrame()
    if anomalies_summary_file.exists():
        anomalies_summary = pd.read_parquet(anomalies_summary_file)

    tire_cliffs_summary = pd.DataFrame()
    if tire_cliffs_summary_file.exists():
        tire_cliffs_summary = pd.read_parquet(tire_cliffs_summary_file)

    # --- Carregar dados processados ---
    laps_df = None
    laps_processed_file = processed_dir / "laps_processed.parquet"
    if laps_processed_file.exists():
        laps_df = pd.read_parquet(laps_processed_file)

    race_control = None
    race_control_file = processed_dir / "race_control_processed.parquet"
    if race_control_file.exists():
        race_control = pd.read_parquet(race_control_file)
        reporter.info("Race control carregado")

    results = pd.DataFrame()
    results_file = processed_dir / "results_processed.parquet"
    if results_file.exists():
        results = pd.read_parquet(results_file)

    weather = pd.DataFrame()
    weather_file = processed_dir / "weather_processed.parquet"
    if weather_file.exists():
        weather = pd.read_parquet(weather_file)

    # --- 4.2: Classificar causa das anomalias ---
    reporter.section("4.2", "Classificando causas de anomalias (SciPy Z-score + race control)")
    laps_anomalies = classify_anomaly_cause(
        laps_anomalies=laps_anomalies,
        race_control=race_control,
        laps_df=laps_df,
    )
    n_driver_errors = int(laps_anomalies[laps_anomalies["is_anomaly"]]["is_driver_error"].sum())
    n_external = int(laps_anomalies["is_anomaly"].sum()) - n_driver_errors
    reporter.info(f"Erros do piloto: {n_driver_errors} | Causas externas: {n_external}")

    # --- 4.3: Detectar undercuts ---
    reporter.section("4.3", "Detectando manobras de undercut")
    undercuts = pd.DataFrame(columns=["driver", "target_driver", "lap", "time_gained_seconds"])
    if laps_df is not None:
        undercuts = detect_undercuts(laps_df)
        reporter.info(f"{len(undercuts)} undercuts detectados")
    else:
        reporter.info("⚠️  laps_processed.parquet não encontrado — undercuts não detectados")

    # --- 4.4: Construir timeline.json ---
    reporter.section("4.4", "Construindo RaceTimeline (validação Pydantic)")
    try:
        timeline = build_race_timeline(
            laps_anomalies=laps_anomalies,
            tire_cliffs=tire_cliffs,
            undercuts=undercuts,
            race_control=race_control,
            laps_df=laps_df,
        )
    except ValidationError as exc:
        reporter.info(f"❌ Falha de validação Pydantic:\n{exc}", indent=0)
        raise

    reporter.info(f"{len(timeline.root)} eventos na timeline (ordenados cronologicamente)")

    out_timeline = timeline_dir / "timeline.json"
    out_timeline.write_text(timeline.to_json(), encoding="utf-8")
    reporter.success(f"timeline.json salvo: {out_timeline}", indent=0)

    # --- 4.5: Construir race_summary.json ---
    reporter.section("4.5", "Construindo race_summary.json")
    if not results.empty and not laps_df is None:
        try:
            race_summary = build_race_summary(
                results=results,
                weather=weather,
                laps=laps_df,
                race_control=race_control if race_control is not None else pd.DataFrame(),
                year=year,
                round_num=round_num,
            )
            out_summary = timeline_dir / "race_summary.json"
            out_summary.write_text(
                race_summary.model_dump_json(indent=2),
                encoding="utf-8",
            )
            reporter.success(f"race_summary.json salvo: {out_summary}", indent=0)
        except Exception as exc:
            reporter.info(f"⚠️  Falha ao gerar race_summary.json: {exc}", indent=0)
            logger.exception("Falha ao gerar race_summary.json")
    else:
        reporter.info("⚠️  results ou laps ausentes — race_summary.json não gerado")

    # --- 4.6: Construir driver_profiles.json ---
    reporter.section("4.6", "Construindo driver_profiles.json")
    if not results.empty and not laps_clustered.empty:
        try:
            profiles = build_driver_profiles(
                results=results,
                laps_clustered=laps_clustered,
                anomalies_summary=anomalies_summary,
                tire_cliffs_summary=tire_cliffs_summary,
                laps=laps_df if laps_df is not None else pd.DataFrame(),
            )
            out_profiles = timeline_dir / "driver_profiles.json"
            out_profiles.write_text(
                json.dumps(
                    [p.model_dump() for p in profiles],
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            reporter.success(f"driver_profiles.json salvo: {out_profiles}", indent=0)
        except Exception as exc:
            reporter.info(f"⚠️  Falha ao gerar driver_profiles.json: {exc}", indent=0)
            logger.exception("Falha ao gerar driver_profiles.json")
    else:
        reporter.info("⚠️  results ou laps_clustered ausentes — driver_profiles.json não gerado")

    return timeline_dir
