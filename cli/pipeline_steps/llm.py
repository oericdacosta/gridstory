"""
Fase 5: Geração de Relatório Jornalístico (DSPy + Groq).

Lê os 3 JSONs validados pelo Pydantic (Fase 4) e gera o relatório
estruturado via DSPy Predict. O output é validado pela
RelatorioSecoes e salvo como relatorio.json no mesmo diretório.

LLM-A: Usa Groq llama-3.3-70b-versatile (latência ~3-5s vs 38s anterior).
LLM-B: Usa dspy.Predict em vez de ChainOfThought (-30% tokens).
ML-C: Passa race_control para validação de incidente_multiplo.
ML-07: Passa driver_quality_scores para informar a LLM sobre dados fragmentados.

Output: data/timelines/races/YEAR/round_XX/relatorio.json
"""

import json
from pathlib import Path

import pandas as pd

from src.llm.reporter import generate_report
from .reporting import Reporter


def run_llm_phase(
    timeline_dir: Path,
    year: int,
    round_num: int,
    processed_dir: Path | None = None,
) -> Path:
    """
    Executa a Fase 5: gera o relatório jornalístico via DSPy + Groq.

    Args:
        timeline_dir:  Diretório com os JSONs da Fase 4 (race_summary, timeline, driver_profiles).
        year:          Ano da temporada.
        round_num:     Número da rodada.
        processed_dir: Diretório com dados processados (para race_control e quality scores).

    Returns:
        Path para o diretório com o relatorio.json gerado.
    """
    reporter = Reporter("FASE 5: RELATÓRIO JORNALÍSTICO (DSPY + GROQ)")
    reporter.header()

    # --- Verificar arquivos de entrada ---
    required = ["race_summary.json", "timeline.json", "driver_profiles.json"]
    missing = [f for f in required if not (timeline_dir / f).exists()]
    if missing:
        reporter.info(f"⚠️  Arquivos ausentes: {missing}. Pulando Fase 5.", indent=0)
        return timeline_dir

    reporter.section("5.1", "Carregando JSONs da Fase 4")
    race_summary = json.loads((timeline_dir / "race_summary.json").read_text(encoding="utf-8"))
    timeline = json.loads((timeline_dir / "timeline.json").read_text(encoding="utf-8"))
    driver_profiles = json.loads((timeline_dir / "driver_profiles.json").read_text(encoding="utf-8"))

    n_events = len(timeline) if isinstance(timeline, list) else "?"
    n_drivers = len(driver_profiles) if isinstance(driver_profiles, list) else "?"
    reporter.info(f"winner={race_summary.get('winner')} | {n_events} eventos | {n_drivers} pilotos")

    # --- ML-C: Carregar race_control para validação de incidente_multiplo ---
    race_control_list: list | None = None
    driver_quality_scores: dict | None = None

    if processed_dir is not None:
        # race_control como lista de dicts para narrative_context
        rc_file = processed_dir / "race_control_processed.parquet"
        if rc_file.exists():
            try:
                rc_df = pd.read_parquet(rc_file)
                race_control_list = rc_df.to_dict(orient="records")
                reporter.info(f"race_control carregado: {len(race_control_list)} mensagens")
            except Exception as exc:
                reporter.info(f"⚠️  Falha ao carregar race_control: {exc}", indent=0)

        # ML-07: driver_quality_scores salvo pela fase de eventos
        qs_file = timeline_dir / "driver_quality_scores.json"
        if qs_file.exists():
            try:
                driver_quality_scores = json.loads(qs_file.read_text(encoding="utf-8"))
                reporter.info(f"driver_quality_scores carregado: {len(driver_quality_scores)} pilotos")
            except Exception as exc:
                reporter.info(f"⚠️  Falha ao carregar driver_quality_scores: {exc}", indent=0)

    # --- Gerar relatório via DSPy ---
    reporter.section("5.2", "Gerando relatório via DSPy Predict (Groq llama-3.3-70b-versatile)")

    experiment_name = f"F1_{year}_LLM_Reports"
    run_name = f"report_y{year}_r{round_num:02d}"

    relatorio = generate_report(
        race_summary=race_summary,
        timeline=timeline,
        driver_profiles=driver_profiles,
        race_control=race_control_list,
        driver_quality_scores=driver_quality_scores,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    # --- Salvar relatorio.json + RELATORIO_CORRIDA.md ---
    reporter.section("5.3", "Salvando relatorio.json e RELATORIO_CORRIDA.md")
    out_path = timeline_dir / "relatorio.json"
    out_path.write_text(relatorio.model_dump_json(indent=2), encoding="utf-8")
    reporter.success(f"relatorio.json salvo: {out_path}", indent=0)

    # Markdown formatado para leitura — salvo na raiz do projeto
    project_root = Path(__file__).parent.parent.parent
    md_path = project_root / "RELATORIO_CORRIDA.md"
    md_path.write_text(relatorio.to_markdown(), encoding="utf-8")
    reporter.success(f"RELATORIO_CORRIDA.md salvo: {md_path}", indent=0)
    reporter.info(f"Total de palavras: {relatorio.word_count()}")

    return timeline_dir
