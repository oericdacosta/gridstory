#!/usr/bin/env python3
"""
PitWall AI - Pipeline Completo End-to-End.

Pipeline unificado que executa:
1. Extração completa de dados da corrida (laps, telemetry, race_control, weather, results)
2. Pré-processamento de TODOS os dados com NumPy, Pandas e SciPy
3. Machine Learning com Scikit-learn (clustering, anomaly detection) + Ruptures (tire cliffs)
4. Eventos Estruturados: classificação de causas, undercuts e geração de timeline.json (Pydantic)

Exemplo de uso:
    # Pipeline completo para uma corrida
    uv run python cli/pipeline.py 2025 1

    # Com polling (aguardar disponibilidade)
    uv run python cli/pipeline.py 2025 1 --polling

    # Mostrar amostras dos dados processados
    uv run python cli/pipeline.py 2025 1 --show-sample
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.pipeline_steps.reporting import print_pipeline_header, print_final_summary
from cli.pipeline_steps.extraction import run_extraction_phase
from cli.pipeline_steps.preprocessing import run_preprocessing_phase
from cli.pipeline_steps.ml import run_ml_phase
from cli.pipeline_steps.events import run_events_phase


def run_complete_pipeline(
    year: int,
    round_num: int,
    use_polling: bool = False,
    show_sample: bool = False,
):
    """
    Executa pipeline completo: extração + pré-processamento + ML + eventos estruturados.

    Args:
        year: Ano da temporada
        round_num: Número da rodada
        use_polling: Se deve aguardar disponibilidade dos dados
        show_sample: Se deve mostrar amostras dos dados processados
    """
    # Cabeçalho
    print_pipeline_header(year, round_num)

    # ========================================================================
    # FASE 1: EXTRAÇÃO DE DADOS
    # ========================================================================
    race_dir = run_extraction_phase(
        year=year,
        round_num=round_num,
        use_polling=use_polling,
    )

    # ========================================================================
    # FASE 2: PRÉ-PROCESSAMENTO
    # ========================================================================
    processed_dir = run_preprocessing_phase(
        race_dir=race_dir,
        year=year,
        round_num=round_num,
        show_sample=show_sample,
    )

    # ========================================================================
    # FASE 3: MACHINE LEARNING
    # ========================================================================
    ml_dir = run_ml_phase(
        processed_dir=processed_dir,
        year=year,
        round_num=round_num,
        show_sample=show_sample,
    )

    # ========================================================================
    # FASE 4: EVENTOS ESTRUTURADOS (PYDANTIC)
    # ========================================================================
    run_events_phase(
        ml_dir=ml_dir,
        processed_dir=processed_dir,
        year=year,
        round_num=round_num,
    )

    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    print_final_summary(race_dir, processed_dir, ml_dir)


def main():
    parser = argparse.ArgumentParser(
        description="PitWall AI - Pipeline Completo (Extração + Pré-processamento + ML)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "year",
        type=int,
        help="Ano da temporada (ex: 2025)",
    )

    parser.add_argument(
        "round",
        type=int,
        help="Número da rodada/corrida (ex: 1)",
    )

    parser.add_argument(
        "--polling",
        action="store_true",
        help="Usar polling para aguardar disponibilidade dos dados",
    )

    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Mostrar amostras dos dados processados",
    )

    args = parser.parse_args()

    try:
        run_complete_pipeline(
            year=args.year,
            round_num=args.round,
            use_polling=args.polling,
            show_sample=args.show_sample,
        )
    except Exception as e:
        print(f"\n❌ Erro durante pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
