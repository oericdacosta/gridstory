"""
Módulo de extração de dados para o pipeline.

Gerencia a Fase 1: Extração completa de dados da corrida usando FastF1.
"""

from pathlib import Path

import fastf1

from src.extraction.orchestrator import extract_race_complete
from .reporting import Reporter


def run_extraction_phase(
    year: int,
    round_num: int,
    use_polling: bool = False,
    output_dir: str = "data/raw/races",
) -> Path:
    """
    Executa a fase de extração de dados.

    Args:
        year: Ano da temporada
        round_num: Número da rodada
        use_polling: Se deve aguardar disponibilidade dos dados
        output_dir: Diretório de saída para dados brutos

    Returns:
        Path para o diretório com os dados extraídos
    """
    reporter = Reporter("FASE 1: EXTRAÇÃO DE DADOS")
    reporter.header()

    # Configurar cache do FastF1
    reporter.step("1", "Configurando cache FastF1")
    cache_dir = Path.home() / ".cache" / "fastf1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    reporter.info(f"Cache habilitado: {cache_dir}", indent=2)

    # Extrair dados
    reporter.step("2", "Extraindo dados da corrida")
    race_dir = extract_race_complete(
        year=year,
        round_number=round_num,
        use_polling=use_polling,
        output_dir=output_dir,
    )

    reporter.success(f"Extração concluída: {race_dir}")

    return race_dir
