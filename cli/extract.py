"""
PitWall AI - CLI de Extração de Dados F1.

Extrai TODOS os dados de uma corrida de Fórmula 1:
- Laps (voltas e estratégia)
- Telemetria completa (todos os pilotos)
- Race Control (safety car, bandeiras, penalidades)
- Weather (condições meteorológicas)
- Results (classificação final)

Exemplo de uso:
    # Extrair corrida completa
    uv run python cli/extract.py 2025 1

    # Extrair com polling (aguardar disponibilidade)
    uv run python cli/extract.py 2025 1 --polling
"""

import argparse
import sys
from pathlib import Path

# Adicionar diretório raiz ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.orchestrator import extract_race_complete


def main():
    parser = argparse.ArgumentParser(
        description="PitWall AI - Extração Completa de Dados F1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Extrair primeira corrida de 2025 (todos os dados)
  uv run python cli/extract.py 2025 1

  # Extrair com polling (aguardar disponibilidade dos dados)
  uv run python cli/extract.py 2025 1 --polling

  # Especificar diretório de saída
  uv run python cli/extract.py 2025 1 --output-dir data/raw/races

Dados extraídos (SEMPRE):
  ✓ Laps (voltas e estratégia)
  ✓ Telemetria (todos os pilotos)
  ✓ Race Control (safety car, bandeiras)
  ✓ Weather (condições meteorológicas)
  ✓ Results (classificação final)
        """,
    )

    # Argumentos posicionais obrigatórios
    parser.add_argument(
        "year",
        type=int,
        help="Ano da temporada (ex: 2025)",
    )

    parser.add_argument(
        "round",
        type=int,
        help="Número da rodada/corrida (ex: 1 para primeira corrida)",
    )

    # Opções
    parser.add_argument(
        "--polling",
        action="store_true",
        help="Usar modo polling para aguardar disponibilidade dos dados",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/races",
        help="Diretório de saída (padrão: data/raw/races)",
    )

    args = parser.parse_args()

    # Habilitar cache do FastF1
    import fastf1

    cache_dir = Path.home() / ".cache" / "fastf1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    print(f"📦 Cache FastF1: {cache_dir}\n")

    # Executar extração
    try:
        extract_race_complete(
            year=args.year,
            round_number=args.round,
            use_polling=args.polling,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"\n❌ Erro durante extração: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
