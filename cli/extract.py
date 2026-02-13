"""
Plataforma de Inteligência de Corridas F1 - CLI de Extração de Dados.

Este script fornece interface de linha de comando para extrair dados de
corridas de Fórmula 1 usando a biblioteca FastF1.

Exemplos de uso:
    # Extrair calendário da temporada
    uv run python cli/extract.py --calendar 2025

    # Extrair uma corrida específica
    uv run python cli/extract.py --race 2025 1

    # Extrair com telemetria completa
    uv run python cli/extract.py --race 2025 1 --telemetry

    # Extrair múltiplas corridas
    uv run python cli/extract.py --batch 2025 "1,2,3,4,5"
"""

import argparse
import sys
from pathlib import Path

# Adicionar diretório raiz ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.orchestrator import (
    extract_race_complete,
    extract_season_calendar,
    extract_multiple_races,
)


def main():
    parser = argparse.ArgumentParser(
        description="F1 Race Intelligence Platform - Extração de Dados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Comandos principais
    parser.add_argument(
        "--calendar",
        type=int,
        metavar="YEAR",
        help="Extrair calendário da temporada (ex: --calendar 2025)",
    )

    parser.add_argument(
        "--race",
        nargs=2,
        type=int,
        metavar=("YEAR", "ROUND"),
        help="Extrair dados de uma corrida (ex: --race 2025 1)",
    )

    parser.add_argument(
        "--batch",
        nargs=2,
        metavar=("YEAR", "ROUNDS"),
        help='Extrair múltiplas corridas (ex: --batch 2025 "1,2,3,4,5")',
    )

    # Opções adicionais
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Incluir telemetria completa de todos os pilotos",
    )

    parser.add_argument(
        "--polling",
        action="store_true",
        help="Usar modo polling (aguardar disponibilidade dos dados)",
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
    print(f"Cache FastF1 habilitado: {cache_dir}\n")

    # Executar comandos
    if args.calendar:
        extract_season_calendar(year=args.calendar)

    elif args.race:
        year, round_num = args.race
        extract_race_complete(
            year=year,
            round_number=round_num,
            use_polling=args.polling,
            save_telemetry=args.telemetry,
            output_dir=args.output_dir,
        )

    elif args.batch:
        year = int(args.batch[0])
        rounds_str = args.batch[1]
        round_numbers = [int(r.strip()) for r in rounds_str.split(",")]

        extract_multiple_races(
            year=year,
            round_numbers=round_numbers,
            save_telemetry=args.telemetry,
            output_dir=args.output_dir,
        )

    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("EXEMPLOS DE USO:")
        print("=" * 70)
        print("\n1. Extrair calendário da temporada 2025:")
        print("   python main.py --calendar 2025")
        print("\n2. Extrair primeira corrida de 2025:")
        print("   python main.py --race 2025 1")
        print("\n3. Extrair com telemetria completa:")
        print("   python main.py --race 2025 1 --telemetry")
        print("\n4. Extrair primeiras 5 corridas:")
        print("   python main.py --batch 2025 1,2,3,4,5")
        print("\n5. Modo polling (aguardar dados):")
        print("   python main.py --race 2026 1 --polling")
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
