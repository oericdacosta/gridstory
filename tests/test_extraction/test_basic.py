"""
Script de exemplo para testar a extração de dados.
Extrai dados da primeira corrida de 2025 para validação.
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.orchestrator import extract_race_complete
import fastf1


def test_basic_extraction():
    """Teste básico: extrair primeira corrida de 2025 sem telemetria"""
    print("\n" + "=" * 70)
    print("TESTE 1: Extração Básica (Primeira corrida 2025)")
    print("=" * 70 + "\n")

    # Habilitar cache
    cache_dir = Path.home() / ".cache" / "fastf1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    try:
        race_dir = extract_race_complete(
            year=2025, round_number=1, use_polling=False, save_telemetry=False
        )
        print(f"\n✓ TESTE 1 PASSOU: Dados salvos em {race_dir}")
        return True
    except Exception as e:
        print(f"\n✗ TESTE 1 FALHOU: {e}")
        return False


def test_with_telemetry():
    """Teste com telemetria: extrair primeira corrida com dados de telemetria"""
    print("\n" + "=" * 70)
    print("TESTE 2: Extração com Telemetria")
    print("=" * 70 + "\n")

    # Habilitar cache
    cache_dir = Path.home() / ".cache" / "fastf1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    try:
        race_dir = extract_race_complete(
            year=2025, round_number=1, use_polling=False, save_telemetry=True
        )
        print(f"\n✓ TESTE 2 PASSOU: Dados com telemetria salvos em {race_dir}")
        return True
    except Exception as e:
        print(f"\n✗ TESTE 2 FALHOU: {e}")
        return False


def test_calendar():
    """Teste de calendário: extrair calendário completo de 2025"""
    print("\n" + "=" * 70)
    print("TESTE 3: Extração de Calendário")
    print("=" * 70 + "\n")

    from src.extraction.orchestrator import extract_season_calendar

    try:
        calendar_file = extract_season_calendar(year=2025)
        print(f"\n✓ TESTE 3 PASSOU: Calendário salvo em {calendar_file}")
        return True
    except Exception as e:
        print(f"\n✗ TESTE 3 FALHOU: {e}")
        return False


def main():
    """Executar todos os testes"""
    print("\n" + "=" * 70)
    print("SUITE DE TESTES DE EXTRAÇÃO DE DADOS F1")
    print("=" * 70)

    results = []

    # Teste 1: Extração básica
    results.append(("Extração Básica", test_basic_extraction()))

    # Teste 2: Extração com telemetria (comentado por padrão - demora mais)
    # results.append(("Extração com Telemetria", test_with_telemetry()))

    # Teste 3: Calendário
    results.append(("Calendário", test_calendar()))

    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO DOS TESTES")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASSOU" if passed else "✗ FALHOU"
        print(f"  {test_name}: {status}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} testes passaram")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
