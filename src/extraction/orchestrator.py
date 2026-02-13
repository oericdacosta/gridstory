"""
Orquestrador principal para extração de dados de F1.

Coordena o processo completo de extração de dados, incluindo polling,
ETL e salvamento em arquivos Parquet.
"""

from src.extraction.calendar import get_season_schedule, get_next_race, save_schedule
from src.extraction.polling import ingest_race_data, quick_load_session
from src.extraction.etl import RaceDataETL


def extract_race_complete(
    year: int,
    round_number: int,
    use_polling: bool = False,
    save_telemetry: bool = False,
    output_dir: str = "data/raw/races",
):
    """
    Pipeline completo de extração de dados de uma corrida.

    Args:
        year: Ano da temporada
        round_number: Número da rodada (corrida)
        use_polling: Se True, usa polling para aguardar disponibilidade.
                    Se False, carrega diretamente (para dados já disponíveis)
        save_telemetry: Se True, extrai e salva telemetria de todos os pilotos
        output_dir: Diretório de saída dos dados

    Returns:
        Caminho do diretório com os dados salvos
    """
    print(f"\n{'=' * 70}")
    print(f"PIPELINE DE EXTRAÇÃO: {year} - Round {round_number}")
    print(f"{'=' * 70}\n")

    # Passo 1: Carregar sessão (com ou sem polling)
    if use_polling:
        print("Modo: POLLING (aguardando disponibilidade)")
        session = ingest_race_data(year, round_number)
    else:
        print("Modo: CARREGAMENTO DIRETO (dados já disponíveis)")
        session = quick_load_session(year, round_number)

    if session is None:
        raise Exception("Falha ao carregar sessão")

    # Passo 2: ETL - Extração e Transformação
    etl = RaceDataETL(session)
    data = etl.extract_all(save_telemetry=save_telemetry)

    # Passo 3: Salvar dados
    race_dir = etl.save_to_parquet(data, output_dir=output_dir)

    print(f"\n{'=' * 70}")
    print(f"✓ PIPELINE CONCLUÍDO COM SUCESSO")
    print(f"  Dados salvos em: {race_dir}")
    print(f"{'=' * 70}\n")

    return race_dir


def extract_season_calendar(year: int, output_dir: str = "data/raw/calendar"):
    """
    Extrai e salva o calendário completo da temporada.

    Args:
        year: Ano da temporada
        output_dir: Diretório de saída

    Returns:
        Caminho do arquivo de calendário
    """
    print(f"\n{'=' * 70}")
    print(f"EXTRAÇÃO DE CALENDÁRIO: Temporada {year}")
    print(f"{'=' * 70}\n")

    schedule = get_season_schedule(year, include_testing=False)
    file_path = save_schedule(schedule, output_dir=output_dir)

    # Mostrar resumo
    print(f"\nResumo do calendário:")
    print(f"  Total de eventos: {len(schedule)}")
    print(f"\nPrimeiras 5 corridas:")
    for idx, row in schedule.head(5).iterrows():
        print(f"  {row['RoundNumber']:02d}. {row['EventName']} - {row['Location']}")

    # Mostrar próxima corrida
    next_race = get_next_race(schedule)
    if next_race:
        print(f"\nPróxima corrida:")
        print(f"  Round {next_race['round_number']}: {next_race['event_name']}")
        print(f"  Local: {next_race['location']}, {next_race['country']}")
        print(f"  Data: {next_race['race_date']}")

    return file_path


def extract_multiple_races(
    year: int,
    round_numbers: list,
    save_telemetry: bool = False,
    output_dir: str = "data/raw/races",
):
    """
    Extrai dados de múltiplas corridas em sequência.

    Args:
        year: Ano da temporada
        round_numbers: Lista de números de rodadas
        save_telemetry: Se True, extrai telemetria
        output_dir: Diretório de saída

    Returns:
        Lista de caminhos dos diretórios criados
    """
    print(f"\n{'=' * 70}")
    print(f"EXTRAÇÃO EM LOTE: {len(round_numbers)} corridas da temporada {year}")
    print(f"{'=' * 70}\n")

    results = []

    for round_num in round_numbers:
        try:
            race_dir = extract_race_complete(
                year=year,
                round_number=round_num,
                use_polling=False,
                save_telemetry=save_telemetry,
                output_dir=output_dir,
            )
            results.append({"round": round_num, "status": "success", "path": race_dir})
            print(f"\n✓ Round {round_num} concluído\n")

        except Exception as e:
            print(f"\n✗ Erro no Round {round_num}: {e}\n")
            results.append({"round": round_num, "status": "error", "error": str(e)})

    # Resumo final
    print(f"\n{'=' * 70}")
    print("RESUMO DA EXTRAÇÃO EM LOTE")
    print(f"{'=' * 70}")

    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

    print(f"  ✓ Sucessos: {success_count}/{len(round_numbers)}")
    print(f"  ✗ Erros: {error_count}/{len(round_numbers)}")

    if error_count > 0:
        print(f"\nRounds com erro:")
        for r in results:
            if r["status"] == "error":
                print(f"  - Round {r['round']}: {r['error']}")

    print(f"\n{'=' * 70}\n")

    return results
