"""
Orquestrador principal para extração de dados de F1.

Coordena o processo completo de extração de dados, incluindo polling,
ETL e salvamento em arquivos Parquet.
"""

from .polling import ingest_race_data, quick_load_session
from .etl import RaceDataETL


def extract_race_complete(
    year: int,
    round_number: int,
    use_polling: bool = False,
    output_dir: str = "data/raw/races",
):
    """
    Pipeline completo de extração de dados de uma corrida.

    SEMPRE extrai TODOS os dados:
    - Laps (voltas e estratégia)
    - Telemetria completa (todos os pilotos)
    - Race Control (safety car, bandeiras, penalidades)
    - Weather (condições meteorológicas)
    - Results (classificação final)

    Args:
        year: Ano da temporada
        round_number: Número da rodada (corrida)
        use_polling: Se True, usa polling para aguardar disponibilidade.
                    Se False, carrega diretamente (para dados já disponíveis)
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

    # Passo 2: ETL - Extração completa de TODOS os dados
    print("\n⚙️  Extraindo TODOS os dados da corrida:")
    print("   • Laps (voltas e estratégia)")
    print("   • Telemetria (todos os pilotos)")
    print("   • Race Control (safety car, bandeiras)")
    print("   • Weather (condições meteorológicas)")
    print("   • Results (classificação final)\n")

    etl = RaceDataETL(session)
    data = etl.extract_all()

    # Passo 3: Salvar dados
    race_dir = etl.save_to_parquet(data, output_dir=output_dir)

    print(f"\n{'=' * 70}")
    print(f"✓ PIPELINE CONCLUÍDO COM SUCESSO")
    print(f"  Dados salvos em: {race_dir}")
    print(f"{'=' * 70}\n")

    return race_dir
