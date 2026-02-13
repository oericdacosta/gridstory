"""
Módulo para mapeamento do calendário de F1.

Fornece funções para carregar, processar e salvar calendários de temporadas.
"""
import fastf1
import pandas as pd
from pathlib import Path
from datetime import datetime


def get_season_schedule(year: int, include_testing: bool = False) -> pd.DataFrame:
    """
    Carrega o calendário da temporada de F1.

    Args:
        year: Ano da temporada
        include_testing: Se True, inclui sessões de teste

    Returns:
        DataFrame com o calendário completo (EventDate, Session5DateUtc, etc.)
    """
    print(f"Carregando calendário da temporada {year}...")
    schedule = fastf1.get_event_schedule(year, include_testing=include_testing)
    print(f"Calendário carregado: {len(schedule)} eventos encontrados.")
    return schedule


def get_next_race(schedule: pd.DataFrame) -> dict:
    """
    Identifica a próxima corrida com base na data atual.

    Args:
        schedule: DataFrame com o calendário da temporada

    Returns:
        Dicionário com informações da próxima corrida
    """
    now = pd.Timestamp.now(tz='UTC')
    upcoming_races = schedule[schedule['Session5DateUtc'] > now]

    if len(upcoming_races) == 0:
        print("Nenhuma corrida futura encontrada.")
        return None

    next_race = upcoming_races.iloc[0]
    return {
        'round_number': next_race['RoundNumber'],
        'event_name': next_race['EventName'],
        'country': next_race['Country'],
        'location': next_race['Location'],
        'race_date': next_race['Session5DateUtc'],
        'event_date': next_race['EventDate']
    }


def save_schedule(schedule: pd.DataFrame, output_dir: str = "data/raw/calendar"):
    """
    Salva o calendário em formato Parquet.

    Args:
        schedule: DataFrame com o calendário
        output_dir: Diretório de saída
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"schedule_{datetime.now().year}.parquet"
    schedule.to_parquet(file_path, index=False)
    print(f"Calendário salvo em: {file_path}")
    return file_path
