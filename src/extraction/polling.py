"""
Módulo de polling para verificação de disponibilidade de dados.

Implementa sistema de tentativas para aguardar disponibilidade dos dados do FastF1.
"""

import fastf1
import time
from typing import Optional


def ingest_race_data(
    year: int, round_number: int, max_retries: int = 10, retry_interval: int = 300
) -> Optional[fastf1.core.Session]:
    """
    Script de polling que aguarda a disponibilidade dos dados da corrida.

    Este é o "Vigia" que verifica periodicamente se os dados já estão
    disponíveis na API do FastF1.

    Args:
        year: Ano da temporada
        round_number: Número da rodada (corrida)
        max_retries: Número máximo de tentativas
        retry_interval: Intervalo entre tentativas em segundos (padrão: 5 min)

    Returns:
        Objeto Session carregado com todos os dados, ou None se falhar
    """
    print(f"\n{'=' * 60}")
    print(f"Iniciando ingestão de dados: {year} - Round {round_number}")
    print(f"{'=' * 60}\n")

    # Passo 1: Instanciar a Sessão
    try:
        session = fastf1.get_session(year, round_number, "Race")
        print(f"Sessão instanciada: {session.event['EventName']}")
        print(f"Local: {session.event['Location']}, {session.event['Country']}")
    except Exception as e:
        print(f"Erro ao instanciar sessão: {e}")
        return None

    # Passo 2: Loop de verificação de disponibilidade
    data_loaded = False
    retry_count = 0

    while not data_loaded and retry_count < max_retries:
        try:
            print(f"\nTentativa {retry_count + 1}/{max_retries}...")
            print("Carregando: Laps, Telemetry, Weather, Messages...")

            # Tenta carregar dados completos
            session.load(laps=True, telemetry=True, weather=True, messages=True)

            data_loaded = True
            print("\n✓ Dados carregados com sucesso!")
            print(f"  - Voltas carregadas: {len(session.laps)}")
            print(f"  - Pilotos: {len(session.drivers)}")
            print(f"  - Mensagens de controle: {len(session.race_control_messages)}")

        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"\n✗ Dados ainda não disponíveis.")
                print(f"  Erro: {e}")
                print(f"  Aguardando {retry_interval}s antes da próxima tentativa...")
                time.sleep(retry_interval)
            else:
                print(f"\n✗ Falha após {max_retries} tentativas.")
                print(f"  Último erro: {e}")

    if not data_loaded:
        raise Exception("Falha ao obter dados após múltiplas tentativas.")

    return session


def quick_load_session(year: int, round_number: int) -> fastf1.core.Session:
    """
    Carrega dados de uma sessão sem polling (para dados já disponíveis).
    Útil para testes com corridas anteriores.

    Args:
        year: Ano da temporada
        round_number: Número da rodada

    Returns:
        Objeto Session carregado
    """
    print(f"Carregamento rápido: {year} - Round {round_number}")
    session = fastf1.get_session(year, round_number, "Race")
    session.load(laps=True, telemetry=True, weather=True, messages=True)
    print(f"✓ Sessão carregada: {session.event['EventName']}")
    return session
