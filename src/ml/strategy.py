"""
Detecção de estratégias de corrida — Undercuts.

Um undercut ocorre quando um piloto para mais cedo que o rival à sua frente,
volta com pneus novos e, graças ao ritmo superior, sai à frente quando o
rival também para.

Sinal matemático detectado:
    1. Piloto A (atrás na pista) para N voltas antes de piloto B (à frente).
    2. Piloto A completa o stint nos boxes com LapTime < LapTime de B no mesmo lap.
    3. Quando B para, A sai à frente (ganho de posição confirmado).

Saída: DataFrame com uma linha por undercut confirmado, pronto para
ser consumido por build_race_timeline() → Pydantic → LLM.
"""

import pandas as pd


# Janela máxima de voltas entre pit stops para considerar um undercut
# (undercuts com > UNDERCUT_WINDOW voltas de diferença não são undercuts — são estratégias diferentes)
UNDERCUT_WINDOW = 4


def detect_undercuts(
    laps_df: pd.DataFrame,
    min_time_gained: float = 0.3,
) -> pd.DataFrame:
    """
    Detecta manobras de undercut em uma corrida.

    Algoritmo:
        Para cada par de pilotos (A, B) onde B está 1 posição à frente de A:
        1. Verifica se A teve um pit stop antes de B (dentro de UNDERCUT_WINDOW voltas).
        2. Verifica se após o pit stop de B, A está posicionado à frente de B.
        3. Estima o tempo ganho por A como o gap médio de LapTime entre eles
           nas voltas após o pit stop de B (voltas de exploração de pneus frescos).

    Args:
        laps_df:         DataFrame com Driver, LapNumber, Position, LapTime_seconds,
                         Stint (detectado pelo pipeline ML).
        min_time_gained: Mínimo de segundos de ganho para confirmar o undercut.
                         Filtra manobras onde a troca de posição foi negligenciável.

    Returns:
        DataFrame com colunas:
        - driver (str): Piloto que executou o undercut
        - target_driver (str): Piloto que sofreu o undercut
        - lap (int): Volta em que o undercut foi consumado (pit stop de target_driver)
        - time_gained_seconds (float): Ganho de tempo estimado
    """
    required = {"Driver", "LapNumber", "Position", "LapTime_seconds", "Stint"}
    missing = required - set(laps_df.columns)
    if missing:
        # Sem as colunas necessárias, retorna DataFrame vazio com o schema correto
        return pd.DataFrame(columns=["driver", "target_driver", "lap", "time_gained_seconds"])

    df = laps_df.copy()

    # Identificar voltas de pit stop: LapNumber onde o piloto troca de Stint
    pit_laps = _find_pit_laps(df)

    if pit_laps.empty:
        return pd.DataFrame(columns=["driver", "target_driver", "lap", "time_gained_seconds"])

    undercuts = []
    seen_pairs: set[tuple[str, str, int]] = set()

    # Para cada pit stop real de cada piloto A, verificar se é um undercut
    for _, pit_row in pit_laps.iterrows():
        driver_a = pit_row["Driver"]
        pit_lap_a = pit_row["pit_lap"]

        # Posição de A antes de parar (volta anterior ao pit)
        pos_a_before = _get_position(df, driver_a, pit_lap_a - 1)
        if pos_a_before is None or pos_a_before <= 1:
            continue  # Líder não faz undercut

        # Candidatos a alvo: QUALQUER piloto que estava à frente de A antes do pit
        # (não apenas o imediatamente adjacente — undercuts podem acontecer saltando posições)
        ahead_drivers = _get_drivers_ahead(df, driver_a, pit_lap_a - 1, pos_a_before)

        for target_driver in ahead_drivers:
            pair_key = (driver_a, target_driver, pit_lap_a)
            if pair_key in seen_pairs:
                continue

            # Verificar se B (target) parou entre pit_lap_a + 1 e pit_lap_a + UNDERCUT_WINDOW
            pit_b_rows = pit_laps[
                (pit_laps["Driver"] == target_driver) &
                (pit_laps["pit_lap"] > pit_lap_a) &
                (pit_laps["pit_lap"] <= pit_lap_a + UNDERCUT_WINDOW)
            ]

            if pit_b_rows.empty:
                continue  # B não parou na janela de undercut

            pit_lap_b = int(pit_b_rows.iloc[0]["pit_lap"])

            # Verificar se A está à frente de B após o pit stop de B
            pos_a_after = _get_position(df, driver_a, pit_lap_b + 1)
            pos_b_after = _get_position(df, target_driver, pit_lap_b + 1)

            if pos_a_after is None or pos_b_after is None:
                continue

            if pos_a_after >= pos_b_after:
                continue  # Não houve troca de posição — não é undercut

            # Estimar tempo ganho: diferença de LapTime na janela entre os dois pits
            # (A com pneus novos vs B ainda em pneus velhos)
            time_gained = _estimate_time_gained(df, driver_a, target_driver, pit_lap_a, pit_lap_b)

            if time_gained < min_time_gained:
                continue  # Ganho negligenciável

            seen_pairs.add(pair_key)
            undercuts.append({
                "driver": driver_a,
                "target_driver": target_driver,
                "lap": pit_lap_b,
                "time_gained_seconds": round(time_gained, 3),
            })

    return pd.DataFrame(undercuts) if undercuts else pd.DataFrame(
        columns=["driver", "target_driver", "lap", "time_gained_seconds"]
    )


def _find_pit_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica as voltas onde cada piloto fez um pit stop real.

    Um pit stop real é identificado pelo reset de TyreLife: o piloto sai dos boxes
    com TyreLife baixo (1-3) enquanto a volta anterior tinha TyreLife alto (>5).

    Descarta artefatos do SC/VSC inicial (voltas 1-5) onde o FastF1 atribui
    múltiplos stints sem TyreLife resetar — o TyreLife cresce continuamente
    (1,2,3,4,5...) sem retornar a 1, sinalizando que não houve troca de pneus.

    Requer coluna 'TyreLife' no DataFrame.
    """
    has_tyre = "TyreLife" in df.columns

    rows = []
    for driver, driver_df in df.groupby("Driver"):
        driver_df = driver_df.sort_values("LapNumber").reset_index(drop=True)
        stint_changes_idx = driver_df.index[driver_df["Stint"].diff() > 0].tolist()

        for idx in stint_changes_idx:
            lap = int(driver_df.loc[idx, "LapNumber"])

            if has_tyre:
                tyre_new = driver_df.loc[idx, "TyreLife"]
                tyre_prev = driver_df.loc[idx - 1, "TyreLife"] if idx > 0 else None

                # Pit stop real: TyreLife reseta (vai de alto para 1/2/3)
                # Artefato SC: TyreLife continua subindo (ex: 3→4, 4→5)
                if tyre_prev is not None and tyre_new > tyre_prev:
                    # TyreLife cresceu — não houve troca de pneus, é artefato SC
                    continue
                if tyre_new > 5:
                    # TyreLife alto na "entrada" de um novo stint → não é pit real
                    continue

            # Excluir pit stops nas primeiras voltas (< 8): SC/VSC inicial,
            # formação e arranque — sem valor estratégico para undercuts
            if lap < 8:
                continue

            rows.append({"Driver": driver, "pit_lap": lap})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Driver", "pit_lap"])


def _get_position(df: pd.DataFrame, driver: str, lap: int) -> int | None:
    """Retorna a posição de um piloto em uma volta específica."""
    row = df[(df["Driver"] == driver) & (df["LapNumber"] == lap)]
    if row.empty or row["Position"].isna().all():
        return None
    return int(row["Position"].iloc[0])


def _get_drivers_ahead(
    df: pd.DataFrame,
    driver_a: str,
    lap: int,
    pos_a: int,
) -> list[str]:
    """
    Retorna lista de pilotos que estavam à frente de A em uma volta.

    Limita a busca a pilotos nas 6 posições à frente de A para evitar
    que o algoritmo tente undercuts de P16 contra P1 (irrealistas).
    """
    max_pos_diff = 6
    lap_data = df[df["LapNumber"] == lap]
    ahead = lap_data[
        (lap_data["Position"] < pos_a) &
        (lap_data["Position"] >= pos_a - max_pos_diff) &
        (lap_data["Driver"] != driver_a)
    ]
    return ahead["Driver"].tolist()


def _get_driver_at_position(df: pd.DataFrame, lap: int, position: int) -> str | None:
    """Retorna o piloto que está em uma determinada posição em uma volta."""
    row = df[(df["LapNumber"] == lap) & (df["Position"] == position)]
    if row.empty:
        return None
    return str(row["Driver"].iloc[0])


def _estimate_time_gained(
    df: pd.DataFrame,
    driver_a: str,
    driver_b: str,
    pit_lap_a: int,
    pit_lap_b: int,
) -> float:
    """
    Estima o tempo ganho pelo undercut.

    O ganho real ocorre DURANTE a janela entre o pit de A e o pit de B:
    A está em pneus novos (rápido), B ainda em pneus velhos (lento).
    Compara médias de LapTime_seconds nessa janela.

    Se a janela for curta (< 2 laps), usa a diferença de LapTime médio geral
    dos dois pilotos como proxy.
    """
    # Voltas entre os dois pits: A em pneus frescos, B em pneus velhos
    gap_laps = list(range(pit_lap_a + 1, pit_lap_b))

    if len(gap_laps) >= 2:
        times_a = df[(df["Driver"] == driver_a) & (df["LapNumber"].isin(gap_laps))]["LapTime_seconds"]
        times_b = df[(df["Driver"] == driver_b) & (df["LapNumber"].isin(gap_laps))]["LapTime_seconds"]

        if not times_a.empty and not times_b.empty:
            gain_per_lap = times_b.mean() - times_a.mean()
            total_gain = gain_per_lap * len(gap_laps)
            return float(max(total_gain, 0.0))

    # Fallback: diferença entre LapTime médio geral dos pilotos (proxy conservador)
    all_a = df[df["Driver"] == driver_a]["LapTime_seconds"].dropna()
    all_b = df[df["Driver"] == driver_b]["LapTime_seconds"].dropna()
    if all_a.empty or all_b.empty:
        return 0.0
    return float(max(all_b.median() - all_a.median(), 0.0))
