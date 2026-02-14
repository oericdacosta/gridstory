"""
Sincronização de telemetria usando scipy.interpolate.

Alinha dados de telemetria de diferentes pilotos em um grid comum de distância,
permitindo comparações diretas e cálculos de delta.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


def synchronize_telemetry(
    telemetry: pd.DataFrame,
    track_length: float,
    num_points: int = 5000,
    telemetry_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Sincroniza dados de telemetria em um grid comum de distância usando interpolação cúbica spline.

    Args:
        telemetry: DataFrame de telemetria bruta com coluna 'Distance' e canais de telemetria
        track_length: Comprimento total da pista em metros
        num_points: Número de pontos no grid sincronizado (padrão: 5000)
        telemetry_columns: Lista de colunas de telemetria para interpolar.
                          Se None, usa ['Speed', 'RPM', 'Throttle', 'Brake', 'nGear', 'DRS']

    Returns:
        DataFrame com telemetria sincronizada em um grid uniforme de distância

    Example:
        >>> telemetry = lap.get_telemetry()
        >>> synchronized = synchronize_telemetry(telemetry, track_length=5281.0)
    """
    if telemetry_columns is None:
        telemetry_columns = ["Speed", "RPM", "Throttle", "Brake", "nGear", "DRS"]

    # Filtrar apenas colunas que existem no DataFrame de telemetria
    available_columns = [col for col in telemetry_columns if col in telemetry.columns]

    if "Distance" not in telemetry.columns:
        raise ValueError("DataFrame de telemetria deve conter coluna 'Distance'")

    if len(available_columns) == 0:
        raise ValueError(
            f"Nenhuma das colunas de telemetria solicitadas {telemetry_columns} encontrada no DataFrame"
        )

    # Criar grid comum de distância
    dist_grid = np.linspace(0, track_length, num=num_points)

    # Preparar DataFrame de saída
    synchronized_data = {"Distance": dist_grid}

    # Obter valores de distância originais (ordenados e sem NaN)
    original_distance = telemetry["Distance"].values
    valid_indices = ~np.isnan(original_distance)

    if not np.any(valid_indices):
        raise ValueError("Nenhum valor de distância válido encontrado na telemetria")

    original_distance = original_distance[valid_indices]

    # Garantir que a distância seja estritamente crescente para interpolação
    if not np.all(np.diff(original_distance) > 0):
        # Ordenar e remover duplicatas
        sort_indices = np.argsort(original_distance)
        original_distance = original_distance[sort_indices]

        # Obter valores únicos
        unique_indices = np.concatenate([[True], np.diff(original_distance) > 0])
        original_distance = original_distance[unique_indices]

    # Interpolar cada canal de telemetria
    for column in available_columns:
        original_values = telemetry[column].values[valid_indices]

        # Aplicar mesma ordenação da distância
        if not np.all(np.diff(telemetry["Distance"].values[valid_indices]) > 0):
            sort_indices = np.argsort(telemetry["Distance"].values[valid_indices])
            original_values = original_values[sort_indices]
            original_values = original_values[unique_indices]

        # Remover valores NaN do canal de telemetria
        valid_values = ~np.isnan(original_values)
        if not np.any(valid_values):
            # Se todos os valores são NaN, preencher com zeros
            synchronized_data[column] = np.zeros_like(dist_grid)
            continue

        channel_distance = original_distance[valid_values]
        channel_values = original_values[valid_values]

        # Verificar se temos pontos suficientes para spline cúbica
        if len(channel_distance) < 4:
            # Recuar para interpolação linear para dados esparsos
            synchronized_data[column] = np.interp(
                dist_grid, channel_distance, channel_values
            )
        else:
            # Usar spline cúbica para interpolação suave
            spline = make_interp_spline(channel_distance, channel_values, k=3)

            # Recortar grid para intervalo válido para evitar extrapolação
            valid_grid = dist_grid[
                (dist_grid >= channel_distance.min())
                & (dist_grid <= channel_distance.max())
            ]

            if len(valid_grid) == 0:
                synchronized_data[column] = np.zeros_like(dist_grid)
            else:
                interpolated = spline(valid_grid)

                # Preencher grid completo com valores extrapolados nas fronteiras
                full_values = np.zeros_like(dist_grid)
                valid_mask = (dist_grid >= channel_distance.min()) & (
                    dist_grid <= channel_distance.max()
                )
                full_values[valid_mask] = interpolated

                # Preencher fora do intervalo com valores das fronteiras
                full_values[dist_grid < channel_distance.min()] = channel_values[0]
                full_values[dist_grid > channel_distance.max()] = channel_values[-1]

                synchronized_data[column] = full_values

    return pd.DataFrame(synchronized_data)


def synchronize_multiple_laps(
    laps: pd.DataFrame,
    track_length: float,
    num_points: int = 5000,
    telemetry_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Sincroniza telemetria para múltiplas voltas, criando uma matriz onde cada linha é uma volta.

    Args:
        laps: DataFrame contendo dados de voltas com telemetria
        track_length: Comprimento total da pista em metros
        num_points: Número de pontos no grid sincronizado
        telemetry_columns: Lista de colunas de telemetria para interpolar

    Returns:
        DataFrame onde cada linha representa uma volta com telemetria sincronizada.
        Colunas incluem metadados da volta e canais de telemetria achatados.

    Example:
        >>> laps = session.laps.pick_driver('VER').pick_quicklaps()
        >>> synchronized_matrix = synchronize_multiple_laps(laps, track_length=5281.0)
    """
    synchronized_laps = []

    for idx, lap in laps.iterrows():
        try:
            telemetry = lap.get_telemetry()

            if telemetry is None or len(telemetry) == 0:
                continue

            synchronized = synchronize_telemetry(
                telemetry, track_length, num_points, telemetry_columns
            )

            # Criar uma linha com metadados da volta
            lap_data = {
                "LapNumber": lap.get("LapNumber", idx),
                "LapTime": lap.get("LapTime", pd.NaT),
                "Driver": lap.get("Driver", "Unknown"),
                "Compound": lap.get("Compound", "Unknown"),
                "TyreLife": lap.get("TyreLife", 0),
            }

            # Adicionar telemetria sincronizada como colunas achatadas
            for column in synchronized.columns:
                if column != "Distance":
                    lap_data[f"{column}_sync"] = [synchronized[column].values]

            synchronized_laps.append(lap_data)

        except Exception:
            # Pular voltas que não podem ser sincronizadas
            continue

    if len(synchronized_laps) == 0:
        raise ValueError("Nenhuma volta pôde ser sincronizada")

    return pd.DataFrame(synchronized_laps)
