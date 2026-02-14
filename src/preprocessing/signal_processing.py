"""
Processamento de sinal e redução de ruído usando scipy.signal.

Remove ruído de sensores e suaviza dados de telemetria preservando
características importantes como pontos de frenagem e zonas de aceleração.
"""

import numpy as np
from scipy.signal import medfilt, savgol_filter


def clean_signal(
    signal: np.ndarray,
    method: str = "median",
    kernel_size: int = 5,
    **kwargs,
) -> np.ndarray:
    """
    Limpa sinal de telemetria removendo ruído preservando bordas.

    Args:
        signal: Array de sinal de entrada (ex: velocidade, aceleração)
        method: Método de filtragem - "median" ou "savgol"
        kernel_size: Tamanho do kernel do filtro (deve ser ímpar para filtro mediano)
        **kwargs: Parâmetros adicionais para filtros específicos
                 - Para savgol: polyorder (padrão: 2)

    Returns:
        Array de sinal limpo

    Example:
        >>> clean_speed = clean_signal(raw_speed, method="median", kernel_size=5)
        >>> smooth_speed = clean_signal(raw_speed, method="savgol", kernel_size=11, polyorder=3)
    """
    if len(signal) == 0:
        return signal

    # Lidar com valores NaN
    if np.all(np.isnan(signal)):
        return signal

    # Para arrays com alguns NaN, interpolar antes de filtrar
    has_nan = np.any(np.isnan(signal))
    if has_nan:
        valid_indices = ~np.isnan(signal)
        if not np.any(valid_indices):
            return signal

        # Interpolação linear para valores NaN
        x = np.arange(len(signal))
        signal = np.interp(x, x[valid_indices], signal[valid_indices])

    if method == "median":
        # Filtro mediano: excelente para remover outliers preservando bordas
        if kernel_size % 2 == 0:
            kernel_size += 1  # Garantir tamanho ímpar do kernel

        cleaned = medfilt(signal, kernel_size=kernel_size)

    elif method == "savgol":
        # Filtro Savitzky-Golay: suaviza sinal preservando derivadas
        polyorder = kwargs.get("polyorder", 2)

        if kernel_size % 2 == 0:
            kernel_size += 1  # Garantir tamanho ímpar do kernel

        # Garantir que kernel_size > polyorder
        if kernel_size <= polyorder:
            kernel_size = polyorder + 2
            if kernel_size % 2 == 0:
                kernel_size += 1

        # Garantir pontos suficientes para o filtro
        if len(signal) < kernel_size:
            # Recuar para kernel menor ou sem filtragem
            if len(signal) >= 3:
                kernel_size = 3
                polyorder = 1
            else:
                return signal

        cleaned = savgol_filter(signal, kernel_size, polyorder)

    else:
        raise ValueError(f"Método desconhecido: {method}. Use 'median' ou 'savgol'")

    return cleaned


def calculate_derivative(
    signal: np.ndarray,
    delta_x: float = 1.0,
    smooth: bool = True,
    kernel_size: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Calcula derivada suave de um sinal (ex: aceleração a partir de velocidade).

    Args:
        signal: Array de sinal de entrada (ex: velocidade vs distância)
        delta_x: Espaçamento entre pontos do sinal (ex: passo de distância em metros)
        smooth: Se deve aplicar suavização Savitzky-Golay antes da derivada
        kernel_size: Tamanho da janela para filtro Savitzky-Golay
        polyorder: Ordem polinomial para filtro Savitzky-Golay

    Returns:
        Derivada do sinal

    Example:
        >>> # Calcular aceleração a partir de velocidade
        >>> acceleration = calculate_derivative(speed, delta_x=distance_step)
        >>> # Calcular jerk a partir de aceleração
        >>> jerk = calculate_derivative(acceleration, delta_x=distance_step)
    """
    if len(signal) < 2:
        return np.array([])

    if smooth:
        # Usar filtro Savitzky-Golay para derivada suave
        if kernel_size % 2 == 0:
            kernel_size += 1

        if kernel_size <= polyorder:
            kernel_size = polyorder + 2
            if kernel_size % 2 == 0:
                kernel_size += 1

        if len(signal) < kernel_size:
            # Recuar para derivada simples para sinais curtos
            derivative = np.gradient(signal, delta_x)
        else:
            # deriv=1 calcula a primeira derivada
            derivative = savgol_filter(
                signal, kernel_size, polyorder, deriv=1, delta=delta_x
            )
    else:
        # Gradiente simples
        derivative = np.gradient(signal, delta_x)

    return derivative


def remove_outliers(
    signal: np.ndarray,
    threshold: float = 3.0,
    method: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detecta e remove outliers do sinal usando métodos estatísticos.

    Args:
        signal: Array de sinal de entrada
        threshold: Número de desvios padrão (para 'zscore') ou
                  múltiplos de MAD (para 'median') para considerar como outlier
        method: Método de detecção - "zscore" ou "median" (MAD - Desvio Absoluto Mediano)

    Returns:
        Tupla de (sinal_limpo, máscara_outlier) onde máscara_outlier é True para outliers

    Example:
        >>> clean_data, outliers = remove_outliers(rpm_data, threshold=3.0)
        >>> print(f"Encontrados {outliers.sum()} outliers")
    """
    if len(signal) == 0:
        return signal, np.array([], dtype=bool)

    # Lidar com NaN
    valid_mask = ~np.isnan(signal)
    if not np.any(valid_mask):
        return signal, np.zeros_like(signal, dtype=bool)

    outlier_mask = np.zeros_like(signal, dtype=bool)

    if method == "zscore":
        # Método Z-score
        mean = np.nanmean(signal)
        std = np.nanstd(signal)

        if std == 0:
            return signal, outlier_mask

        z_scores = np.abs((signal - mean) / std)
        outlier_mask = z_scores > threshold

    elif method == "median":
        # Método MAD (Desvio Absoluto Mediano) - mais robusto a outliers
        median = np.nanmedian(signal)
        mad = np.nanmedian(np.abs(signal - median))

        if mad == 0:
            return signal, outlier_mask

        # Z-score modificado usando MAD
        modified_z_scores = 0.6745 * np.abs(signal - median) / mad
        outlier_mask = modified_z_scores > threshold

    else:
        raise ValueError(f"Método desconhecido: {method}. Use 'zscore' ou 'median'")

    # Substituir outliers com valores interpolados
    cleaned = signal.copy()
    if np.any(outlier_mask):
        valid_indices = ~outlier_mask
        x = np.arange(len(signal))

        if np.any(valid_indices):
            cleaned[outlier_mask] = np.interp(
                x[outlier_mask], x[valid_indices], signal[valid_indices]
            )

    return cleaned, outlier_mask


def apply_telemetry_pipeline(
    telemetry_dict: dict[str, np.ndarray],
    noise_reduction: bool = True,
    outlier_removal: bool = True,
    calculate_derivatives: bool = False,
) -> dict[str, np.ndarray]:
    """
    Aplica pipeline completo de processamento de sinal a canais de telemetria.

    Args:
        telemetry_dict: Dicionário de nomes de canais de telemetria para arrays de sinal
        noise_reduction: Se deve aplicar filtragem mediana para redução de ruído
        outlier_removal: Se deve detectar e remover outliers
        calculate_derivatives: Se deve calcular derivadas (ex: aceleração)

    Returns:
        Dicionário com telemetria processada e canais derivados opcionais

    Example:
        >>> telemetry = {
        ...     'Speed': speed_array,
        ...     'Throttle': throttle_array,
        ...     'Brake': brake_array,
        ... }
        >>> processed = apply_telemetry_pipeline(telemetry, calculate_derivatives=True)
        >>> # processed agora contém 'Speed', 'Speed_derivative', 'Throttle', etc.
    """
    processed = {}

    for channel_name, signal in telemetry_dict.items():
        current_signal = signal.copy()

        # Etapa 1: Remover outliers
        if outlier_removal:
            current_signal, _ = remove_outliers(
                current_signal, threshold=3.0, method="median"
            )

        # Etapa 2: Reduzir ruído
        if noise_reduction:
            current_signal = clean_signal(
                current_signal, method="median", kernel_size=5
            )

        processed[channel_name] = current_signal

        # Etapa 3: Calcular derivadas se solicitado
        if calculate_derivatives and channel_name in ["Speed", "Throttle", "Brake"]:
            derivative = calculate_derivative(current_signal, smooth=True)
            processed[f"{channel_name}_derivative"] = derivative

    return processed
