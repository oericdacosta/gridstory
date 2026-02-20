"""
Detecção de change points em degradação de pneus usando Ruptures (PELT).

Detecta o ponto exato de mudança de regime de degradação do pneu ("tire cliff")
dentro de cada stint, usando PELT (Pruned Exact Linear Time).

Papéis do output do Isolation Forest neste módulo:
- Sinal primário: LapTime_delta — processado pelo PELT
- Filtro: is_anomaly — laps anômalos removidos do sinal antes do fit() para
  evitar falsos breakpoints por incidentes, SC, pit stops

A validação usa slope (inclinação) do LapTime_delta pré-cliff, não anomaly_score.
Razão: anomaly_score detecta outliers pontuais; degradação de pneu é tendência
gradual. Slope positivo crescente antes do cliff = ritmo piorando = cliff real.

Filtro de magnitude (Fix 1):
  cliff_delta_magnitude < min_cliff_magnitude → cliff descartado como falso positivo.
  Magnitude negativa = ficando mais rápido = transição race start, não degradação.
"""

import numpy as np
import pandas as pd

try:
    import ruptures as rpt
except ImportError as e:
    raise ImportError(
        "Módulo 'ruptures' não encontrado. Instale com: uv add ruptures"
    ) from e

from src.utils.config import get_config


def detect_tire_changepoints(
    df: pd.DataFrame,
    signal_column: str = 'LapTime_delta',
    anomaly_col: str = 'is_anomaly',
    driver_column: str = 'Driver',
    stint_column: str = 'Stint',
    lap_column: str = 'LapNumber',
    penalty: float | None = None,
    min_size: int | None = None,
    jump: int | None = None,
    model: str | None = None,
    min_cliff_magnitude: float | None = None,
    validation_window: int | None = None,
    slope_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detecta tire cliffs (mudanças de regime de degradação) por driver × stint.

    Usa PELT (Pruned Exact Linear Time) do pacote ruptures para detectar
    breakpoints no sinal LapTime_delta, filtrando laps anômalos antes do fit.

    Filtros pós-PELT (semânticos):
    - Magnitude mínima: cliffs com magnitude < min_cliff_magnitude são descartados.
      Magnitude negativa = ficando mais rápido = falso positivo (race start, SC).
    - Validação por slope: se o LapTime_delta nos laps pré-cliff tem slope positivo
      (ritmo degradando), o cliff é considerado validado. Mais correto que usar
      anomaly_score (que detecta outliers pontuais, não tendências graduais).

    Args:
        df: DataFrame com dados de voltas (output de detect_anomalies_isolation_forest)
        signal_column: Coluna processada pelo PELT (padrão: 'LapTime_delta')
        anomaly_col: Coluna booleana — laps True são removidos do sinal antes do fit()
        driver_column: Coluna de identificação do piloto
        stint_column: Coluna de stint
        lap_column: Coluna de número de volta
        penalty: Penalidade PELT (None = usa config.yaml)
        min_size: Mínimo de laps entre breakpoints (None = usa config.yaml)
        jump: Step do grid de busca (None = usa config.yaml)
        model: Modelo de custo do Ruptures (None = usa config.yaml, padrão 'l2')
        min_cliff_magnitude: Magnitude mínima positiva em segundos para aceitar cliff.
                             Cliffs abaixo deste valor são falsos positivos (None = config)
        validation_window: Laps antes do cliff para calcular slope (None = config)
        slope_threshold: Slope mínimo (s/volta) para validar cliff (None = config)

    Returns:
        Tupla (laps_df, changepoints_df):
        - laps_df: df original + 'stint_regime' (int) e 'is_cliff_lap' (bool)
        - changepoints_df: um row por (Driver, Stint) com:
            n_laps_in_stint, n_anomalies_filtered, n_laps_analyzed,
            n_changepoints, has_cliff, cliff_lap, all_cliff_laps,
            laps_before_cliff, cliff_delta_magnitude,
            regime_pre_cliff_mean, regime_post_cliff_mean,
            pre_cliff_slope, cliff_validated

    Example:
        >>> laps_cp, cliffs = detect_tire_changepoints(laps_anomalies_df)
        >>> real_cliffs = cliffs[cliffs['has_cliff'] & cliffs['cliff_validated']]
        >>> print(real_cliffs[['Driver', 'Stint', 'cliff_lap', 'cliff_delta_magnitude']])
    """
    config = get_config()

    # Parâmetros do config
    if penalty is None:
        penalty = config.get_ruptures_penalty()
    if min_size is None:
        min_size = config.get_ruptures_min_size()
    if jump is None:
        jump = config.get_ruptures_jump()
    if model is None:
        model = config.get_ruptures_model()
    if min_cliff_magnitude is None:
        min_cliff_magnitude = config.get_ruptures_min_cliff_magnitude()
    if validation_window is None:
        validation_window = config.get_ruptures_validation_window()
    if slope_threshold is None:
        slope_threshold = config.get_ruptures_validation_slope_threshold()

    validation_enabled = config.get_ruptures_validation_enabled()

    # Verificar colunas obrigatórias
    required_cols = [driver_column, stint_column, lap_column, signal_column]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas não encontradas no DataFrame: {missing}")

    has_anomaly_col = anomaly_col in df.columns

    laps_df = df.copy()
    laps_df['stint_regime'] = 0
    laps_df['is_cliff_lap'] = False

    changepoint_rows = []

    for driver, driver_df in laps_df.groupby(driver_column):
        for stint, stint_df in driver_df.groupby(stint_column):
            stint_df = stint_df.sort_values(lap_column)
            n_laps_in_stint = len(stint_df)

            # Filtrar laps anômalos do sinal
            if has_anomaly_col:
                clean_mask = ~stint_df[anomaly_col].astype(bool)
                clean_df = stint_df[clean_mask]
                n_anomalies_filtered = int((~clean_mask).sum())
            else:
                clean_df = stint_df
                n_anomalies_filtered = 0

            n_laps_analyzed = len(clean_df)

            if n_laps_analyzed < min_size * 2:
                _append_no_cliff_row(
                    changepoint_rows, driver, stint,
                    n_laps_in_stint, n_anomalies_filtered, n_laps_analyzed,
                    driver_column, stint_column,
                )
                continue

            signal_values = clean_df[signal_column].values
            signal = signal_values.reshape(-1, 1)
            lap_numbers = clean_df[lap_column].values

            # Rodar PELT
            try:
                algo = rpt.Pelt(model=model, min_size=min_size, jump=jump)
                algo.fit(signal)
                breakpoints_raw = algo.predict(pen=penalty)
            except Exception:
                _append_no_cliff_row(
                    changepoint_rows, driver, stint,
                    n_laps_in_stint, n_anomalies_filtered, n_laps_analyzed,
                    driver_column, stint_column,
                )
                continue

            # Remover último elemento (len(signal) por convenção do ruptures)
            breakpoints = breakpoints_raw[:-1]

            if not breakpoints:
                _append_no_cliff_row(
                    changepoint_rows, driver, stint,
                    n_laps_in_stint, n_anomalies_filtered, n_laps_analyzed,
                    driver_column, stint_column,
                )
                continue

            # Converter índices → LapNumbers reais
            cliff_lap_numbers = [
                int(lap_numbers[bp]) for bp in breakpoints if bp < len(lap_numbers)
            ]

            if not cliff_lap_numbers:
                _append_no_cliff_row(
                    changepoint_rows, driver, stint,
                    n_laps_in_stint, n_anomalies_filtered, n_laps_analyzed,
                    driver_column, stint_column,
                )
                continue

            first_cliff_lap = cliff_lap_numbers[0]
            first_bp_idx = breakpoints[0]

            # Magnitude: média pós - média pré no sinal filtrado
            pre_signal = signal_values[:first_bp_idx]
            post_signal = signal_values[first_bp_idx:]
            regime_pre_mean = float(np.mean(pre_signal)) if len(pre_signal) > 0 else np.nan
            regime_post_mean = float(np.mean(post_signal)) if len(post_signal) > 0 else np.nan
            cliff_delta_magnitude = (
                regime_post_mean - regime_pre_mean
                if not (np.isnan(regime_pre_mean) or np.isnan(regime_post_mean))
                else np.nan
            )

            # FIX 1: Filtro de magnitude mínima
            # Magnitude negativa = ficando mais rápido = falso positivo (race start)
            # Magnitude abaixo do mínimo = mudança insignificante
            if np.isnan(cliff_delta_magnitude) or cliff_delta_magnitude < min_cliff_magnitude:
                _append_no_cliff_row(
                    changepoint_rows, driver, stint,
                    n_laps_in_stint, n_anomalies_filtered, n_laps_analyzed,
                    driver_column, stint_column,
                )
                continue

            # FIX 2: Validação por slope do sinal pré-cliff
            # Slope positivo = ritmo estava degradando antes do cliff = cliff real
            pre_cliff_slope = np.nan
            cliff_validated = False

            if validation_enabled and first_bp_idx >= 2:
                # Usar os últimos validation_window laps antes do cliff (no sinal filtrado)
                window_start = max(0, first_bp_idx - validation_window)
                pre_window_signal = signal_values[window_start:first_bp_idx]

                if len(pre_window_signal) >= 2:
                    x = np.arange(len(pre_window_signal), dtype=float)
                    slope, _ = np.polyfit(x, pre_window_signal, 1)
                    pre_cliff_slope = float(slope)
                    cliff_validated = pre_cliff_slope > slope_threshold

            laps_before_cliff = int(first_bp_idx)

            # Atribuir stint_regime e is_cliff_lap no laps_df original
            all_stint_laps_idx = stint_df.index
            cliff_lap_sorted = sorted(cliff_lap_numbers)

            for idx in all_stint_laps_idx:
                lap_num = laps_df.loc[idx, lap_column]
                regime = 0
                for i, cliff_ln in enumerate(cliff_lap_sorted):
                    if lap_num >= cliff_ln:
                        regime = i + 1
                laps_df.loc[idx, 'stint_regime'] = regime
                laps_df.loc[idx, 'is_cliff_lap'] = lap_num in cliff_lap_sorted

            changepoint_rows.append({
                driver_column: driver,
                stint_column: stint,
                'n_laps_in_stint': n_laps_in_stint,
                'n_anomalies_filtered': n_anomalies_filtered,
                'n_laps_analyzed': n_laps_analyzed,
                'n_changepoints': len(cliff_lap_numbers),
                'has_cliff': True,
                'cliff_lap': first_cliff_lap,
                'all_cliff_laps': cliff_lap_sorted,      # FIX 5: todos os breakpoints
                'laps_before_cliff': laps_before_cliff,
                'cliff_delta_magnitude': cliff_delta_magnitude,
                'regime_pre_cliff_mean': regime_pre_mean,
                'regime_post_cliff_mean': regime_post_mean,
                'pre_cliff_slope': pre_cliff_slope,      # FIX 2: slope pré-cliff
                'cliff_validated': cliff_validated,
            })

    changepoints_df = (
        pd.DataFrame(changepoint_rows)
        if changepoint_rows
        else _empty_changepoints_df(driver_column, stint_column)
    )

    return laps_df, changepoints_df


def _append_no_cliff_row(
    rows: list,
    driver: str,
    stint: int,
    n_laps_in_stint: int,
    n_anomalies_filtered: int,
    n_laps_analyzed: int,
    driver_column: str,
    stint_column: str,
) -> None:
    """Adiciona row 'sem cliff' à lista de resultados."""
    rows.append({
        driver_column: driver,
        stint_column: stint,
        'n_laps_in_stint': n_laps_in_stint,
        'n_anomalies_filtered': n_anomalies_filtered,
        'n_laps_analyzed': n_laps_analyzed,
        'n_changepoints': 0,
        'has_cliff': False,
        'cliff_lap': None,
        'all_cliff_laps': [],
        'laps_before_cliff': None,
        'cliff_delta_magnitude': np.nan,
        'regime_pre_cliff_mean': np.nan,
        'regime_post_cliff_mean': np.nan,
        'pre_cliff_slope': np.nan,
        'cliff_validated': False,
    })


def _empty_changepoints_df(driver_column: str, stint_column: str) -> pd.DataFrame:
    """Retorna DataFrame vazio com schema correto."""
    return pd.DataFrame(columns=[
        driver_column, stint_column,
        'n_laps_in_stint', 'n_anomalies_filtered', 'n_laps_analyzed',
        'n_changepoints', 'has_cliff', 'cliff_lap', 'all_cliff_laps',
        'laps_before_cliff', 'cliff_delta_magnitude',
        'regime_pre_cliff_mean', 'regime_post_cliff_mean',
        'pre_cliff_slope', 'cliff_validated',
    ])


def summarize_cliffs(
    changepoints_df: pd.DataFrame,
    driver_column: str = 'Driver',
) -> pd.DataFrame:
    """
    Gera sumário de cliffs por piloto.

    Args:
        changepoints_df: Output de detect_tire_changepoints()
        driver_column: Coluna de identificação do piloto

    Returns:
        DataFrame com uma linha por piloto:
        - total_stints: Total de stints analisados
        - stints_with_cliff: Stints com cliff detectado (após filtro de magnitude)
        - cliff_rate_pct: % de stints com cliff
        - cliffs_validated: Cliffs validados por slope pré-cliff
        - validated_rate_pct: % de cliffs validados
        - mean_cliff_magnitude: Magnitude média do cliff (segundos)
        - mean_laps_before_cliff: Média de voltas antes do primeiro cliff
        - mean_pre_cliff_slope: Slope médio pré-cliff dos stints com cliff

    Example:
        >>> summary = summarize_cliffs(changepoints_df)
        >>> print(summary.to_string(index=False))
    """
    if changepoints_df.empty or driver_column not in changepoints_df.columns:
        return pd.DataFrame()

    rows = []
    for driver, driver_df in changepoints_df.groupby(driver_column):
        total_stints = len(driver_df)
        cliffs_df = driver_df[driver_df['has_cliff'] == True]
        stints_with_cliff = len(cliffs_df)
        cliff_rate_pct = 100.0 * stints_with_cliff / total_stints if total_stints > 0 else 0.0

        cliffs_validated = (
            int(cliffs_df['cliff_validated'].sum())
            if 'cliff_validated' in cliffs_df.columns else 0
        )
        validated_rate_pct = (
            100.0 * cliffs_validated / stints_with_cliff if stints_with_cliff > 0 else 0.0
        )

        mean_magnitude = (
            float(cliffs_df['cliff_delta_magnitude'].mean())
            if stints_with_cliff > 0 else np.nan
        )
        mean_laps_before = (
            float(cliffs_df['laps_before_cliff'].mean())
            if stints_with_cliff > 0 and 'laps_before_cliff' in cliffs_df.columns else np.nan
        )
        mean_slope = (
            float(cliffs_df['pre_cliff_slope'].mean())
            if stints_with_cliff > 0 and 'pre_cliff_slope' in cliffs_df.columns else np.nan
        )

        rows.append({
            driver_column: driver,
            'total_stints': total_stints,
            'stints_with_cliff': stints_with_cliff,
            'cliff_rate_pct': round(cliff_rate_pct, 1),
            'cliffs_validated': cliffs_validated,
            'validated_rate_pct': round(validated_rate_pct, 1),
            'mean_cliff_magnitude': round(mean_magnitude, 4) if not np.isnan(mean_magnitude) else np.nan,
            'mean_laps_before_cliff': round(mean_laps_before, 1) if not np.isnan(mean_laps_before) else np.nan,
            'mean_pre_cliff_slope': round(mean_slope, 4) if not np.isnan(mean_slope) else np.nan,
        })

    return pd.DataFrame(rows).sort_values(driver_column).reset_index(drop=True)
