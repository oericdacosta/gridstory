"""
Módulo de pré-processamento para o pipeline.

Gerencia a Fase 2: Pré-processamento de todos os 5 tipos de dados
(laps, telemetry, race_control, weather, results).
"""

from pathlib import Path

import pandas as pd

from src.preprocessing.interpolation import synchronize_telemetry
from src.preprocessing.signal_processing import apply_telemetry_pipeline
from src.preprocessing.feature_engineering import (
    enrich_dataframe_with_stats,
    preprocess_race_control,
    preprocess_weather,
    preprocess_results,
)
from .reporting import Reporter


def run_preprocessing_phase(
    race_dir: Path,
    year: int,
    round_num: int,
    show_sample: bool = False,
) -> Path:
    """
    Executa a fase de pré-processamento de todos os dados.

    Args:
        race_dir: Diretório com dados brutos
        year: Ano da temporada
        round_num: Número da rodada
        show_sample: Se deve mostrar amostras dos dados

    Returns:
        Path para o diretório com dados processados
    """
    reporter = Reporter("FASE 2: PRÉ-PROCESSAMENTO")
    reporter.header()

    # Setup diretórios
    processed_dir = Path("data/processed/races") / f"{year}" / f"round_{round_num:02d}"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Processar cada tipo de dado
    _preprocess_laps(race_dir, processed_dir, reporter, show_sample)
    _preprocess_telemetry(race_dir, processed_dir, reporter, show_sample)
    _preprocess_race_control(race_dir, processed_dir, reporter, show_sample)
    _preprocess_weather(race_dir, processed_dir, reporter, show_sample)
    _preprocess_results(race_dir, processed_dir, reporter, show_sample)

    reporter.success(f"Pré-processamento concluído: {processed_dir}", indent=0)

    return processed_dir


def _preprocess_laps(
    race_dir: Path,
    processed_dir: Path,
    reporter: Reporter,
    show_sample: bool,
):
    """Pré-processa dados de voltas (features estatísticas)."""
    reporter.section("2.1", "Pré-processando LAPS (features estatísticas)")

    laps_file = race_dir / "laps.parquet"
    if not laps_file.exists():
        reporter.info(f"⚠️  Arquivo não encontrado: {laps_file}")
        return

    laps_df = pd.read_parquet(laps_file)
    reporter.info(f"{len(laps_df)} voltas carregadas")

    # Aplicar feature engineering
    laps_processed = enrich_dataframe_with_stats(
        laps_df,
        value_column='LapTime_seconds',
        group_by=['Driver', 'Compound'] if 'Compound' in laps_df.columns else ['Driver'],
        include_degradation=True
    )

    # Salvar
    output_file = processed_dir / "laps_processed.parquet"
    laps_processed.to_parquet(output_file, index=False)

    reporter.success(f"Laps processados: {output_file}")
    reporter.metric("Outliers detectados", laps_processed['is_outlier'].sum())
    reporter.metric("Features adicionadas", len(laps_processed.columns) - len(laps_df.columns))

    if show_sample:
        reporter.sample(
            laps_processed,
            columns=['Driver', 'LapNumber', 'LapTime_seconds', 'z_score', 'is_outlier', 'degradation_slope']
        )


def _preprocess_telemetry(
    race_dir: Path,
    processed_dir: Path,
    reporter: Reporter,
    show_sample: bool,
):
    """Pré-processa dados de telemetria (sincronização + limpeza)."""
    reporter.section("2.2", "Pré-processando TELEMETRIA (sincronização + limpeza)")

    telemetry_dir = race_dir / "telemetry"
    if not telemetry_dir.exists() or not telemetry_dir.is_dir():
        reporter.info(f"⚠️  Diretório de telemetria não encontrado: {telemetry_dir}")
        return

    telemetry_files = list(telemetry_dir.glob("*.parquet"))
    reporter.info(f"{len(telemetry_files)} pilotos encontrados")

    processed_telemetry_dir = processed_dir / "telemetry"
    processed_telemetry_dir.mkdir(exist_ok=True)

    # Auto-detectar comprimento da pista
    first_file = telemetry_files[0]
    sample_tel = pd.read_parquet(first_file)
    track_length = sample_tel['Distance'].max() if 'Distance' in sample_tel.columns else 5000.0
    reporter.info(f"🏁 Comprimento da pista: {track_length:.0f}m")

    for tel_file in telemetry_files:
        driver = tel_file.stem
        telemetry_df = pd.read_parquet(tel_file)

        if 'Distance' not in telemetry_df.columns or len(telemetry_df) == 0:
            continue

        # Sincronizar (num_points carregado de config.yaml)
        synchronized = synchronize_telemetry(
            telemetry_df,
            track_length=track_length,
        )

        # Extrair canais para processamento
        telemetry_dict = {}
        for col in ['Speed', 'RPM', 'Throttle', 'Brake', 'nGear', 'DRS']:
            if col in synchronized.columns:
                telemetry_dict[col] = synchronized[col].values

        # Aplicar pipeline de processamento
        processed = apply_telemetry_pipeline(
            telemetry_dict,
            noise_reduction=True,
            outlier_removal=True,
            calculate_derivatives=True
        )

        # Salvar
        processed_df = pd.DataFrame(processed)
        output_file = processed_telemetry_dir / f"{driver}_processed.parquet"
        processed_df.to_parquet(output_file, index=False)

        reporter.success(f"{driver}: {len(processed_df)} pontos, {len(processed_df.columns)} canais")

    if show_sample and telemetry_files:
        sample_driver = telemetry_files[0].stem
        sample_file = processed_telemetry_dir / f"{sample_driver}_processed.parquet"
        if sample_file.exists():
            sample_df = pd.read_parquet(sample_file)
            print(f"\n   Amostra telemetria {sample_driver} (primeiras 5 linhas):")
            print(sample_df.head(5).to_string(index=False))


def _preprocess_race_control(
    race_dir: Path,
    processed_dir: Path,
    reporter: Reporter,
    show_sample: bool,
):
    """Pré-processa dados de race control (eventos e flags)."""
    reporter.section("2.3", "Pré-processando RACE CONTROL (eventos e flags)")

    race_control_file = race_dir / "race_control.parquet"
    if not race_control_file.exists():
        reporter.info(f"⚠️  Arquivo não encontrado: {race_control_file}")
        return

    race_control_df = pd.read_parquet(race_control_file)
    reporter.info(f"{len(race_control_df)} mensagens carregadas")

    # Processar
    race_control_processed = preprocess_race_control(race_control_df)

    # Salvar
    output_file = processed_dir / "race_control_processed.parquet"
    race_control_processed.to_parquet(output_file, index=False)

    reporter.success(f"Race Control processado: {output_file}")
    reporter.metric("Safety Car eventos", race_control_processed['is_safety_car'].sum())
    reporter.metric("Bandeiras", race_control_processed['is_flag'].sum())
    reporter.metric("Penalidades", race_control_processed['is_penalty'].sum())

    if show_sample:
        reporter.sample(
            race_control_processed,
            columns=['time_seconds', 'category', 'is_safety_car', 'is_flag', 'event_severity']
        )


def _preprocess_weather(
    race_dir: Path,
    processed_dir: Path,
    reporter: Reporter,
    show_sample: bool,
):
    """Pré-processa dados de clima (tendências)."""
    reporter.section("2.4", "Pré-processando WEATHER (clima e tendências)")

    weather_file = race_dir / "weather.parquet"
    if not weather_file.exists():
        reporter.info(f"⚠️  Arquivo não encontrado: {weather_file}")
        return

    weather_df = pd.read_parquet(weather_file)
    reporter.info(f"{len(weather_df)} registros meteorológicos")

    # Processar
    weather_processed = preprocess_weather(weather_df)

    # Salvar
    output_file = processed_dir / "weather_processed.parquet"
    weather_processed.to_parquet(output_file, index=False)

    reporter.success(f"Weather processado: {output_file}")
    if 'rainfall_indicator' in weather_processed.columns:
        reporter.metric("Períodos de chuva", weather_processed['rainfall_indicator'].sum())
    if 'weather_change' in weather_processed.columns:
        reporter.metric("Mudanças bruscas", weather_processed['weather_change'].sum())

    if show_sample:
        cols_to_show = ['time_seconds', 'AirTemp', 'TrackTemp', 'temp_delta', 'rainfall_indicator']
        available_cols = [c for c in cols_to_show if c in weather_processed.columns]
        reporter.sample(weather_processed, columns=available_cols)


def _preprocess_results(
    race_dir: Path,
    processed_dir: Path,
    reporter: Reporter,
    show_sample: bool,
):
    """Pré-processa resultados (classificação e desempenho)."""
    reporter.section("2.5", "Pré-processando RESULTS (classificação e desempenho)")

    results_file = race_dir / "results.parquet"
    if not results_file.exists():
        reporter.info(f"⚠️  Arquivo não encontrado: {results_file}")
        return

    results_df = pd.read_parquet(results_file)
    reporter.info(f"{len(results_df)} pilotos classificados")

    # Processar
    results_processed = preprocess_results(results_df)

    # Salvar
    output_file = processed_dir / "results_processed.parquet"
    results_processed.to_parquet(output_file, index=False)

    reporter.success(f"Results processados: {output_file}")
    if 'finish_status' in results_processed.columns:
        finished = results_processed['finish_status'].sum()
        reporter.metric("Pilotos que terminaram", f"{finished}/{len(results_processed)}")
    if 'position_gain' in results_processed.columns:
        gainers = results_processed['position_gain'].sum()
        reporter.metric("Pilotos que ganharam posições", gainers)

    if show_sample:
        cols_to_show = ['Abbreviation', 'final_position', 'grid_position', 'position_change', 'performance_score']
        available_cols = [c for c in cols_to_show if c in results_processed.columns]
        if 'Abbreviation' not in results_processed.columns and 'Driver' in results_processed.columns:
            available_cols = ['Driver'] + available_cols[1:]
        reporter.sample(results_processed, columns=available_cols)
