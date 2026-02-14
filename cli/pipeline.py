#!/usr/bin/env python3
"""
PitWall AI - Pipeline Completo End-to-End.

Pipeline unificado que executa:
1. Extração completa de dados da corrida (laps, telemetry, race_control, weather, results)
2. Pré-processamento de TODOS os dados com NumPy, Pandas e SciPy
3. Salvamento dos dados processados

Exemplo de uso:
    # Pipeline completo para uma corrida
    uv run python cli/pipeline.py 2025 1

    # Com polling (aguardar disponibilidade)
    uv run python cli/pipeline.py 2025 1 --polling

    # Mostrar amostras dos dados processados
    uv run python cli/pipeline.py 2025 1 --show-sample
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.orchestrator import extract_race_complete
from src.preprocessing.interpolation import synchronize_telemetry
from src.preprocessing.signal_processing import apply_telemetry_pipeline
from src.preprocessing.feature_engineering import (
    enrich_dataframe_with_stats,
    preprocess_race_control,
    preprocess_weather,
    preprocess_results,
)


def run_complete_pipeline(
    year: int,
    round_num: int,
    use_polling: bool = False,
    show_sample: bool = False,
):
    """
    Executa pipeline completo: extração + pré-processamento.

    Args:
        year: Ano da temporada
        round_num: Número da rodada
        use_polling: Se deve aguardar disponibilidade dos dados
        show_sample: Se deve mostrar amostras dos dados processados
    """
    print("\n" + "=" * 80)
    print("🏎️  PITWALL AI - PIPELINE COMPLETO")
    print("=" * 80)
    print(f"📅 Temporada: {year}, Rodada: {round_num}")
    print("=" * 80)

    # ========================================================================
    # FASE 1: EXTRAÇÃO DE DADOS
    # ========================================================================
    print("\n" + "=" * 80)
    print("📥 FASE 1: EXTRAÇÃO DE DADOS")
    print("=" * 80)

    # Habilitar cache do FastF1
    import fastf1

    cache_dir = Path.home() / ".cache" / "fastf1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    # Extrair dados
    race_dir = extract_race_complete(
        year=year,
        round_number=round_num,
        use_polling=use_polling,
        output_dir="data/raw/races",
    )

    print(f"\n✅ Extração concluída: {race_dir}")

    # ========================================================================
    # FASE 2: PRÉ-PROCESSAMENTO
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔬 FASE 2: PRÉ-PROCESSAMENTO")
    print("=" * 80)

    processed_dir = Path("data/processed/races") / f"{year}" / f"round_{round_num:02d}"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # 2.1: Pré-processar LAPS (Features Estatísticas)
    # ------------------------------------------------------------------------
    print("\n🔄 2.1 Pré-processando LAPS (features estatísticas)...")

    laps_file = Path(race_dir) / "laps.parquet"
    if laps_file.exists():
        laps_df = pd.read_parquet(laps_file)
        print(f"   📊 {len(laps_df)} voltas carregadas")

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

        print(f"   ✅ Laps processados: {output_file}")
        print(f"      • Outliers detectados: {laps_processed['is_outlier'].sum()}")
        print(f"      • Features adicionadas: {len(laps_processed.columns) - len(laps_df.columns)}")

        if show_sample:
            print(f"\n   Amostra (primeiras 5 linhas):")
            print(laps_processed.head(5)[['Driver', 'LapNumber', 'LapTime_seconds', 'z_score', 'is_outlier', 'degradation_slope']].to_string(index=False))
    else:
        print(f"   ⚠️  Arquivo não encontrado: {laps_file}")

    # ------------------------------------------------------------------------
    # 2.2: Pré-processar TELEMETRIA (Sincronização + Limpeza)
    # ------------------------------------------------------------------------
    print("\n🔄 2.2 Pré-processando TELEMETRIA (sincronização + limpeza)...")

    telemetry_dir = Path(race_dir) / "telemetry"
    if telemetry_dir.exists() and telemetry_dir.is_dir():
        telemetry_files = list(telemetry_dir.glob("*.parquet"))
        print(f"   📊 {len(telemetry_files)} pilotos encontrados")

        processed_telemetry_dir = processed_dir / "telemetry"
        processed_telemetry_dir.mkdir(exist_ok=True)

        # Auto-detectar comprimento da pista
        first_file = telemetry_files[0]
        sample_tel = pd.read_parquet(first_file)
        track_length = sample_tel['Distance'].max() if 'Distance' in sample_tel.columns else 5000.0
        print(f"   🏁 Comprimento da pista: {track_length:.0f}m")

        for tel_file in telemetry_files:
            driver = tel_file.stem
            telemetry_df = pd.read_parquet(tel_file)

            if 'Distance' in telemetry_df.columns and len(telemetry_df) > 0:
                # Sincronizar
                synchronized = synchronize_telemetry(
                    telemetry_df,
                    track_length=track_length,
                    num_points=1000,
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

                print(f"   ✅ {driver}: {len(processed_df)} pontos, {len(processed_df.columns)} canais")

        if show_sample:
            sample_driver = telemetry_files[0].stem
            sample_file = processed_telemetry_dir / f"{sample_driver}_processed.parquet"
            sample_df = pd.read_parquet(sample_file)
            print(f"\n   Amostra telemetria {sample_driver} (primeiras 5 linhas):")
            print(sample_df.head(5).to_string(index=False))
    else:
        print(f"   ⚠️  Diretório de telemetria não encontrado: {telemetry_dir}")

    # ------------------------------------------------------------------------
    # 2.3: Pré-processar RACE CONTROL (Eventos e Flags)
    # ------------------------------------------------------------------------
    print("\n🔄 2.3 Pré-processando RACE CONTROL (eventos e flags)...")

    race_control_file = Path(race_dir) / "race_control.parquet"
    if race_control_file.exists():
        race_control_df = pd.read_parquet(race_control_file)
        print(f"   📊 {len(race_control_df)} mensagens carregadas")

        # Processar
        race_control_processed = preprocess_race_control(race_control_df)

        # Salvar
        output_file = processed_dir / "race_control_processed.parquet"
        race_control_processed.to_parquet(output_file, index=False)

        print(f"   ✅ Race Control processado: {output_file}")
        print(f"      • Safety Car eventos: {race_control_processed['is_safety_car'].sum()}")
        print(f"      • Bandeiras: {race_control_processed['is_flag'].sum()}")
        print(f"      • Penalidades: {race_control_processed['is_penalty'].sum()}")

        if show_sample:
            print(f"\n   Amostra (primeiras 5 linhas):")
            print(race_control_processed.head(5)[['time_seconds', 'category', 'is_safety_car', 'is_flag', 'event_severity']].to_string(index=False))
    else:
        print(f"   ⚠️  Arquivo não encontrado: {race_control_file}")

    # ------------------------------------------------------------------------
    # 2.4: Pré-processar WEATHER (Clima e Tendências)
    # ------------------------------------------------------------------------
    print("\n🔄 2.4 Pré-processando WEATHER (clima e tendências)...")

    weather_file = Path(race_dir) / "weather.parquet"
    if weather_file.exists():
        weather_df = pd.read_parquet(weather_file)
        print(f"   📊 {len(weather_df)} registros meteorológicos")

        # Processar
        weather_processed = preprocess_weather(weather_df)

        # Salvar
        output_file = processed_dir / "weather_processed.parquet"
        weather_processed.to_parquet(output_file, index=False)

        print(f"   ✅ Weather processado: {output_file}")
        if 'rainfall_indicator' in weather_processed.columns:
            print(f"      • Períodos de chuva: {weather_processed['rainfall_indicator'].sum()}")
        if 'weather_change' in weather_processed.columns:
            print(f"      • Mudanças bruscas: {weather_processed['weather_change'].sum()}")

        if show_sample:
            print(f"\n   Amostra (primeiras 5 linhas):")
            cols_to_show = ['time_seconds', 'AirTemp', 'TrackTemp', 'temp_delta', 'rainfall_indicator']
            available_cols = [c for c in cols_to_show if c in weather_processed.columns]
            print(weather_processed.head(5)[available_cols].to_string(index=False))
    else:
        print(f"   ⚠️  Arquivo não encontrado: {weather_file}")

    # ------------------------------------------------------------------------
    # 2.5: Pré-processar RESULTS (Classificação e Desempenho)
    # ------------------------------------------------------------------------
    print("\n🔄 2.5 Pré-processando RESULTS (classificação e desempenho)...")

    results_file = Path(race_dir) / "results.parquet"
    if results_file.exists():
        results_df = pd.read_parquet(results_file)
        print(f"   📊 {len(results_df)} pilotos classificados")

        # Processar
        results_processed = preprocess_results(results_df)

        # Salvar
        output_file = processed_dir / "results_processed.parquet"
        results_processed.to_parquet(output_file, index=False)

        print(f"   ✅ Results processados: {output_file}")
        if 'finish_status' in results_processed.columns:
            finished = results_processed['finish_status'].sum()
            print(f"      • Pilotos que terminaram: {finished}/{len(results_processed)}")
        if 'position_gain' in results_processed.columns:
            gainers = results_processed['position_gain'].sum()
            print(f"      • Pilotos que ganharam posições: {gainers}")

        if show_sample:
            print(f"\n   Amostra (primeiras 5 linhas):")
            cols_to_show = ['Abbreviation', 'final_position', 'grid_position', 'position_change', 'performance_score']
            available_cols = [c for c in cols_to_show if c in results_processed.columns]
            if 'Abbreviation' not in results_processed.columns and 'Driver' in results_processed.columns:
                available_cols[0] = 'Driver'
            print(results_processed.head(5)[available_cols].to_string(index=False))
    else:
        print(f"   ⚠️  Arquivo não encontrado: {results_file}")

    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETO CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    print(f"\n📁 Dados brutos salvos em:")
    print(f"   {race_dir}")
    print(f"\n📁 Dados processados salvos em:")
    print(f"   {processed_dir}")

    print("\n📊 Arquivos processados:")
    for file in sorted(processed_dir.glob("**/*.parquet")):
        size_kb = file.stat().st_size / 1024
        print(f"   • {file.relative_to(processed_dir)}: {size_kb:.1f} KB")

    print("\n" + "=" * 80)
    print("\n🎯 Próximos passos:")
    print("   1. Os dados estão prontos para análise ML")
    print("   2. Use os dados processados para treinar modelos")
    print("   3. Implemente pipeline ML (Ruptures, Isolation Forest, etc.)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="PitWall AI - Pipeline Completo (Extração + Pré-processamento)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "year",
        type=int,
        help="Ano da temporada (ex: 2025)",
    )

    parser.add_argument(
        "round",
        type=int,
        help="Número da rodada/corrida (ex: 1)",
    )

    parser.add_argument(
        "--polling",
        action="store_true",
        help="Usar polling para aguardar disponibilidade dos dados",
    )

    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Mostrar amostras dos dados processados",
    )

    args = parser.parse_args()

    try:
        run_complete_pipeline(
            year=args.year,
            round_num=args.round,
            use_polling=args.polling,
            show_sample=args.show_sample,
        )
    except Exception as e:
        print(f"\n❌ Erro durante pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
