#!/usr/bin/env python3
"""
CLI para pré-processamento de dados de telemetria F1.

Aplica pipeline SciPy nos dados brutos extraídos do FastF1.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.interpolation import synchronize_telemetry
from src.preprocessing.signal_processing import apply_telemetry_pipeline
from src.preprocessing.feature_engineering import enrich_dataframe_with_stats
from src.utils.config import get_config


def load_lap_data(year: int, round_num: int, session_type: str = "R") -> pd.DataFrame:
    """Carrega dados de voltas de um arquivo Parquet."""
    data_dir = Path(__file__).parent.parent / "data" / "raw"

    # Tentar estrutura hierárquica primeiro (data/raw/races/YEAR/round_XX/laps.parquet)
    hierarchical_path = data_dir / "races" / str(year) / f"round_{round_num:02d}" / "laps.parquet"

    # Fallback para estrutura flat (data/raw/laps_YEAR_XX_R.parquet)
    flat_path = data_dir / f"laps_{year}_{round_num:02d}_{session_type}.parquet"

    if hierarchical_path.exists():
        file_path = hierarchical_path
    elif flat_path.exists():
        file_path = flat_path
    else:
        raise FileNotFoundError(
            f"Arquivo não encontrado.\n"
            f"Tentou: {hierarchical_path}\n"
            f"Tentou: {flat_path}"
        )

    print(f"📂 Carregando: {file_path}")
    return pd.read_parquet(file_path)


def load_telemetry_data(year: int, round_num: int, session_type: str = "R") -> pd.DataFrame:
    """Carrega dados de telemetria de um arquivo Parquet."""
    data_dir = Path(__file__).parent.parent / "data" / "raw"

    # Tentar estrutura hierárquica (data/raw/races/YEAR/round_XX/telemetry/*.parquet)
    hierarchical_dir = data_dir / "races" / str(year) / f"round_{round_num:02d}" / "telemetry"

    # Fallback para estrutura flat
    flat_path = data_dir / f"telemetry_{year}_{round_num:02d}_{session_type}.parquet"

    if hierarchical_dir.exists() and hierarchical_dir.is_dir():
        # Carregar todos os arquivos de telemetria e concatenar
        telemetry_files = list(hierarchical_dir.glob("*.parquet"))

        if not telemetry_files:
            raise FileNotFoundError(f"Nenhum arquivo de telemetria encontrado em: {hierarchical_dir}")

        print(f"📂 Carregando telemetria de: {hierarchical_dir}")
        print(f"   Arquivos encontrados: {len(telemetry_files)} pilotos")

        dfs = []
        for file in telemetry_files:
            df = pd.read_parquet(file)
            # Adicionar coluna Driver baseada no nome do arquivo se não existir
            if 'Driver' not in df.columns:
                df['Driver'] = file.stem  # Nome do arquivo sem extensão
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    elif flat_path.exists():
        print(f"📂 Carregando: {flat_path}")
        return pd.read_parquet(flat_path)

    else:
        raise FileNotFoundError(
            f"Arquivo não encontrado.\n"
            f"Tentou: {hierarchical_dir}\n"
            f"Tentou: {flat_path}"
        )


def show_dataframe_sample(df: pd.DataFrame, title: str, max_rows: int = 10, max_cols: int = 15):
    """
    Mostra amostra de um DataFrame em formato de tabela.

    Args:
        df: DataFrame para mostrar
        title: Título da tabela
        max_rows: Número máximo de linhas para mostrar
        max_cols: Número máximo de colunas para mostrar
    """
    print("\n" + "=" * 100)
    print(f"📊 {title}")
    print("=" * 100)

    # Configurar pandas para melhor visualização
    pd.set_option('display.max_columns', max_cols)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 20)

    # Mostrar informações gerais
    print(f"\n📈 Dimensões: {len(df)} linhas × {len(df.columns)} colunas")
    print(f"📋 Colunas: {list(df.columns)}\n")

    # Mostrar amostra
    if len(df) > max_rows:
        print(f"Mostrando primeiras {max_rows} linhas de {len(df)}:\n")
        sample = df.head(max_rows)
    else:
        sample = df

    # Mostrar tabela
    print(sample.to_string(index=False))

    # Resetar opções do pandas
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')


def preprocess_lap_features(laps_df: pd.DataFrame, driver: str = None, show_sample: bool = False) -> pd.DataFrame:
    """
    Aplica engenharia de features estatísticas nos dados de voltas.

    Args:
        laps_df: DataFrame com dados de voltas
        driver: Filtrar por piloto específico (opcional)

    Returns:
        DataFrame enriquecido com features estatísticas
    """
    print("\n🔬 PASSO 1: Engenharia de Features Estatísticas")
    print("=" * 80)

    # Filtrar driver se especificado
    if driver:
        laps_df = laps_df[laps_df['Driver'] == driver].copy()
        print(f"📊 Filtrando piloto: {driver}")

    print(f"📊 Total de voltas: {len(laps_df)}")

    # Mostrar amostra dos dados BRUTOS se solicitado
    if show_sample:
        show_dataframe_sample(
            laps_df,
            "DADOS BRUTOS (antes do pré-processamento)",
            max_rows=10
        )

    # Determinar qual coluna de tempo usar
    if 'LapTime_seconds' in laps_df.columns:
        time_column = 'LapTime_seconds'
    elif 'LapTime' in laps_df.columns:
        time_column = 'LapTime'
    else:
        raise ValueError("Nenhuma coluna de tempo de volta encontrada (LapTime ou LapTime_seconds)")

    print(f"📊 Usando coluna: {time_column}")

    # Mostrar dados brutos
    print("\n📈 Estatísticas ANTES do pré-processamento:")
    if time_column in laps_df.columns:
        lap_times = laps_df[time_column]

        # Se não estiver em segundos, converter
        if time_column == 'LapTime':
            lap_times = pd.to_timedelta(lap_times).dt.total_seconds()

        # Filtrar NaN para estatísticas
        valid_times = lap_times.dropna()
        if len(valid_times) > 0:
            print(f"  Tempo médio de volta: {valid_times.mean():.3f}s")
            print(f"  Desvio padrão: {valid_times.std():.3f}s")
            print(f"  Tempo mínimo: {valid_times.min():.3f}s")
            print(f"  Tempo máximo: {valid_times.max():.3f}s")

    # Aplicar pré-processamento
    enriched = enrich_dataframe_with_stats(
        laps_df,
        value_column=time_column,
        group_by=['Driver', 'Compound'] if 'Compound' in laps_df.columns else ['Driver'],
        include_degradation=True
    )

    # Mostrar resultados
    print("\n✨ Estatísticas DEPOIS do pré-processamento:")
    print(f"  Colunas adicionadas: {[col for col in enriched.columns if col not in laps_df.columns]}")

    if 'is_outlier' in enriched.columns:
        outliers = enriched['is_outlier'].sum()
        print(f"  Outliers detectados: {outliers} / {len(enriched)} ({outliers/len(enriched)*100:.1f}%)")

    if 'z_score' in enriched.columns:
        print(f"  Z-score range: [{enriched['z_score'].min():.2f}, {enriched['z_score'].max():.2f}]")

    if 'degradation_slope' in enriched.columns:
        # Mostrar resumo de degradação por piloto
        print("\n🔧 Taxa de Degradação de Pneu (Top pilotos):")
        print("   [Valores negativos = tempos melhorando, positivos = pneu degradando]\n")

        # Agrupar por piloto para mostrar resumo
        drivers = enriched['Driver'].unique()[:10]  # Mostrar apenas top 10 pilotos

        for driver in sorted(drivers):
            driver_data = enriched[enriched['Driver'] == driver]

            print(f"  🏎️  {driver}:")

            # Agrupar por stint
            if 'Stint' in driver_data.columns:
                for stint in sorted(driver_data['Stint'].unique()):
                    stint_data = driver_data[driver_data['Stint'] == stint].iloc[0]

                    if not np.isnan(stint_data['degradation_slope']):
                        compound = stint_data.get('Compound', 'N/A')
                        slope = stint_data['degradation_slope']
                        r2 = stint_data['degradation_r_squared']

                        # Formatar stint number
                        stint_num = int(stint) if isinstance(stint, (np.integer, np.floating)) else stint

                        # Indicador visual
                        indicator = "📈" if slope > 0 else "📉" if slope < -0.5 else "➡️"

                        print(f"     Stint {stint_num} ({compound:12s}): {slope:+.3f}s/volta {indicator} (R²={r2:.2f})")
            else:
                # Sem stints, mostrar por composto
                for compound in driver_data['Compound'].unique():
                    comp_data = driver_data[driver_data['Compound'] == compound].iloc[0]

                    if not np.isnan(comp_data['degradation_slope']):
                        slope = comp_data['degradation_slope']
                        r2 = comp_data['degradation_r_squared']
                        indicator = "📈" if slope > 0 else "📉" if slope < -0.5 else "➡️"

                        print(f"     {compound:12s}: {slope:+.3f}s/volta {indicator} (R²={r2:.2f})")

            print()  # Linha em branco entre pilotos

    # Mostrar amostra dos dados PRÉ-PROCESSADOS se solicitado
    if show_sample:
        show_dataframe_sample(
            enriched,
            "DADOS PRÉ-PROCESSADOS (depois do pré-processamento)",
            max_rows=10
        )

        # Mostrar comparação de colunas
        print("\n" + "=" * 100)
        print("📊 COMPARAÇÃO DE COLUNAS")
        print("=" * 100)

        original_cols = set(laps_df.columns)
        new_cols = set(enriched.columns) - original_cols

        print(f"\n✅ Colunas originais ({len(original_cols)}):")
        print(f"   {', '.join(sorted(original_cols))}")

        print(f"\n✨ Colunas adicionadas ({len(new_cols)}):")
        for col in sorted(new_cols):
            # Mostrar exemplo de valor
            sample_val = enriched[col].iloc[0]
            if isinstance(sample_val, float):
                print(f"   • {col:30s} (exemplo: {sample_val:.3f})")
            else:
                print(f"   • {col:30s} (exemplo: {sample_val})")

    return enriched


def preprocess_telemetry_signals(
    telemetry_df: pd.DataFrame,
    track_length: float,
    driver: str = None,
    lap_number: int = None,
    show_sample: bool = False
) -> dict:
    """
    Aplica processamento de sinal na telemetria.

    Args:
        telemetry_df: DataFrame com telemetria bruta
        track_length: Comprimento da pista em metros
        driver: Filtrar por piloto (opcional)
        lap_number: Filtrar por número de volta (opcional)

    Returns:
        Dicionário com telemetria processada
    """
    print("\n🔬 PASSO 2: Processamento de Sinal de Telemetria")
    print("=" * 80)

    # Filtrar
    if driver:
        telemetry_df = telemetry_df[telemetry_df['Driver'] == driver].copy()
        print(f"📊 Filtrando piloto: {driver}")

    if lap_number:
        telemetry_df = telemetry_df[telemetry_df['LapNumber'] == lap_number].copy()
        print(f"📊 Filtrando volta: {lap_number}")

    print(f"📊 Total de pontos de telemetria: {len(telemetry_df)}")

    # Mostrar amostra dos dados BRUTOS se solicitado
    if show_sample:
        show_dataframe_sample(
            telemetry_df,
            "TELEMETRIA BRUTA (antes do pré-processamento)",
            max_rows=10
        )

    # Verificar colunas disponíveis
    telemetry_cols = ['Speed', 'RPM', 'Throttle', 'Brake', 'nGear', 'DRS']
    available_cols = [col for col in telemetry_cols if col in telemetry_df.columns]

    print(f"📊 Canais de telemetria disponíveis: {available_cols}")

    if 'Distance' not in telemetry_df.columns:
        print("⚠️  Coluna 'Distance' não encontrada - pulando sincronização")
        return {}

    # Sincronizar telemetria (num_points carregado de config.yaml)
    config = get_config()
    num_points = config.get_num_points()
    print(f"\n🔄 Sincronizando telemetria (grid: {num_points} pontos)")
    synchronized = synchronize_telemetry(
        telemetry_df,
        track_length=track_length,
        telemetry_columns=available_cols
    )

    print(f"✅ Telemetria sincronizada: {len(synchronized)} pontos")

    # Extrair canais para processamento
    telemetry_dict = {}
    for col in available_cols:
        if col in synchronized.columns:
            telemetry_dict[col] = synchronized[col].values

    # Mostrar estatísticas ANTES
    print("\n📈 Estatísticas ANTES do processamento de sinal:")
    for channel, values in telemetry_dict.items():
        print(f"  {channel}: média={np.nanmean(values):.2f}, std={np.nanstd(values):.2f}")

    # Aplicar pipeline de processamento
    processed = apply_telemetry_pipeline(
        telemetry_dict,
        noise_reduction=True,
        outlier_removal=True,
        calculate_derivatives=True
    )

    # Mostrar estatísticas DEPOIS
    print("\n✨ Estatísticas DEPOIS do processamento de sinal:")
    for channel in available_cols:
        if channel in processed:
            values = processed[channel]
            print(f"  {channel}: média={np.nanmean(values):.2f}, std={np.nanstd(values):.2f}")

    # Mostrar derivadas calculadas
    derivatives = [key for key in processed.keys() if '_derivative' in key]
    if derivatives:
        print(f"\n📐 Derivadas calculadas: {derivatives}")
        for deriv in derivatives:
            values = processed[deriv]
            print(f"  {deriv}: min={np.nanmin(values):.2f}, max={np.nanmax(values):.2f}")

    # Mostrar amostra da telemetria PRÉ-PROCESSADA se solicitado
    if show_sample and processed:
        # Converter dict para DataFrame para visualização
        processed_df = pd.DataFrame({
            k: v[:100]  # Primeiras 100 amostras
            for k, v in processed.items()
        })

        show_dataframe_sample(
            processed_df,
            "TELEMETRIA PRÉ-PROCESSADA (sincronizada e limpa)",
            max_rows=10
        )

    return processed


def save_processed_data(data: pd.DataFrame, year: int, round_num: int, session_type: str, data_type: str):
    """Salva dados processados em Parquet."""
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{data_type}_processed_{year}_{round_num:02d}_{session_type}.parquet"

    data.to_parquet(output_file, index=False)
    print(f"\n💾 Dados salvos: {output_file}")
    print(f"   Tamanho: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"   Linhas: {len(data)}, Colunas: {len(data.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description="Pré-processamento de dados de telemetria F1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Pré-processar dados de voltas da corrida 1 de 2025
  python cli/preprocess.py --year 2025 --round 1 --laps

  # Pré-processar telemetria de um piloto específico
  python cli/preprocess.py --year 2025 --round 1 --telemetry --driver VER --lap 10

  # Pré-processar tudo (voltas + telemetria)
  python cli/preprocess.py --year 2025 --round 1 --all
        """
    )

    parser.add_argument("--year", type=int, required=True, help="Ano da temporada")
    parser.add_argument("--round", type=int, required=True, help="Número da rodada")
    parser.add_argument("--session", type=str, default="R", choices=["R", "Q", "FP1", "FP2", "FP3", "S"],
                       help="Tipo de sessão (padrão: R)")

    # Tipo de dados para processar
    parser.add_argument("--laps", action="store_true", help="Pré-processar dados de voltas")
    parser.add_argument("--telemetry", action="store_true", help="Pré-processar telemetria")
    parser.add_argument("--all", action="store_true", help="Pré-processar tudo")

    # Filtros
    parser.add_argument("--driver", type=str, help="Filtrar por piloto (ex: VER, HAM)")
    parser.add_argument("--lap", type=int, help="Filtrar por número de volta (apenas para telemetria)")

    # Opções
    parser.add_argument("--track-length", type=float, help="Comprimento da pista em metros (auto-detectado se não fornecido)")
    parser.add_argument("--save", action="store_true", help="Salvar dados processados")
    parser.add_argument("--show-sample", action="store_true", help="Mostrar amostra dos dados em formato de tabela")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🏎️  PRÉ-PROCESSAMENTO DE DADOS F1")
    print("=" * 80)
    print(f"📅 Ano: {args.year}, Rodada: {args.round}, Sessão: {args.session}")

    # Determinar o que processar
    process_laps = args.laps or args.all
    process_telemetry = args.telemetry or args.all

    if not process_laps and not process_telemetry:
        print("\n⚠️  Nenhum tipo de dado especificado. Use --laps, --telemetry ou --all")
        parser.print_help()
        return

    try:
        # Processar voltas
        if process_laps:
            laps_df = load_lap_data(args.year, args.round, args.session)
            enriched_laps = preprocess_lap_features(
                laps_df,
                driver=args.driver,
                show_sample=args.show_sample
            )

            if args.save:
                save_processed_data(enriched_laps, args.year, args.round, args.session, "laps")

        # Processar telemetria
        if process_telemetry:
            telemetry_df = load_telemetry_data(args.year, args.round, args.session)

            # Detectar comprimento da pista se não fornecido
            track_length = args.track_length
            if track_length is None and 'Distance' in telemetry_df.columns:
                track_length = telemetry_df['Distance'].max()
                print(f"\n🏁 Comprimento da pista auto-detectado: {track_length:.0f}m")

            if track_length:
                processed_telemetry = preprocess_telemetry_signals(
                    telemetry_df,
                    track_length=track_length,
                    driver=args.driver,
                    lap_number=args.lap,
                    show_sample=args.show_sample
                )

                # Converter dict para DataFrame para salvar
                if args.save and processed_telemetry:
                    processed_df = pd.DataFrame(processed_telemetry)
                    save_processed_data(processed_df, args.year, args.round, args.session, "telemetry")
            else:
                print("\n⚠️  Comprimento da pista não disponível - pulando processamento de telemetria")

        print("\n" + "=" * 80)
        print("✅ PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}")
        print("\nDados disponíveis:")
        data_dir = Path(__file__).parent.parent / "data" / "raw"
        if data_dir.exists():
            for file in sorted(data_dir.glob("*.parquet")):
                print(f"  - {file.name}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Erro durante pré-processamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
