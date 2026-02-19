#!/usr/bin/env python3
"""
CLI para análise de Machine Learning com tracking MLFlow.

Este script executa análises de ML (clustering + anomaly detection) em dados
de corridas já processados, com tracking completo de métricas via MLFlow.

Usage:
    # Análise completa com MLFlow tracking
    uv run python cli/ml_analysis.py --year 2025 --round 1 --mlflow

    # Análise apenas de clustering
    uv run python cli/ml_analysis.py --year 2025 --round 1 --clustering --mlflow

    # Análise apenas de anomaly detection
    uv run python cli/ml_analysis.py --year 2025 --round 1 --anomaly --mlflow

    # Análise de piloto específico
    uv run python cli/ml_analysis.py --year 2025 --round 1 --driver VER --mlflow

    # Comparar runs anteriores
    uv run python cli/ml_analysis.py --compare --experiment "F1_2025_Round_01"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml import (
    run_race_analysis,
    setup_mlflow,
    compare_runs,
    get_best_run,
)


def parse_args():
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Análise de ML com tracking MLFlow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Análise completa com MLFlow
  uv run python cli/ml_analysis.py --year 2025 --round 1 --mlflow

  # Apenas clustering
  uv run python cli/ml_analysis.py --year 2025 --round 1 --clustering --mlflow

  # Piloto específico
  uv run python cli/ml_analysis.py --year 2025 --round 1 --driver VER --mlflow

  # Comparar runs
  uv run python cli/ml_analysis.py --compare --experiment "F1_2025_Round_01"
        """
    )

    # Grupo: Análise de corrida
    race_group = parser.add_argument_group('Análise de corrida')
    race_group.add_argument(
        '--year',
        type=int,
        help='Ano da corrida (ex: 2025)'
    )
    race_group.add_argument(
        '--round',
        type=int,
        help='Número da rodada (ex: 1 para primeira corrida)'
    )
    race_group.add_argument(
        '--driver',
        type=str,
        help='Código do piloto para análise específica (ex: VER, HAM, LEC)'
    )

    # Grupo: Tipo de análise
    analysis_group = parser.add_argument_group('Tipo de análise')
    analysis_group.add_argument(
        '--clustering',
        action='store_true',
        help='Executar apenas clustering (K-Means)'
    )
    analysis_group.add_argument(
        '--anomaly',
        action='store_true',
        help='Executar apenas detecção de anomalias (Isolation Forest)'
    )

    # Grupo: MLFlow
    mlflow_group = parser.add_argument_group('MLFlow')
    mlflow_group.add_argument(
        '--mlflow',
        action='store_true',
        help='Habilitar tracking com MLFlow'
    )
    mlflow_group.add_argument(
        '--experiment',
        type=str,
        help='Nome do experimento MLFlow (padrão: F1_YEAR_Round_XX)'
    )
    mlflow_group.add_argument(
        '--run-name',
        type=str,
        help='Nome do run MLFlow (padrão: auto-gerado)'
    )

    # Grupo: Comparação de runs
    compare_group = parser.add_argument_group('Comparação de runs')
    compare_group.add_argument(
        '--compare',
        action='store_true',
        help='Comparar runs anteriores'
    )
    compare_group.add_argument(
        '--best',
        action='store_true',
        help='Mostrar melhor run baseado em silhouette score'
    )
    compare_group.add_argument(
        '--max-runs',
        type=int,
        default=10,
        help='Número máximo de runs para comparar (padrão: 10)'
    )

    # Grupo: Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--save',
        action='store_true',
        help='Salvar resultados em data/ml/'
    )
    output_group.add_argument(
        '--show-metrics',
        action='store_true',
        help='Mostrar métricas detalhadas no terminal'
    )

    return parser.parse_args()


def load_race_data(year: int, round_number: int) -> pd.DataFrame | None:
    """
    Carrega dados processados de uma corrida.

    Args:
        year: Ano da corrida
        round_number: Número da rodada

    Returns:
        DataFrame com dados de voltas ou None se não encontrado
    """
    processed_dir = Path("data/processed/races")

    # Tentar carregar laps processados
    laps_file = processed_dir / f"{year}/round_{round_number:02d}/laps_processed.parquet"

    if not laps_file.exists():
        print(f"❌ Arquivo não encontrado: {laps_file}")
        print(f"   Execute primeiro: uv run python cli/pipeline.py {year} {round_number}")
        return None

    print(f"📂 Carregando dados de: {laps_file}")
    laps_df = pd.read_parquet(laps_file)

    # Adicionar Year e Round se não existirem
    if 'Year' not in laps_df.columns:
        laps_df['Year'] = year
    if 'Round' not in laps_df.columns:
        laps_df['Round'] = round_number

    return laps_df


def save_results(results: dict, year: int, round_number: int, driver: str | None = None):
    """
    Salva resultados da análise.

    Args:
        results: Dicionário com resultados da análise
        year: Ano da corrida
        round_number: Número da rodada
        driver: Código do piloto (opcional)
    """
    ml_dir = Path("data/ml/races")

    # Criar diretório de saída
    output_dir = ml_dir / f"{year}/round_{round_number:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sufixo para piloto específico
    suffix = f"_{driver}" if driver else ""

    # Salvar DataFrames
    if 'laps_clustered' in results:
        output_file = output_dir / f"laps_clustered{suffix}.parquet"
        results['laps_clustered'].to_parquet(output_file, index=False)
        print(f"✅ Salvo: {output_file}")

    if 'laps_anomalies' in results:
        output_file = output_dir / f"laps_anomalies{suffix}.parquet"
        results['laps_anomalies'].to_parquet(output_file, index=False)
        print(f"✅ Salvo: {output_file}")

    if 'cluster_statistics' in results:
        output_file = output_dir / f"cluster_statistics{suffix}.parquet"
        results['cluster_statistics'].to_parquet(output_file, index=False)
        print(f"✅ Salvo: {output_file}")

    if 'summary' in results:
        output_file = output_dir / f"analysis_summary{suffix}.parquet"
        results['summary'].to_parquet(output_file, index=False)
        print(f"✅ Salvo: {output_file}")


def print_metrics(results: dict):
    """
    Imprime métricas de forma formatada.

    Args:
        results: Dicionário com resultados da análise
    """
    print("\n" + "="*60)
    print("📊 MÉTRICAS DA ANÁLISE")
    print("="*60)

    # Summary
    if 'summary' in results:
        print("\n📋 Sumário Geral:")
        for col in results['summary'].columns:
            value = results['summary'][col].iloc[0]
            print(f"   {col}: {value}")

    # Clustering metrics
    if 'clustering_metrics' in results:
        print("\n🎯 Métricas de Clustering:")
        metrics_df = results['clustering_metrics']
        for col in metrics_df.columns:
            value = metrics_df[col].iloc[0]
            if isinstance(value, float):
                print(f"   {col}: {value:.4f}")
            else:
                print(f"   {col}: {value}")

    # Anomaly metrics
    if 'anomaly_metrics' in results:
        print("\n🔍 Métricas de Detecção de Anomalias:")
        metrics_df = results['anomaly_metrics']
        for col in metrics_df.columns:
            value = metrics_df[col].iloc[0]
            if isinstance(value, float):
                print(f"   {col}: {value:.4f}")
            else:
                print(f"   {col}: {value}")

    # Cluster statistics
    if 'cluster_statistics' in results:
        print("\n📈 Estatísticas por Cluster:")
        print(results['cluster_statistics'].to_string(index=False))

    print("\n" + "="*60)


def compare_experiment_runs(experiment_name: str, max_runs: int = 10):
    """
    Compara runs de um experimento.

    Args:
        experiment_name: Nome do experimento
        max_runs: Número máximo de runs para comparar
    """
    print(f"\n🔬 Comparando runs do experimento: {experiment_name}")
    print("="*60)

    comparison = compare_runs(
        experiment_name=experiment_name,
        metric_names=['silhouette_score', 'davies_bouldin_score', 'n_anomalies', 'anomaly_rate'],
        max_runs=max_runs
    )

    if comparison.empty:
        print("❌ Nenhum run encontrado para este experimento.")
        return

    # Mostrar comparação
    print("\n📊 Comparação de Runs:")
    print(comparison.to_string(index=False))

    # Mostrar melhor run
    best = get_best_run(experiment_name, 'silhouette_score', ascending=False)
    if best:
        print(f"\n🏆 Melhor Run (Silhouette Score):")
        print(f"   Run ID: {best['run_id']}")
        print(f"   Run Name: {best['run_name']}")
        print(f"   Métricas:")
        for metric, value in best['metrics'].items():
            if isinstance(value, float):
                print(f"      {metric}: {value:.4f}")
            else:
                print(f"      {metric}: {value}")


def main():
    """Função principal."""
    args = parse_args()

    # Modo comparação
    if args.compare:
        if not args.experiment:
            print("❌ Erro: --experiment é requerido para comparação")
            sys.exit(1)

        compare_experiment_runs(args.experiment, args.max_runs)
        return

    # Modo análise
    if not args.year or not args.round:
        print("❌ Erro: --year e --round são requeridos")
        sys.exit(1)

    # Carregar dados
    laps_df = load_race_data(args.year, args.round)
    if laps_df is None:
        sys.exit(1)

    print(f"📊 Total de voltas carregadas: {len(laps_df)}")
    if 'Driver' in laps_df.columns:
        print(f"👥 Pilotos: {laps_df['Driver'].nunique()}")

    # Determinar tipo de análise
    if args.clustering and not args.anomaly:
        analysis_type = 'clustering'
    elif args.anomaly and not args.clustering:
        analysis_type = 'anomaly'
    else:
        analysis_type = 'all'

    print(f"🔬 Tipo de análise: {analysis_type}")

    # Determinar nome do experimento MLFlow
    experiment_name = args.experiment
    if args.mlflow and experiment_name is None:
        experiment_name = f"F1_{args.year}_Round_{args.round:02d}"

    # Executar análise
    print("\n⚙️  Executando análise de ML...")
    results = run_race_analysis(
        laps_df=laps_df,
        analysis_type=analysis_type,
        driver=args.driver,
        enable_mlflow=args.mlflow,
        experiment_name=experiment_name,
        run_name=args.run_name,
    )

    # Mostrar métricas
    if args.show_metrics:
        print_metrics(results)
    else:
        # Mostrar sumário básico
        print("\n✅ Análise concluída!")
        if 'summary' in results:
            print("\n📋 Sumário:")
            print(results['summary'].to_string(index=False))

    # MLFlow run ID
    if 'mlflow_run_id' in results:
        print(f"\n📊 MLFlow Run ID: {results['mlflow_run_id']}")
        print(f"   Experimento: {experiment_name}")
        print(f"\n💡 Para visualizar no MLFlow UI:")
        print(f"   mlflow ui")
        print(f"   Depois acesse: http://localhost:5000")

    # Salvar resultados
    if args.save:
        print("\n💾 Salvando resultados...")
        save_results(results, args.year, args.round, args.driver)

    print("\n✨ Concluído!")


if __name__ == "__main__":
    main()
