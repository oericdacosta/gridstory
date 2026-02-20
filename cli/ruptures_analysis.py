#!/usr/bin/env python3
"""
CLI para análise de Change Point Detection (Ruptures/PELT) com tracking MLFlow.

Detecta tire cliffs (mudanças de regime de degradação de pneu) dentro de cada stint,
usando o algoritmo PELT do pacote ruptures.

Entrada: laps_anomalies.parquet (output da análise de anomalias)
Saída:   laps_changepoints.parquet, tire_cliffs.parquet

Usage:
    # Análise sem MLFlow (apenas métricas no terminal)
    uv run python cli/ruptures_analysis.py --year 2025 --round 1 --show-metrics

    # Análise com MLFlow tracking e salvamento
    uv run python cli/ruptures_analysis.py --year 2025 --round 1 --mlflow --save

    # Análise de piloto específico
    uv run python cli/ruptures_analysis.py --year 2025 --round 1 --driver VER --show-metrics

    # Calibração de penalty: testa range definido em config.yaml e loga cada run no MLFlow
    uv run python cli/ruptures_analysis.py --year 2025 --round 1 --penalty-search --mlflow

    # Comparar runs anteriores por cliff_rate
    uv run python cli/ruptures_analysis.py --compare --experiment "F1_2025_Round_01_Ruptures"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml import (
    detect_tire_changepoints,
    summarize_cliffs,
    calculate_changepoint_metrics,
    setup_mlflow,
    track_changepoint_run,
    compare_runs,
    get_best_run,
)
from src.utils.config import get_config


def parse_args():
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Change Point Detection (Ruptures/PELT) com tracking MLFlow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Análise sem MLFlow
  uv run python cli/ruptures_analysis.py --year 2025 --round 1 --show-metrics

  # Com MLFlow e salvamento
  uv run python cli/ruptures_analysis.py --year 2025 --round 1 --mlflow --save

  # Piloto específico
  uv run python cli/ruptures_analysis.py --year 2025 --round 1 --driver VER

  # Calibração de penalty (testa range do config.yaml, loga no MLFlow)
  uv run python cli/ruptures_analysis.py --year 2025 --round 1 --penalty-search --mlflow

  # Comparar runs
  uv run python cli/ruptures_analysis.py --compare --experiment "F1_2025_Round_01_Ruptures"
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
        help='Nome do experimento MLFlow (padrão: F1_YEAR_Round_XX_Ruptures)'
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
        help='Comparar runs anteriores por cliff_rate'
    )
    compare_group.add_argument(
        '--max-runs',
        type=int,
        default=10,
        help='Número máximo de runs para comparar (padrão: 10)'
    )

    # Grupo: Parâmetros de análise
    params_group = parser.add_argument_group('Parâmetros de análise')
    params_group.add_argument(
        '--penalty-search',
        action='store_true',
        help='Testar múltiplas penalties (range em config.yaml) e logar cada uma como run MLFlow separado'
    )

    # Grupo: Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--save',
        action='store_true',
        help='Salvar resultados em data/ml/races/YEAR/round_XX/'
    )
    output_group.add_argument(
        '--show-metrics',
        action='store_true',
        help='Mostrar métricas detalhadas no terminal'
    )

    return parser.parse_args()


def load_anomalies_data(year: int, round_number: int, driver: str | None = None) -> pd.DataFrame | None:
    """
    Carrega dados de anomalias de uma corrida.

    Args:
        year: Ano da corrida
        round_number: Número da rodada
        driver: Código do piloto (opcional)

    Returns:
        DataFrame com dados de anomalias ou None se não encontrado
    """
    ml_dir = Path("data/ml/races")
    suffix = f"_{driver}" if driver else ""
    anomalies_file = ml_dir / f"{year}/round_{round_number:02d}/laps_anomalies{suffix}.parquet"

    if not anomalies_file.exists():
        print(f"Arquivo nao encontrado: {anomalies_file}")
        print(f"   Execute primeiro: uv run python cli/ml_analysis.py --year {year} --round {round_number} --anomaly --save")
        return None

    print(f"Carregando dados de: {anomalies_file}")
    df = pd.read_parquet(anomalies_file)

    # Verificar colunas necessárias
    required = ['is_anomaly', 'anomaly_score']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Colunas ausentes no arquivo de anomalias: {missing}")
        print("   Certifique-se de usar --return_scores=True na detecção de anomalias.")
        return None

    # Verificar LapTime_delta
    if 'LapTime_delta' not in df.columns:
        print("Coluna 'LapTime_delta' nao encontrada. Tentando calcular...")
        if 'LapTime_seconds' in df.columns and 'Driver' in df.columns:
            df['LapTime_delta'] = df.groupby('Driver')['LapTime_seconds'].transform(
                lambda x: x - x.median()
            )
            print("   LapTime_delta calculado como desvio em relacao a mediana por piloto.")
        else:
            print("   Impossivel calcular LapTime_delta sem LapTime_seconds e Driver.")
            return None

    return df


def save_results(
    laps_df: pd.DataFrame,
    changepoints_df: pd.DataFrame,
    year: int,
    round_number: int,
    driver: str | None = None,
):
    """Salva resultados da análise de change points."""
    ml_dir = Path("data/ml/races")
    output_dir = ml_dir / f"{year}/round_{round_number:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{driver}" if driver else ""

    laps_file = output_dir / f"laps_changepoints{suffix}.parquet"
    laps_df.to_parquet(laps_file, index=False)
    print(f"Salvo: {laps_file}")

    cliffs_file = output_dir / f"tire_cliffs{suffix}.parquet"
    changepoints_df.to_parquet(cliffs_file, index=False)
    print(f"Salvo: {cliffs_file}")


def print_metrics(metrics: dict, changepoints_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Imprime métricas de forma formatada."""
    print("\n" + "="*60)
    print("METRICAS - CHANGE POINT DETECTION (PELT)")
    print("="*60)

    print("\nMetricas Globais:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    if not summary_df.empty:
        print("\nSumario por Piloto:")
        print(summary_df.to_string(index=False))

    if not changepoints_df.empty:
        cliffs_only = changepoints_df[changepoints_df['has_cliff'] == True]
        if not cliffs_only.empty:
            print(f"\nDetalhes dos Cliffs Detectados ({len(cliffs_only)} stints):")
            cols = ['Driver', 'Stint', 'cliff_lap', 'laps_before_cliff',
                    'cliff_delta_magnitude', 'cliff_validated']
            available_cols = [c for c in cols if c in cliffs_only.columns]
            print(cliffs_only[available_cols].to_string(index=False))

    print("\n" + "="*60)


def compare_experiment_runs(experiment_name: str, max_runs: int = 10):
    """Compara runs de um experimento por cliff_rate."""
    print(f"\nComparando runs do experimento: {experiment_name}")
    print("="*60)

    comparison = compare_runs(
        experiment_name=experiment_name,
        metric_names=['cliff_rate', 'n_cliffs_detected', 'cliff_validated_rate', 'mean_cliff_magnitude'],
        max_runs=max_runs
    )

    if comparison.empty:
        print("Nenhum run encontrado para este experimento.")
        return

    print("\nComparacao de Runs:")
    print(comparison.to_string(index=False))

    # Melhor run por cliff_rate
    best = get_best_run(experiment_name, 'cliff_rate', ascending=False)
    if best:
        print(f"\nMelhor Run (cliff_rate):")
        print(f"   Run ID: {best['run_id']}")
        print(f"   Run Name: {best['run_name']}")
        print(f"   Metricas:")
        for metric, value in best['metrics'].items():
            if isinstance(value, float):
                print(f"      {metric}: {value:.4f}")
            else:
                print(f"      {metric}: {value}")


def main():
    """Função principal."""
    args = parse_args()
    config = get_config()

    # Modo comparação
    if args.compare:
        if not args.experiment:
            print("Erro: --experiment e requerido para comparacao")
            sys.exit(1)
        compare_experiment_runs(args.experiment, args.max_runs)
        return

    # Modo análise
    if not args.year or not args.round:
        print("Erro: --year e --round sao requeridos")
        sys.exit(1)

    # Carregar dados
    df = load_anomalies_data(args.year, args.round, args.driver)
    if df is None:
        sys.exit(1)

    print(f"Total de voltas carregadas: {len(df)}")
    if 'Driver' in df.columns:
        print(f"Pilotos: {df['Driver'].nunique()}")
    if 'Stint' in df.columns:
        print(f"Stints: {df.groupby(['Driver', 'Stint']).ngroups if 'Driver' in df.columns else df['Stint'].nunique()}")

    # Parâmetros base do config
    base_params = {
        'algorithm': config.get_ruptures_algorithm(),
        'model': config.get_ruptures_model(),
        'penalty': config.get_ruptures_penalty(),
        'min_size': config.get_ruptures_min_size(),
        'jump': config.get_ruptures_jump(),
        'min_cliff_magnitude': config.get_ruptures_min_cliff_magnitude(),
        'validation_enabled': config.get_ruptures_validation_enabled(),
        'validation_window': config.get_ruptures_validation_window(),
        'slope_threshold': config.get_ruptures_validation_slope_threshold(),
    }

    # Determinar nome do experimento
    experiment_name = args.experiment
    if args.mlflow and experiment_name is None:
        experiment_name = f"F1_{args.year}_Round_{args.round:02d}_Ruptures"

    if args.mlflow:
        setup_mlflow(experiment_name)

    tags = {
        'year': str(args.year),
        'round': str(args.round),
        'algorithm': base_params['algorithm'],
    }
    if args.driver:
        tags['driver'] = args.driver

    # Modo penalty search: testa múltiplas penalties e loga como runs separados
    if args.penalty_search:
        if not args.mlflow:
            print("Aviso: --penalty-search recomenda --mlflow para comparar runs.")

        penalties = config.get_ruptures_penalty_search_range()
        print(f"\nPenalty search: testando {len(penalties)} valores — {penalties}")

        for pen in penalties:
            params = {**base_params, 'penalty': pen}
            print(f"\n  penalty={pen}...", end=" ")

            laps_df, changepoints_df = detect_tire_changepoints(df, penalty=pen)
            summary_df = summarize_cliffs(changepoints_df)
            metrics = calculate_changepoint_metrics(changepoints_df)

            cliff_rate = metrics['cliff_rate']
            validated = metrics.get('cliff_validated_rate', 0)
            print(f"cliff_rate={cliff_rate:.1f}%, validated={validated:.1f}%")

            if args.mlflow:
                run_name = f"PELT_{base_params['model']}_pen{pen}_{args.year}_R{args.round:02d}"
                if args.driver:
                    run_name += f"_{args.driver}"

                track_changepoint_run(
                    run_name=run_name,
                    changepoints_df=changepoints_df,
                    params=params,
                    tags=tags,
                )

        print(f"\nPenalty search concluida. Use 'mlflow ui' para comparar os {len(penalties)} runs.")
        return

    # Modo normal: executa com penalty do config (ou --penalty-search desativado)
    params = base_params
    print(f"\nExecutando PELT (model={params['model']}, penalty={params['penalty']}, min_size={params['min_size']})...")

    laps_df, changepoints_df = detect_tire_changepoints(df)
    summary_df = summarize_cliffs(changepoints_df)
    metrics = calculate_changepoint_metrics(changepoints_df)

    n_cliffs = metrics['n_cliffs_detected']
    n_stints = metrics['n_stints_analyzed']
    cliff_rate = metrics['cliff_rate']
    print(f"\nAnalise concluida!")
    print(f"   Stints analisados: {n_stints}")
    print(f"   Cliffs detectados: {n_cliffs} ({cliff_rate:.1f}%)")

    if args.show_metrics:
        print_metrics(metrics, changepoints_df, summary_df)

    # MLFlow tracking (modo normal)
    if args.mlflow:
        run_name = args.run_name or f"PELT_{params['model']}_pen{params['penalty']}_{args.year}_R{args.round:02d}"
        if args.driver:
            run_name += f"_{args.driver}"

        artifacts = {
            'tire_cliffs.parquet': changepoints_df,
            'cliff_summary.parquet': summary_df,
        }

        run_id = track_changepoint_run(
            run_name=run_name,
            changepoints_df=changepoints_df,
            params=params,
            artifacts=artifacts,
            tags=tags,
        )

        print(f"\nMLFlow Run ID: {run_id}")
        print(f"   Experimento: {experiment_name}")
        print(f"\nPara visualizar no MLFlow UI:")
        print(f"   mlflow ui")
        print(f"   Depois acesse: http://localhost:5000")

    # Salvar resultados
    if args.save:
        print("\nSalvando resultados...")
        save_results(laps_df, changepoints_df, args.year, args.round, args.driver)

    print("\nConcluido!")


if __name__ == "__main__":
    main()
