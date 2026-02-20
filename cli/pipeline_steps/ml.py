"""
Módulo de Machine Learning para o pipeline.

Wrapper fino sobre src.ml.pipeline.run_race_analysis() — fonte única de verdade
para clustering (K-Means/DBSCAN), anomaly detection (Isolation Forest),
change point detection (Ruptures/PELT) e tracking MLFlow (config-driven).
"""

from pathlib import Path

import pandas as pd

from src.ml.pipeline import run_race_analysis
from src.ml.anomaly_detection import summarize_anomalies
from src.ml.change_point import summarize_cliffs
from .reporting import Reporter


def run_ml_phase(
    processed_dir: Path,
    year: int,
    round_num: int,
    show_sample: bool = False,
) -> Path:
    """
    Executa a fase de Machine Learning.

    Delega toda a lógica de ML para run_race_analysis() (fonte única de verdade).
    O tracking MLFlow é config-driven: habilitado via mlflow.enabled em config.yaml.

    Args:
        processed_dir: Diretório com dados pré-processados
        year: Ano da temporada
        round_num: Número da rodada
        show_sample: Se deve mostrar amostras dos resultados

    Returns:
        Path para o diretório com resultados de ML
    """
    reporter = Reporter("FASE 3: MACHINE LEARNING (SCIKIT-LEARN + RUPTURES)")
    reporter.header()

    ml_dir = Path("data/ml/races") / f"{year}" / f"round_{round_num:02d}"
    ml_dir.mkdir(parents=True, exist_ok=True)

    laps_processed_file = processed_dir / "laps_processed.parquet"
    if not laps_processed_file.exists():
        reporter.info("⚠️  Laps processados não encontrados. Pulando ML.", indent=0)
        return ml_dir

    reporter.section("", "Carregando laps processados para análise ML")
    laps_df = pd.read_parquet(laps_processed_file)
    reporter.info(f"{len(laps_df)} voltas carregadas")

    # Executar pipeline completo (clustering + anomaly + changepoint + MLflow via config)
    reporter.section("3.1–3.4", "Executando pipeline de ML")
    results = run_race_analysis(
        laps_df=laps_df,
        analysis_type='all',
        year=year,
        round_number=round_num,
        # enable_mlflow=None → lê mlflow.enabled do config.yaml automaticamente
    )

    if 'mlflow_run_id' in results:
        reporter.info(f"MLFlow run: {results['mlflow_run_id']}")

    # Salvar resultados como parquet para steps downstream
    _save_results(results, ml_dir, reporter)

    # Reportar métricas
    _report_metrics(results, reporter, show_sample)

    reporter.success(f"Machine Learning concluído: {ml_dir}", indent=0)
    return ml_dir


def _save_results(results: dict, ml_dir: Path, reporter: Reporter) -> None:
    """Salva DataFrames de resultado como parquet."""
    files = {
        'laps_clustered':     'laps_clustered.parquet',
        'laps_anomalies':     'laps_anomalies.parquet',
        'cluster_statistics': 'cluster_statistics.parquet',
        'laps_changepoints':  'laps_changepoints.parquet',
        'tire_cliffs':        'tire_cliffs.parquet',
        'tire_cliffs_summary': 'tire_cliffs_summary.parquet',
    }
    for key, filename in files.items():
        if key in results and results[key] is not None:
            out = ml_dir / filename
            results[key].to_parquet(out, index=False)
            reporter.success(f"Salvo: {out}")

    # Sumário de anomalias por piloto
    if 'laps_anomalies' in results and 'Driver' in results['laps_anomalies'].columns:
        summary = summarize_anomalies(results['laps_anomalies'], group_by='Driver')
        out = ml_dir / 'anomalies_summary.parquet'
        summary.to_parquet(out, index=False)
        reporter.success(f"Salvo: {out}")


def _report_metrics(results: dict, reporter: Reporter, show_sample: bool) -> None:
    """Reporta métricas principais no terminal."""
    # Clustering
    if 'laps_clustered' in results:
        n_clusters = results['laps_clustered']['cluster_label'].nunique()
        reporter.metric("Clusters identificados", n_clusters)

        if show_sample and 'Driver' in results['laps_clustered'].columns:
            df = results['laps_clustered']
            print("\n   Ritmos por piloto (amostra):")
            for driver in df['Driver'].unique()[:3]:
                stats = df[df['Driver'] == driver].groupby('cluster_label')['LapTime_seconds'].agg(['mean', 'count'])
                print(f"\n   {driver}:")
                for cluster_id, row in stats.iterrows():
                    semantic = df[df['cluster_label'] == cluster_id]['cluster_semantic'].iloc[0] \
                        if 'cluster_semantic' in df.columns else str(cluster_id)
                    print(f"      [{semantic}] {row['mean']:.3f}s (n={int(row['count'])})")

    # Anomalias
    if 'laps_anomalies' in results:
        df = results['laps_anomalies']
        n_anomalies = int(df['is_anomaly'].sum())
        anomaly_rate = 100 * n_anomalies / len(df)
        reporter.metric("Anomalias detectadas", f"{n_anomalies}/{len(df)} ({anomaly_rate:.2f}%)")

        if show_sample and 'Driver' in df.columns:
            summary = summarize_anomalies(df, group_by='Driver')
            print("\n   Anomalias por piloto (top 5):")
            for _, row in summary.nlargest(5, 'anomalies_count').iterrows():
                print(f"      {row['Driver']}: {int(row['anomalies_count'])} ({row['anomaly_rate']:.2f}%)")

    # Change points
    if 'tire_cliffs' in results:
        df = results['tire_cliffs']
        n_cliffs = int(df['has_cliff'].sum())
        n_stints = len(df)
        cliff_rate = 100 * n_cliffs / n_stints if n_stints > 0 else 0
        reporter.metric("Tire cliffs detectados", f"{n_cliffs}/{n_stints} stints ({cliff_rate:.1f}%)")

        if show_sample and 'tire_cliffs_summary' in results:
            summary = results['tire_cliffs_summary']
            print("\n   Cliffs por piloto:")
            for _, row in summary.iterrows():
                print(f"      {row['Driver']}: {int(row['stints_with_cliff'])}/{int(row['total_stints'])} stints "
                      f"({row['cliff_rate_pct']:.0f}%) — validados: {int(row['cliffs_validated'])}")
