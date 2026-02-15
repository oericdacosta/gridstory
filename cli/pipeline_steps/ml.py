"""
Módulo de Machine Learning para o pipeline.

Gerencia a Fase 3: Pré-processamento ML + Clustering + Detecção de Anomalias.
"""

from pathlib import Path

import pandas as pd

from src.preprocessing.feature_engineering import (
    impute_missing_values,
    encode_categorical_variables,
    scale_features,
)
from src.ml.clustering import cluster_laps_kmeans
from src.ml.anomaly_detection import detect_anomalies_isolation_forest, summarize_anomalies
from .reporting import Reporter


def run_ml_phase(
    processed_dir: Path,
    year: int,
    round_num: int,
    show_sample: bool = False,
) -> Path:
    """
    Executa a fase de Machine Learning.

    Args:
        processed_dir: Diretório com dados pré-processados
        year: Ano da temporada
        round_num: Número da rodada
        show_sample: Se deve mostrar amostras dos resultados

    Returns:
        Path para o diretório com resultados de ML
    """
    reporter = Reporter("FASE 3: MACHINE LEARNING (SCIKIT-LEARN)")
    reporter.header()

    # Setup diretórios
    ml_dir = Path("data/ml/races") / f"{year}" / f"round_{round_num:02d}"
    ml_dir.mkdir(parents=True, exist_ok=True)

    # Verificar se laps processados existem
    laps_processed_file = processed_dir / "laps_processed.parquet"
    if not laps_processed_file.exists():
        reporter.info("⚠️  Laps processados não encontrados. Pulando ML.", indent=0)
        return ml_dir

    reporter.section("", "Carregando laps processados para análise ML")
    laps_processed = pd.read_parquet(laps_processed_file)
    reporter.info(f"{len(laps_processed)} voltas carregadas")

    # Pré-processar para ML
    laps_scaled = _preprocess_for_ml(laps_processed, reporter)

    # Executar análises
    _run_clustering(laps_scaled, ml_dir, reporter, show_sample)
    _run_anomaly_detection(laps_scaled, ml_dir, reporter, show_sample)

    reporter.success(f"Machine Learning concluído: {ml_dir}", indent=0)

    return ml_dir


def _preprocess_for_ml(df: pd.DataFrame, reporter: Reporter) -> pd.DataFrame:
    """Aplica pré-processamento necessário para ML (imputação, encoding, scaling)."""
    reporter.section("3.1", "Pré-processamento para ML")

    # 3.1.1: Imputação de valores faltantes
    numeric_cols = ['LapTime_seconds', 'Sector1Time_seconds', 'Sector2Time_seconds',
                   'Sector3Time_seconds', 'TyreLife']
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]

    df_imputed = impute_missing_values(
        df,
        numeric_columns=existing_numeric_cols,
        strategy='median',
        use_knn=False
    )
    reporter.success("Imputação concluída (estratégia: median)")

    # 3.1.2: Encoding de variáveis categóricas
    categorical_cols = ['Compound', 'TrackStatus']
    existing_categorical_cols = [col for col in categorical_cols if col in df_imputed.columns]

    if existing_categorical_cols:
        df_encoded = encode_categorical_variables(
            df_imputed,
            categorical_columns=existing_categorical_cols,
            drop_first=True
        )
        reporter.success(f"Encoding concluído ({', '.join(existing_categorical_cols)})")
    else:
        df_encoded = df_imputed

    # 3.1.3: Escalonamento de features
    scale_cols = [col for col in existing_numeric_cols if col in df_encoded.columns]
    df_scaled = scale_features(
        df_encoded,
        numeric_columns=scale_cols,
        scaler_type='robust'
    )
    reporter.success("Escalonamento concluído (scaler: robust)")

    return df_scaled


def _run_clustering(
    df: pd.DataFrame,
    ml_dir: Path,
    reporter: Reporter,
    show_sample: bool,
):
    """Executa análise de clustering (K-Means)."""
    reporter.section("3.2", "Clustering (K-Means) - Análise de Ritmo")

    feature_cols = [col for col in ['LapTime_seconds', 'Sector1Time_seconds']
                   if col in df.columns]

    if len(feature_cols) < 1:
        reporter.info("⚠️  Features insuficientes para clustering")
        return

    df_clustered = cluster_laps_kmeans(
        df,
        feature_columns=feature_cols,
        n_clusters=None,  # Auto-detect usando silhouette
        group_by='Driver' if 'Driver' in df.columns else None
    )

    # Salvar resultados
    output_file = ml_dir / "laps_clustered.parquet"
    df_clustered.to_parquet(output_file, index=False)

    n_clusters = df_clustered['cluster_label'].nunique()
    reporter.success(f"Clustering concluído: {output_file}")
    reporter.metric("Clusters identificados", n_clusters)

    if show_sample and 'Driver' in df_clustered.columns:
        print(f"\n   Ritmos identificados por piloto (amostra):")
        sample_drivers = df_clustered['Driver'].unique()[:3]
        for driver in sample_drivers:
            driver_data = df_clustered[df_clustered['Driver'] == driver]
            cluster_stats = driver_data.groupby('cluster_label')['LapTime_seconds'].agg(['mean', 'count'])
            print(f"\n   {driver}:")
            for cluster_id, row in cluster_stats.iterrows():
                print(f"      Cluster {cluster_id}: {row['mean']:.3f}s (n={int(row['count'])} voltas)")


def _run_anomaly_detection(
    df: pd.DataFrame,
    ml_dir: Path,
    reporter: Reporter,
    show_sample: bool,
):
    """Executa detecção de anomalias (Isolation Forest)."""
    reporter.section("3.3", "Detecção de Anomalias (Isolation Forest)")

    feature_cols = [col for col in ['LapTime_seconds', 'Sector1Time_seconds']
                   if col in df.columns]

    if len(feature_cols) < 1:
        reporter.info("⚠️  Features insuficientes para detecção de anomalias")
        return

    # contamination e outros parâmetros carregados de config.yaml
    df_anomalies = detect_anomalies_isolation_forest(
        df,
        feature_columns=feature_cols,
        group_by='Driver' if 'Driver' in df.columns else None,
        return_scores=True
    )

    # Salvar resultados
    output_file = ml_dir / "laps_anomalies.parquet"
    df_anomalies.to_parquet(output_file, index=False)

    n_anomalies = df_anomalies['is_anomaly'].sum()
    anomaly_rate = 100 * n_anomalies / len(df_anomalies)

    reporter.success(f"Detecção de anomalias concluída: {output_file}")
    reporter.metric("Anomalias detectadas", f"{n_anomalies}/{len(df_anomalies)} ({anomaly_rate:.2f}%)")

    # Sumário por piloto
    if 'Driver' in df_anomalies.columns:
        summary = summarize_anomalies(df_anomalies, group_by='Driver')
        summary_file = ml_dir / "anomalies_summary.parquet"
        summary.to_parquet(summary_file, index=False)
        reporter.metric("Sumário salvo", summary_file)

        if show_sample:
            print(f"\n   Anomalias por piloto (top 5):")
            top5 = summary.nlargest(5, 'anomalies_count')
            for _, row in top5.iterrows():
                driver_col = 'Driver' if 'Driver' in row.index else row.index[0]
                print(f"      {row[driver_col]}: {int(row['anomalies_count'])} anomalias ({row['anomaly_rate']:.2f}%)")

            # Mostrar algumas voltas anômalas
            if n_anomalies > 0:
                print(f"\n   Exemplos de voltas anômalas:")
                anomaly_examples = df_anomalies[df_anomalies['is_anomaly']].nsmallest(5, 'anomaly_score')
                cols_to_show = ['Driver', 'LapNumber', 'LapTime_seconds', 'anomaly_score']
                available_cols = [c for c in cols_to_show if c in anomaly_examples.columns]
                print(anomaly_examples[available_cols].to_string(index=False))
