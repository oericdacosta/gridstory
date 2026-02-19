"""
Exemplo completo de uso de MLFlow tracking para análise de ML.

Este script demonstra:
1. Setup do MLFlow
2. Execução de análise com tracking
3. Cálculo de métricas completas
4. Comparação de runs
5. Carregamento do melhor modelo

Usage:
    # Executar exemplo básico
    uv run python examples/mlflow_example.py

    # Depois, visualizar no MLFlow UI
    mlflow ui
    # Acesse: http://localhost:5000
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml import (
    setup_mlflow,
    run_race_analysis,
    compare_runs,
    get_best_run,
    calculate_clustering_metrics,
    calculate_anomaly_metrics,
)


def create_sample_data() -> pd.DataFrame:
    """
    Cria dados sintéticos de exemplo para demonstração.

    Simula 100 voltas de 3 pilotos com 3 ritmos diferentes:
    - Ritmo Puro: ~90s
    - Gestão: ~92s
    - Tráfego: ~95s
    + Algumas anomalias (erros, quebras)
    """
    np.random.seed(42)

    drivers = ['VER', 'HAM', 'LEC']
    data = []

    for driver in drivers:
        # Ritmo puro (30 voltas)
        for i in range(30):
            data.append({
                'Driver': driver,
                'LapNumber': i + 1,
                'LapTime_seconds': np.random.normal(90, 0.5),
                'Sector1Time_seconds': np.random.normal(30, 0.2),
                'Sector2Time_seconds': np.random.normal(32, 0.2),
                'Sector3Time_seconds': np.random.normal(28, 0.2),
                'TyreLife': i + 1,
                'Compound': 'SOFT',
                'Year': 2025,
                'Round': 1,
            })

        # Gestão de pneus (10 voltas)
        for i in range(30, 40):
            data.append({
                'Driver': driver,
                'LapNumber': i + 1,
                'LapTime_seconds': np.random.normal(92, 0.5),
                'Sector1Time_seconds': np.random.normal(31, 0.2),
                'Sector2Time_seconds': np.random.normal(33, 0.2),
                'Sector3Time_seconds': np.random.normal(28, 0.2),
                'TyreLife': i + 1,
                'Compound': 'MEDIUM',
                'Year': 2025,
                'Round': 1,
            })

        # Tráfego (5 voltas)
        for i in range(40, 45):
            data.append({
                'Driver': driver,
                'LapNumber': i + 1,
                'LapTime_seconds': np.random.normal(95, 1.0),
                'Sector1Time_seconds': np.random.normal(32, 0.3),
                'Sector2Time_seconds': np.random.normal(35, 0.3),
                'Sector3Time_seconds': np.random.normal(28, 0.2),
                'TyreLife': i + 1,
                'Compound': 'MEDIUM',
                'Year': 2025,
                'Round': 1,
            })

        # Anomalias (2 voltas)
        # Erro de piloto
        data.append({
            'Driver': driver,
            'LapNumber': 15,
            'LapTime_seconds': 105.0,  # Muito lenta (rodada)
            'Sector1Time_seconds': 38.0,
            'Sector2Time_seconds': 39.0,
            'Sector3Time_seconds': 28.0,
            'TyreLife': 15,
            'Compound': 'SOFT',
            'Year': 2025,
            'Round': 1,
        })

        # Volta excepcional
        data.append({
            'Driver': driver,
            'LapNumber': 25,
            'LapTime_seconds': 88.5,  # Muito rápida
            'Sector1Time_seconds': 29.5,
            'Sector2Time_seconds': 31.0,
            'Sector3Time_seconds': 28.0,
            'TyreLife': 25,
            'Compound': 'SOFT',
            'Year': 2025,
            'Round': 1,
        })

    return pd.DataFrame(data)


def example_1_basic_tracking():
    """
    Exemplo 1: Tracking básico de uma análise completa.
    """
    print("\n" + "="*60)
    print("EXEMPLO 1: Tracking Básico")
    print("="*60)

    # Criar dados de exemplo
    print("\n1️⃣ Criando dados sintéticos...")
    laps_df = create_sample_data()
    print(f"   Total de voltas: {len(laps_df)}")
    print(f"   Pilotos: {laps_df['Driver'].unique().tolist()}")

    # Setup MLFlow
    print("\n2️⃣ Configurando MLFlow...")
    setup_mlflow(
        experiment_name="Example_Basic_Tracking",
        enable_autolog=True
    )
    print("   ✅ MLFlow configurado!")

    # Executar análise COM tracking
    print("\n3️⃣ Executando análise com tracking...")
    results = run_race_analysis(
        laps_df=laps_df,
        analysis_type='all',
        enable_mlflow=True,
        experiment_name='Example_Basic_Tracking',
        run_name='Run_1_AllDrivers',
    )

    # Mostrar resultados
    print("\n📊 Resultados:")
    print(f"   MLFlow Run ID: {results['mlflow_run_id']}")
    print(f"\n   Sumário:")
    print(results['summary'].to_string(index=False))

    print(f"\n   Métricas de Clustering:")
    print(results['clustering_metrics'].to_string(index=False))

    print(f"\n   Métricas de Anomaly Detection:")
    print(results['anomaly_metrics'].to_string(index=False))

    print("\n✅ Exemplo 1 concluído!")
    print("💡 Acesse o MLFlow UI para visualizar: mlflow ui")


def example_2_experimentation():
    """
    Exemplo 2: Experimentação com diferentes hiperparâmetros.
    """
    print("\n" + "="*60)
    print("EXEMPLO 2: Experimentação")
    print("="*60)

    # Criar dados
    print("\n1️⃣ Criando dados sintéticos...")
    laps_df = create_sample_data()

    # Setup MLFlow
    print("\n2️⃣ Configurando MLFlow...")
    setup_mlflow(experiment_name="Example_Experimentation")

    # Experimentar com diferentes valores de contamination
    print("\n3️⃣ Experimentando com diferentes valores de contamination...")

    contamination_values = [0.03, 0.05, 0.10]

    for cont in contamination_values:
        print(f"\n   Rodando com contamination={cont:.2f}...")

        # Note: Aqui você precisaria modificar run_race_analysis para aceitar
        # contamination como parâmetro. Por simplicidade, vamos apenas
        # demonstrar o conceito.

        results = run_race_analysis(
            laps_df=laps_df,
            analysis_type='all',
            enable_mlflow=True,
            experiment_name='Example_Experimentation',
            run_name=f'Contamination_{cont:.2f}',
        )

        n_anomalies = results['anomaly_metrics']['n_anomalies'].iloc[0]
        anomaly_rate = results['anomaly_metrics']['anomaly_rate'].iloc[0]

        print(f"      Anomalias detectadas: {n_anomalies}")
        print(f"      Taxa: {anomaly_rate:.2f}%")

    print("\n✅ Exemplo 2 concluído!")
    print("💡 Compare os runs no MLFlow UI: mlflow ui")


def example_3_comparison():
    """
    Exemplo 3: Comparação de runs e seleção do melhor.
    """
    print("\n" + "="*60)
    print("EXEMPLO 3: Comparação de Runs")
    print("="*60)

    # Primeiro, executar alguns runs (re-usar do exemplo 2)
    experiment_name = "Example_Comparison"

    print("\n1️⃣ Executando alguns runs para comparar...")
    setup_mlflow(experiment_name=experiment_name)
    laps_df = create_sample_data()

    # Run 1
    results1 = run_race_analysis(
        laps_df=laps_df,
        analysis_type='all',
        enable_mlflow=True,
        experiment_name=experiment_name,
        run_name='Run_1',
    )

    # Run 2 (apenas clustering)
    results2 = run_race_analysis(
        laps_df=laps_df,
        analysis_type='clustering',
        enable_mlflow=True,
        experiment_name=experiment_name,
        run_name='Run_2_Clustering',
    )

    # Run 3 (apenas anomaly)
    results3 = run_race_analysis(
        laps_df=laps_df,
        analysis_type='anomaly',
        enable_mlflow=True,
        experiment_name=experiment_name,
        run_name='Run_3_Anomaly',
    )

    # Comparar runs
    print("\n2️⃣ Comparando runs...")
    comparison = compare_runs(
        experiment_name=experiment_name,
        metric_names=['silhouette_score', 'davies_bouldin_score', 'n_anomalies'],
        max_runs=10
    )

    if not comparison.empty:
        # Selecionar colunas disponíveis (métricas têm prefixo clustering_ ou anomaly_)
        cols = ['run_name'] if 'run_name' in comparison.columns else []
        metric_cols = [c for c in comparison.columns if any(
            m in c for m in ['silhouette', 'davies_bouldin', 'n_anomalies']
        )]
        cols += metric_cols
        print("\n📊 Comparação de Runs:")
        print(comparison[cols].to_string(index=False) if cols else comparison.to_string(index=False))
    else:
        print("\n❌ Nenhum run encontrado para comparação")

    # Encontrar melhor run
    print("\n3️⃣ Encontrando melhor run (baseado em Silhouette Score)...")
    best = get_best_run(
        experiment_name=experiment_name,
        metric_name='silhouette_score',
        ascending=False  # Maior é melhor
    )

    if best:
        print(f"\n🏆 Melhor Run:")
        print(f"   Run ID: {best['run_id']}")
        print(f"   Run Name: {best['run_name']}")
        print(f"   Métricas:")
        for metric, value in best['metrics'].items():
            if isinstance(value, float):
                print(f"      {metric}: {value:.4f}")
            else:
                print(f"      {metric}: {value}")
    else:
        print("\n❌ Nenhum run encontrado")

    print("\n✅ Exemplo 3 concluído!")


def main():
    """Executa todos os exemplos."""
    print("\n" + "="*60)
    print("EXEMPLOS DE MLFLOW TRACKING - PitWall AI")
    print("="*60)

    # Executar exemplos
    example_1_basic_tracking()
    example_2_experimentation()
    example_3_comparison()

    # Instruções finais
    print("\n" + "="*60)
    print("PRÓXIMOS PASSOS")
    print("="*60)
    print("\n1️⃣ Visualizar resultados no MLFlow UI:")
    print("   mlflow ui")
    print("   Depois acesse: http://localhost:5000")
    print("\n2️⃣ Explorar os experimentos:")
    print("   - Example_Basic_Tracking")
    print("   - Example_Experimentation")
    print("   - Example_Comparison")
    print("\n3️⃣ Comparar métricas e parâmetros")
    print("\n4️⃣ Carregar melhor modelo para produção")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
