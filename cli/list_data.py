#!/usr/bin/env python3
"""
Script auxiliar para listar dados disponíveis.
"""

from pathlib import Path
import pandas as pd


def list_available_data():
    """Lista todos os dados disponíveis no projeto."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    print("\n" + "=" * 80)
    print("📊 DADOS DISPONÍVEIS NO PROJETO")
    print("=" * 80)

    # Dados brutos
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        print("\n📁 DADOS BRUTOS (data/raw/):")

        # Verificar estrutura hierárquica (races/YEAR/round_XX/)
        races_dir = raw_dir / "races"
        if races_dir.exists():
            for year_dir in sorted(races_dir.iterdir()):
                if not year_dir.is_dir():
                    continue

                print(f"\n  📅 Temporada: {year_dir.name}")

                for round_dir in sorted(year_dir.iterdir()):
                    if not round_dir.is_dir():
                        continue

                    print(f"\n    🏁 {round_dir.name}:")

                    # Laps
                    laps_file = round_dir / "laps.parquet"
                    if laps_file.exists():
                        df = pd.read_parquet(laps_file)
                        size_kb = laps_file.stat().st_size / 1024
                        print(f"      📊 laps.parquet ({size_kb:.1f} KB)")
                        print(f"         {len(df):,} voltas, {len(df.columns)} colunas")
                        if 'Driver' in df.columns:
                            drivers = df['Driver'].unique()
                            print(f"         {len(drivers)} pilotos: {', '.join(sorted(drivers)[:5])}...")

                    # Telemetry
                    telemetry_dir = round_dir / "telemetry"
                    if telemetry_dir.exists():
                        telemetry_files = list(telemetry_dir.glob("*.parquet"))
                        print(f"      📈 telemetry/ ({len(telemetry_files)} pilotos)")

                        # Amostra de um arquivo
                        if telemetry_files:
                            sample = pd.read_parquet(telemetry_files[0])
                            channels = [col for col in sample.columns if col in ['Speed', 'RPM', 'Throttle', 'Brake', 'DRS', 'Distance']]
                            print(f"         Canais: {', '.join(channels)}")
                            print(f"         ~{len(sample):,} pontos por piloto")

                    # Outros arquivos
                    for other_file in ['results.parquet', 'race_control.parquet', 'weather.parquet']:
                        file_path = round_dir / other_file
                        if file_path.exists():
                            size_kb = file_path.stat().st_size / 1024
                            print(f"      📄 {other_file} ({size_kb:.1f} KB)")

        # Verificar arquivos flat na raiz
        flat_files = sorted(raw_dir.glob("*.parquet"))
        if flat_files:
            print("\n  📄 Arquivos na raiz:")
            for file in flat_files:
                size_kb = file.stat().st_size / 1024
                df = pd.read_parquet(file)
                print(f"    {file.name} ({size_kb:.1f} KB, {len(df):,} linhas)")

        # Calendários
        calendar_dir = raw_dir / "calendar"
        if calendar_dir.exists():
            print("\n  📅 Calendários:")
            for cal_file in sorted(calendar_dir.glob("*.parquet")):
                df = pd.read_parquet(cal_file)
                print(f"    {cal_file.name}: {len(df)} corridas")

        if not races_dir.exists() and not flat_files and not (raw_dir / "calendar").exists():
            print("  (vazio)")
    else:
        print("\n📁 DADOS BRUTOS: pasta não existe")

    # Dados processados
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        print("\n\n📁 DADOS PRÉ-PROCESSADOS (data/processed/):")
        files = sorted(processed_dir.glob("*.parquet"))

        if files:
            for file in files:
                size_kb = file.stat().st_size / 1024
                df = pd.read_parquet(file)
                print(f"\n  ✨ {file.name}")
                print(f"     Tamanho: {size_kb:.1f} KB")
                print(f"     Linhas: {len(df):,}, Colunas: {len(df.columns)}")

                # Mostrar colunas de features adicionadas
                feature_cols = [col for col in df.columns if any(x in col for x in ['z_score', 'degradation', 'derivative', 'group_'])]
                if feature_cols:
                    print(f"     Features: {', '.join(feature_cols[:3])}...")
        else:
            print("  (vazio)")
    else:
        print("\n\n📁 DADOS PRÉ-PROCESSADOS: pasta não existe")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    list_available_data()
