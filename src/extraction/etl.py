"""
Módulo ETL (Extract, Transform, Load) para estruturação dos dados.

Responsável por extrair, transformar e carregar dados de corridas de F1,
incluindo voltas, telemetria, clima e resultados.
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


class RaceDataETL:
    """
    Classe responsável pela extração e estruturação dos dados de corrida.

    Extrai os quatro pilares de dados conforme planejamento:
    A. Dados de Voltas e Estratégia (session.laps)
    B. Telemetria do Carro (session.car_data)
    C. Controle de Corrida e Contexto (race_control_messages, weather_data)
    D. Resultados Finais (session.results)
    """

    def __init__(self, session: fastf1.core.Session):
        self.session = session
        self.event_name = session.event["EventName"]
        self.year = session.event["EventDate"].year
        self.round_number = session.event["RoundNumber"]

    def extract_laps_data(self) -> pd.DataFrame:
        """
        A. Extrai dados de voltas e estratégia.

        Parâmetros críticos:
        - LapTime, Sector1Time, Sector2Time, Sector3Time (análise de ritmo)
        - PitInTime, PitOutTime (duração do pit stop)
        - Compound (tipo de pneu)
        - TyreLife (voltas do pneu - crucial para degradação)
        - FreshTyre (pneu novo no início do stint)
        - IsAccurate (filtro de qualidade)
        """
        print("\nExtraindo dados de voltas...")
        df_laps = self.session.laps.copy()

        # Converter Timedelta para segundos (float) para facilitar ML
        time_columns = [
            "LapTime",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
            "PitInTime",
            "PitOutTime",
        ]

        for col in time_columns:
            if col in df_laps.columns:
                df_laps[f"{col}_seconds"] = df_laps[col].dt.total_seconds()

        # Selecionar colunas críticas
        critical_cols = [
            "Driver",
            "DriverNumber",
            "LapNumber",
            "LapTime_seconds",
            "Sector1Time_seconds",
            "Sector2Time_seconds",
            "Sector3Time_seconds",
            "Compound",
            "TyreLife",
            "FreshTyre",
            "IsAccurate",
            "PitInTime_seconds",
            "PitOutTime_seconds",
            "Stint",
            "Team",
            "TrackStatus",
            "Position",
        ]

        # Filtrar apenas colunas que existem
        available_cols = [col for col in critical_cols if col in df_laps.columns]
        df_laps_clean = df_laps[available_cols].copy()

        print(f"  ✓ {len(df_laps_clean)} voltas extraídas")
        print(f"  ✓ {len(df_laps_clean['Driver'].unique())} pilotos")

        return df_laps_clean

    def _get_driver_abbreviations(self) -> Dict[str, str]:
        """
        Cria mapeamento de número do piloto para abreviação.

        Returns:
            Dicionário {driver_number: abbreviation} (ex: {'1': 'VER', '44': 'HAM'})
        """
        driver_map = {}
        results = self.session.results

        for idx, row in results.iterrows():
            driver_number = str(row["DriverNumber"])
            abbreviation = row["Abbreviation"]
            driver_map[driver_number] = abbreviation

        return driver_map

    def extract_telemetry_data(
        self, drivers: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        B. Extrai telemetria sincronizada de cada piloto.

        Parâmetros críticos:
        - Speed, RPM, Throttle, Brake, nGear, DRS

        Args:
            drivers: Lista de abreviações de pilotos (ex: ['VER', 'HAM']).
                    Se None, extrai de todos os pilotos.
        """
        print("\nExtraindo telemetria...")

        # Criar mapeamento número -> abreviação
        driver_map = self._get_driver_abbreviations()

        # Se drivers não especificado, usar todos os pilotos
        if drivers is None:
            # session.drivers retorna números, converter para abreviações
            driver_numbers = self.session.drivers
            drivers_to_process = [
                (num, driver_map.get(num, num)) for num in driver_numbers
            ]
        else:
            # Drivers especificados: assumir que são abreviações
            # Criar mapeamento reverso para encontrar números
            abbr_to_num = {v: k for k, v in driver_map.items()}
            drivers_to_process = [(abbr_to_num.get(drv, drv), drv) for drv in drivers]

        telemetry_dict = {}

        for driver_num, driver_abbr in drivers_to_process:
            try:
                print(f"  Processando piloto: {driver_abbr} (#{driver_num})")

                # Pegar todas as voltas do piloto (usando número)
                driver_laps = self.session.laps.pick_drivers(driver_num)

                if len(driver_laps) == 0:
                    print(f"    ⚠ Sem voltas para {driver_abbr}")
                    continue

                # Pegar telemetria do carro
                car_data = driver_laps.get_car_data()

                # Adicionar distância para referência espacial comum
                car_data = car_data.add_distance()

                # Selecionar colunas críticas de telemetria
                telemetry_cols = [
                    "Time",
                    "Speed",
                    "RPM",
                    "Throttle",
                    "Brake",
                    "nGear",
                    "DRS",
                    "Distance",
                ]

                available_tel_cols = [
                    col for col in telemetry_cols if col in car_data.columns
                ]
                telemetry_clean = car_data[available_tel_cols].copy()

                # Converter Time para segundos (tratamento robusto)
                if "Time" in telemetry_clean.columns and len(telemetry_clean) > 0:
                    try:
                        telemetry_clean["Time_seconds"] = telemetry_clean[
                            "Time"
                        ].dt.total_seconds()
                    except AttributeError:
                        if hasattr(telemetry_clean["Time"].iloc[0], "total_seconds"):
                            telemetry_clean["Time_seconds"] = telemetry_clean[
                                "Time"
                            ].apply(lambda x: x.total_seconds())
                        else:
                            pass

                # Usar abreviação como chave do dicionário
                telemetry_dict[driver_abbr] = telemetry_clean
                print(f"    ✓ {len(telemetry_clean)} pontos de telemetria")

            except Exception as e:
                print(f"    ✗ Erro ao processar {driver_abbr}: {e}")
                continue

        print(f"  ✓ Telemetria extraída de {len(telemetry_dict)} pilotos")
        return telemetry_dict

    def extract_race_control_messages(self) -> pd.DataFrame:
        """
        C. Extrai mensagens de controle de corrida.

        Inclui: Bandeiras Amarelas, SC, VSC, Penalidades
        """
        print("\nExtraindo mensagens de controle de corrida...")
        messages = self.session.race_control_messages.copy()

        # Converter Time para segundos (tratamento robusto para diferentes tipos)
        if "Time" in messages.columns and len(messages) > 0:
            try:
                # Tentar conversão direta se for Timedelta
                messages["Time_seconds"] = messages["Time"].dt.total_seconds()
            except AttributeError:
                # Se não for Timedelta, pode ser Timestamp ou outro tipo
                # Converter para numérico (segundos desde epoch ou início)
                if hasattr(messages["Time"].iloc[0], "total_seconds"):
                    messages["Time_seconds"] = messages["Time"].apply(
                        lambda x: x.total_seconds()
                    )
                else:
                    # Se for timestamp, usar como está (será convertido depois se necessário)
                    pass

        print(f"  ✓ {len(messages)} mensagens extraídas")
        return messages

    def extract_weather_data(self) -> pd.DataFrame:
        """
        C. Extrai dados meteorológicos.

        Parâmetros: AirTemp, TrackTemp, Rainfall, WindSpeed
        """
        print("\nExtraindo dados meteorológicos...")
        weather = self.session.weather_data.copy()

        # Converter Time para segundos (tratamento robusto para diferentes tipos)
        if "Time" in weather.columns and len(weather) > 0:
            try:
                # Tentar conversão direta se for Timedelta
                weather["Time_seconds"] = weather["Time"].dt.total_seconds()
            except AttributeError:
                # Se não for Timedelta, pode ser Timestamp ou outro tipo
                if hasattr(weather["Time"].iloc[0], "total_seconds"):
                    weather["Time_seconds"] = weather["Time"].apply(
                        lambda x: x.total_seconds()
                    )
                else:
                    # Se for timestamp, usar como está
                    pass

        # Colunas meteorológicas críticas
        weather_cols = [
            "Time",
            "Time_seconds",
            "AirTemp",
            "TrackTemp",
            "Rainfall",
            "WindSpeed",
            "Humidity",
            "Pressure",
        ]

        available_weather_cols = [col for col in weather_cols if col in weather.columns]
        weather_clean = weather[available_weather_cols].copy()

        print(f"  ✓ {len(weather_clean)} registros meteorológicos")
        return weather_clean

    def extract_results(self) -> pd.DataFrame:
        """
        D. Extrai resultados finais da corrida.

        Parâmetros: Position, Points, GridPosition, Status
        """
        print("\nExtraindo resultados finais...")
        results = self.session.results.copy()

        # Colunas críticas de resultado
        result_cols = [
            "DriverNumber",
            "Abbreviation",
            "FullName",
            "TeamName",
            "Position",
            "GridPosition",
            "Points",
            "Status",
            "ClassifiedPosition",
        ]

        available_result_cols = [col for col in result_cols if col in results.columns]
        results_clean = results[available_result_cols].copy()

        print(f"  ✓ {len(results_clean)} resultados extraídos")
        return results_clean

    def extract_all(self) -> Dict:
        """
        Extrai TODOS os dados de uma vez.

        SEMPRE extrai dados completos incluindo telemetria de todos os pilotos.

        Returns:
            Dicionário com todos os dados extraídos:
            - event_info: Metadados do evento
            - laps: Dados de voltas e estratégia
            - telemetry: Telemetria completa de todos os pilotos
            - race_control: Mensagens de controle de corrida
            - weather: Dados meteorológicos
            - results: Resultados finais
        """
        print(f"\n{'=' * 60}")
        print(f"ETL COMPLETO: {self.event_name} ({self.year})")
        print(f"{'=' * 60}")

        data = {
            "event_info": {
                "event_name": self.event_name,
                "year": self.year,
                "round_number": self.round_number,
                "location": self.session.event["Location"],
                "country": self.session.event["Country"],
            },
            "laps": self.extract_laps_data(),
            "telemetry": self.extract_telemetry_data(),
            "race_control": self.extract_race_control_messages(),
            "weather": self.extract_weather_data(),
            "results": self.extract_results(),
        }

        return data

    def save_to_parquet(self, data: Dict, output_dir: str = "data/raw/races"):
        """
        Salva todos os dados em formato Parquet.

        Args:
            data: Dicionário com dados extraídos
            output_dir: Diretório de saída
        """
        print(f"\n{'=' * 60}")
        print("SALVANDO DADOS")
        print(f"{'=' * 60}")

        # Criar estrutura de diretórios
        race_dir = Path(output_dir) / f"{self.year}" / f"round_{self.round_number:02d}"
        race_dir.mkdir(parents=True, exist_ok=True)

        # Salvar cada tipo de dado
        print(f"\nDiretório: {race_dir}")

        # Laps
        laps_file = race_dir / "laps.parquet"
        data["laps"].to_parquet(laps_file, index=False)
        print(f"  ✓ Laps: {laps_file}")

        # Race Control
        rc_file = race_dir / "race_control.parquet"
        data["race_control"].to_parquet(rc_file, index=False)
        print(f"  ✓ Race Control: {rc_file}")

        # Weather
        weather_file = race_dir / "weather.parquet"
        data["weather"].to_parquet(weather_file, index=False)
        print(f"  ✓ Weather: {weather_file}")

        # Results
        results_file = race_dir / "results.parquet"
        data["results"].to_parquet(results_file, index=False)
        print(f"  ✓ Results: {results_file}")

        # Telemetry (se disponível)
        if "telemetry" in data:
            telemetry_dir = race_dir / "telemetry"
            telemetry_dir.mkdir(exist_ok=True)

            for driver, tel_data in data["telemetry"].items():
                tel_file = telemetry_dir / f"{driver}.parquet"
                tel_data.to_parquet(tel_file, index=False)
                print(f"  ✓ Telemetria {driver}: {tel_file}")

        # Salvar metadados
        import json

        metadata_file = race_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(data["event_info"], f, indent=2, default=str)
        print(f"  ✓ Metadata: {metadata_file}")

        print(f"\n{'=' * 60}")
        print("✓ TODOS OS DADOS SALVOS COM SUCESSO")
        print(f"{'=' * 60}\n")

        return race_dir
