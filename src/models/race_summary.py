"""
Modelos Pydantic para o resumo geral de uma corrida de F1.

Contém contexto estrutural calculado deterministicamente a partir dos parquets
processados (results, weather, laps, race_control) — sem LLM.

Output: race_summary.json
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WeatherSummary(BaseModel):
    """Resumo das condições meteorológicas da corrida."""

    model_config = ConfigDict(extra="forbid")

    condition: Literal["dry", "wet", "mixed"] = Field(
        ...,
        description="Condição geral: dry (sem chuva), wet (chuva), mixed (misto).",
    )
    air_temp_avg_c: float = Field(..., description="Temperatura média do ar em °C.")
    track_temp_avg_c: float = Field(..., description="Temperatura média da pista em °C.")
    had_rainfall: bool = Field(..., description="True se houve precipitação em algum momento.")


class PodiumEntry(BaseModel):
    """Entrada do pódio."""

    model_config = ConfigDict(extra="forbid")

    position: int = Field(..., ge=1, le=3, description="Posição (1, 2 ou 3).")
    driver: str = Field(..., min_length=3, max_length=3, description="Sigla do piloto.")
    team: str = Field(..., description="Nome da equipe.")
    gap_to_leader: str = Field(
        ...,
        description="Gap para o líder: '0.000' para vencedor, '+5.234s' ou '+1 lap'.",
    )


class DnfEntry(BaseModel):
    """Registro de abandono (DNF)."""

    model_config = ConfigDict(extra="forbid")

    driver: str = Field(..., min_length=3, max_length=3, description="Sigla do piloto.")
    on_lap: int | None = Field(None, description="Volta em que o piloto abandonou.")


class RaceSummary(BaseModel):
    """
    Resumo estrutural de uma corrida de F1.

    Todos os campos são calculados deterministicamente — sem inferência da LLM.
    Fornece contexto factual para DSPY/Agno gerarem narrativa.
    """

    model_config = ConfigDict(extra="forbid")

    year: int = Field(..., description="Ano da temporada.")
    round: int = Field(..., description="Número da rodada.")
    total_laps: int = Field(..., gt=0, description="Total de voltas da corrida.")
    winner: str = Field(..., min_length=3, max_length=3, description="Sigla do vencedor.")
    fastest_lap_driver: str = Field(
        ..., min_length=3, max_length=3, description="Piloto com a volta mais rápida."
    )
    fastest_lap_time: float = Field(..., gt=0, description="Tempo da volta mais rápida em segundos.")
    podium: list[PodiumEntry] = Field(..., min_length=1, max_length=3)
    dnfs: list[DnfEntry] = Field(default_factory=list, description="Pilotos que abandonaram.")
    safety_car_count: int = Field(..., ge=0, description="Número de Safety Cars durante a corrida.")
    weather: WeatherSummary
