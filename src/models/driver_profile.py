"""
Modelos Pydantic para o perfil de desempenho de cada piloto em uma corrida.

Calculados deterministicamente a partir de laps_clustered, anomalies_summary,
tire_cliffs_summary e results_processed.

Output: driver_profiles.json
"""

from pydantic import BaseModel, ConfigDict, Field


class CompoundUsage(BaseModel):
    """Uso de composto de pneu por número de voltas."""

    model_config = ConfigDict(extra="forbid")

    compound: str = Field(..., description="Composto: 'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'.")
    laps: int = Field(..., gt=0, description="Número de voltas rodadas neste composto.")


class DriverProfile(BaseModel):
    """
    Perfil de desempenho de um piloto em uma corrida específica.

    Todos os campos são calculados deterministicamente — sem inferência da LLM.
    Permite que DSPY/Agno discutam estratégia e desempenho por piloto.
    """

    model_config = ConfigDict(extra="forbid")

    driver: str = Field(..., min_length=3, max_length=3, description="Sigla do piloto.")
    team: str = Field(..., description="Nome da equipe.")
    final_position: int | None = Field(None, description="Posição final (None se DNF).")
    grid_position: int | None = Field(None, description="Posição de largada.")
    positions_gained: int = Field(
        ...,
        description="Posições ganhas vs largada (positivo=ganhou, negativo=perdeu).",
    )
    points: float = Field(..., ge=0, description="Pontos marcados na corrida.")
    push_pct: float = Field(..., ge=0.0, le=1.0, description="Fração de voltas no cluster 'push'.")
    base_pct: float = Field(..., ge=0.0, le=1.0, description="Fração de voltas no cluster 'base'.")
    degraded_pct: float = Field(..., ge=0.0, le=1.0, description="Fração de voltas no cluster 'degraded'.")
    compounds_used: list[CompoundUsage] = Field(..., min_length=1)
    had_tire_cliff: bool = Field(..., description="True se o piloto teve ao menos um tire cliff.")
    cliff_count: int = Field(..., ge=0, description="Número de tire cliffs durante a corrida.")
    anomaly_count: int = Field(..., ge=0, description="Número de voltas anômalas detectadas.")
