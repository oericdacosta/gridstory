"""
Modelos Pydantic para eventos de corrida — Muralha de Fogo entre ML e LLM.

Valida os outputs do pipeline de Machine Learning antes de qualquer contato
com DSPY/Agno. Qualquer dado inválido levanta ValidationError aqui.

Fluxo:
    Isolation Forest / Ruptures / strategy.py
        → dicionários brutos
            → RaceTimeline.model_validate()  ← Pydantic barra dado ruim aqui
                → timeline.json
                    → DSPY / Agno (LLM nunca vê lixo)

Formato semântico: a LLM recebe apenas campos narrativos (winner/loser, type,
positions_lost) — sem métricas internas de ML (anomaly_score, z_score).
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, RootModel, model_validator


class BaseRaceEvent(BaseModel):
    """Campos comuns a todos os eventos de corrida."""

    model_config = ConfigDict(extra="forbid", strict=True)

    lap: int = Field(..., gt=0, description="Volta em que o evento ocorreu.")


class DriverErrorEvent(BaseRaceEvent):
    """
    Anomalia de ritmo atribuída ao piloto ou mecânica.

    O pipeline ML (Isolation Forest + Z-score) identificou uma queda de ritmo
    sem causa externa (SC/VSC/bandeira) identificada.
    """

    type: Literal["driver_error"] = "driver_error"
    driver: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Sigla do piloto (ex: VER, HAM, NOR).",
    )


class ExternalIncidentEvent(BaseRaceEvent):
    """
    Anomalia de ritmo causada por evento externo (Safety Car, VSC, bandeira).

    O pipeline ML cruzou a volta anômala com race_control e identificou
    um evento de corrida externo como causa.
    """

    type: Literal["external_incident"] = "external_incident"
    driver: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Sigla do piloto (ex: VER, HAM, NOR).",
    )


class TireDropoffEvent(BaseRaceEvent):
    """
    Queda de ritmo por degradação de pneu (tire cliff), detectada pelo Ruptures/PELT.

    Indica que o pneu do piloto atingiu o limite de desempenho e a degradação
    acelerou significativamente.
    """

    type: Literal["tire_dropoff"] = "tire_dropoff"
    driver: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Sigla do piloto (ex: VER, HAM, NOR).",
    )
    lap_time_drop_seconds: float = Field(
        ...,
        gt=0.0,
        description="Magnitude da queda de ritmo em segundos.",
    )
    cliff_validated: bool = Field(
        ...,
        description="True se o cliff foi validado por slope positivo nas voltas anteriores.",
    )
    positions_lost: int | None = Field(
        None,
        description="Posições perdidas durante a queda de ritmo (None se não calculável).",
    )


class UndercutEvent(BaseRaceEvent):
    """
    Manobra de undercut detectada por strategy.detect_undercuts().

    Ocorre quando um piloto para antes do rival, volta com pneus novos
    e sai à frente na pista.
    """

    type: Literal["undercut"] = "undercut"
    winner: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Piloto que executou o undercut (saiu na frente).",
    )
    loser: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Piloto que sofreu o undercut (perdeu a posição).",
    )
    time_gained_seconds: float = Field(
        ...,
        description="Ganho de tempo estimado da manobra em segundos.",
    )


class OvercutEvent(BaseRaceEvent):
    """
    Manobra de overcut detectada por strategy.detect_overcuts() (ML-06).

    Ocorre quando um piloto (A) permanece em pista enquanto o rival (B) à sua
    frente entra nos boxes. A para mais tarde e ainda mantém ou ganha a posição
    sobre B graças ao ritmo no stint longo.
    """

    type: Literal["overcut"] = "overcut"
    winner: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Piloto que executou o overcut (manteve/ganhou posição).",
    )
    loser: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Piloto que sofreu o overcut (perdeu a posição ao parar antes).",
    )
    time_saved_seconds: float = Field(
        ...,
        description="Vantagem mantida estimada em segundos.",
    )


class SafetyCarEvent(BaseRaceEvent):
    """
    Safety Car implantado durante a corrida.

    Lido de race_control_processed.parquet onde is_safety_car=True
    e Message contém 'SAFETY CAR DEPLOYED'.
    """

    type: Literal["safety_car"] = "safety_car"
    deployed_on_lap: int = Field(
        ...,
        gt=0,
        description="Volta em que o Safety Car foi implantado.",
    )
    duration_laps: int = Field(
        ...,
        ge=1,
        description="Número de voltas que o Safety Car permaneceu em pista.",
    )
    cause: str | None = Field(
        None,
        description="Causa do safety car extraída do race_control (ex: 'CAR 14 (ALO) RETIRED').",
    )


class PenaltyEvent(BaseRaceEvent):
    """
    Penalidade aplicada pela FIA a um piloto.

    Lido de race_control_processed.parquet onde is_penalty=True.
    """

    type: Literal["penalty"] = "penalty"
    driver: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Sigla do piloto penalizado.",
    )
    reason: str = Field(
        ...,
        description="Motivo da penalidade (ex: 'Unsafe Release').",
    )


class RetirementEvent(BaseRaceEvent):
    """
    Abandono (DNF) de um piloto.

    Adicionado deterministicamente para cada piloto nos DNFs do race_summary
    que não possui outro evento associado, garantindo que todo abandono seja
    narrado pela LLM.
    """

    type: Literal["retirement"] = "retirement"
    driver: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Sigla do piloto que abandonou.",
    )
    cause: str | None = Field(
        None,
        description="Causa do abandono extraída do race_control, se disponível.",
    )


# Tipo discriminado: o Pydantic instancia a classe correta baseado em 'type'
AnyRaceEvent = Annotated[
    Union[
        DriverErrorEvent,
        ExternalIncidentEvent,
        TireDropoffEvent,
        UndercutEvent,
        OvercutEvent,
        SafetyCarEvent,
        PenaltyEvent,
        RetirementEvent,
    ],
    Discriminator("type"),
]


class RaceTimeline(RootModel[list[AnyRaceEvent]]):
    """
    Timeline cronológica de eventos de uma corrida, validada pelo Pydantic.

    Uso:
        timeline = RaceTimeline.model_validate(raw_events_list)
        json_str = timeline.to_json()  # pronto para DSPY/Agno
    """

    @model_validator(mode="after")
    def sort_events_by_lap(self) -> "RaceTimeline":
        """Ordena cronologicamente após validação — garante ordem para a LLM."""
        self.root.sort(key=lambda event: event.lap)
        return self

    def to_json(self) -> str:
        """Serializa para JSON sem campos nulos — economiza tokens da LLM."""
        return self.model_dump_json(indent=2, exclude_none=True)
