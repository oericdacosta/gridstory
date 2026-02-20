"""
Modelos Pydantic para validação de dados — Muralha de Fogo entre ML e LLM.
"""

from .race_events import (
    BaseRaceEvent,
    DriverErrorEvent,
    ExternalIncidentEvent,
    TireDropoffEvent,
    UndercutEvent,
    SafetyCarEvent,
    PenaltyEvent,
    AnyRaceEvent,
    RaceTimeline,
)
from .race_summary import (
    WeatherSummary,
    PodiumEntry,
    DnfEntry,
    RaceSummary,
)
from .driver_profile import (
    CompoundUsage,
    DriverProfile,
)

__all__ = [
    # Race Events (timeline.json)
    "BaseRaceEvent",
    "DriverErrorEvent",
    "ExternalIncidentEvent",
    "TireDropoffEvent",
    "UndercutEvent",
    "SafetyCarEvent",
    "PenaltyEvent",
    "AnyRaceEvent",
    "RaceTimeline",
    # Race Summary (race_summary.json)
    "WeatherSummary",
    "PodiumEntry",
    "DnfEntry",
    "RaceSummary",
    # Driver Profiles (driver_profiles.json)
    "CompoundUsage",
    "DriverProfile",
]
