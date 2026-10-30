"""Eventos legíveis que explicam o que aconteceu em cada passo."""

from dataclasses import dataclass, field
from enum import Enum


class EventKind(Enum):
    ACTION = "action"
    PAMP_ARRIVAL = "pamp_arrival"
    PATHOGEN_ARRIVAL = "pathogen_arrival"
    RECOGNITION = "recognition"
    ACTIVATION = "activation"
    PHENOTYPE = "phenotype"
    SECRETION = "secretion"
    RECRUITMENT = "recruitment"
    INTERACTION = "interaction"
    KILLING = "killing"
    DAMAGE = "damage"
    REPAIR = "repair"
    REGULATION = "regulation"
    FADE = "fade"
    PATHOGEN_ERADICATED = "pathogen_eradicated"
    EPISODE_END = "episode_end"


class EventSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Event:
    """Um fato ocorrido no passo ``t`` do episódio."""

    t: int
    kind: EventKind
    message: str
    nodes: tuple[str, ...] = field(default_factory=tuple)
    edges: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    severity: EventSeverity = EventSeverity.INFO
