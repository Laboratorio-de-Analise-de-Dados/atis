"""Entidades biológicas do modelo imune."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import ClassVar


class NodeType(Enum):
    PAMP = "pamp"
    PATHOGEN = "pathogen"
    MACROPHAGE = "macrophage"
    PRO_CYTOKINE = "pro_cytokine"
    ANTI_CYTOKINE = "anti_cytokine"
    CHEMOKINE = "chemokine"
    REPAIR_FACTOR = "repair_factor"
    DAMAGE = "damage"


class EdgeType(Enum):
    RECOGNITION = "recognition"
    ACTIVATION = "activation"
    PRODUCTION = "production"
    RECRUITMENT = "recruitment"
    INTERACTION = "interaction"
    REGULATION = "regulation"
    REPAIR = "repair"


class Action(IntEnum):
    NOOP = 0
    SECRETE_PRO = 1
    SECRETE_ANTI = 2
    SECRETE_CHEMO = 3
    SECRETE_REPAIR = 4

    # Aliases mantidos para o piloto e para os testes antigos.
    ACTIVATE = SECRETE_PRO
    REGULATE = SECRETE_ANTI


class Phenotype(Enum):
    M0 = "M0"
    M1 = "M1"
    M2 = "M2"


ACTION_LABELS = {
    Action.NOOP: "não fazer nada",
    Action.SECRETE_PRO: "secretar citocina pró-inflamatória",
    Action.SECRETE_ANTI: "secretar citocina anti-inflamatória",
    Action.SECRETE_CHEMO: "secretar quimiocina",
    Action.SECRETE_REPAIR: "secretar fator de reparo",
}


@dataclass
class Entity(ABC):
    """Qualquer coisa que pode representar um nó."""

    id: str
    node_type: ClassVar[NodeType]

    @abstractmethod
    def signal(self) -> float:
        """Número que os vizinhos enxergam dessa entidade."""

    @abstractmethod
    def state(self) -> dict:
        """Estado interno completo para logs e análises."""


@dataclass
class Substance(Entity):
    """Entidades passivas, definidas apenas por uma quantidade."""

    level: float = 0.0

    def signal(self) -> float:
        return self.level

    def state(self) -> dict:
        return {"level": self.level}


class PAMP(Substance):
    node_type = NodeType.PAMP


class Pathogen(Substance):
    node_type = NodeType.PATHOGEN


class Cytokine(Substance):
    """Base para mediadores solúveis (compatível com o piloto)."""

    node_type = NodeType.PRO_CYTOKINE


class ProCytokine(Cytokine):
    node_type = NodeType.PRO_CYTOKINE


class AntiCytokine(Cytokine):
    node_type = NodeType.ANTI_CYTOKINE


class Chemokine(Substance):
    node_type = NodeType.CHEMOKINE


class RepairFactor(Substance):
    node_type = NodeType.REPAIR_FACTOR


class TissueDamage(Substance):
    node_type = NodeType.DAMAGE


@dataclass
class Cell(Entity):
    """Entidades ativas: têm estado interno e podem agir."""

    activation: float = 0.2
    memory: float = 0.0
    phenotype: Phenotype = Phenotype.M0
    pro_score: float = 0.0
    anti_score: float = 0.0
    last_action: Action = Action.NOOP
    phenotype_threshold: float = 0.2
    action_activation: float = 0.2

    def signal(self) -> float:
        return self.activation

    def state(self) -> dict:
        return {
            "activation": self.activation,
            "memory": self.memory,
            "phenotype": self.phenotype.value,
            "pro_score": self.pro_score,
            "anti_score": self.anti_score,
        }

    def apply_action(self, action: Action) -> None:
        """A célula muda a si mesma; a dinâmica muda o ambiente."""
        action = Action(action)
        self.last_action = action

        if action == Action.SECRETE_PRO:
            self.pro_score = 0.7 * self.pro_score + 0.3
            self.anti_score *= 0.9
            self.activation = min(1.0, self.activation + self.action_activation)
        elif action == Action.SECRETE_ANTI:
            self.anti_score = 0.7 * self.anti_score + 0.3
            self.pro_score *= 0.9
            self.activation = max(0.0, self.activation - self.action_activation)
        elif action == Action.SECRETE_CHEMO:
            self.pro_score *= 0.95
            self.anti_score *= 0.95
            self.activation = min(1.0, self.activation + 0.25 * self.action_activation)
        elif action == Action.SECRETE_REPAIR:
            self.anti_score = 0.9 * self.anti_score + 0.1
            self.pro_score *= 0.95
            self.activation = max(0.0, self.activation - 0.25 * self.action_activation)

        self._update_phenotype()
        self.memory = 0.9 * self.memory + 0.1 * self.activation

    def _update_phenotype(self) -> None:
        diff = self.pro_score - self.anti_score
        if diff > self.phenotype_threshold:
            self.phenotype = Phenotype.M1
        elif diff < -self.phenotype_threshold:
            self.phenotype = Phenotype.M2
        else:
            self.phenotype = Phenotype.M0


class Macrophage(Cell):
    node_type = NodeType.MACROPHAGE


@dataclass
class NeighborView:
    """O que a célula enxerga de um vizinho."""

    node_type: NodeType
    signal: float
    edge_types: tuple[EdgeType, ...] = ()
    strength: float = 0.0


@dataclass
class Observation:
    """O que uma unidade de célula enxerga: o_i = f(x_i, N_i)."""

    own: dict
    neighbors: dict
