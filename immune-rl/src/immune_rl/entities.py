"""Entidades biológicas"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import ClassVar

class NodeType(Enum):
    PAMP = "pamp"
    PATHOGEN = "pathogen"
    MACROPHAGE = "macrophage"
    CYTOKINE = "cytokine"

class EdgeType(Enum):
    RECOGNITION = "recognition"
    ACTIVATION = "activation"
    PRODUCTION = "production"

class Action(IntEnum):
    NOOP = 0
    ACTIVATE = 1
    REGULATE = 2


@dataclass
class Entity(ABC):
    """Qualquer coisa que pode representar um nó"""
    id:str
    node_type: ClassVar[NodeType]

    @abstractmethod
    def signal(self) -> float:
        """Número que os vizinhos enxergam dessa entidade"""

    @abstractmethod
    def state(self) -> dict:
        """Estado interno completo para logs e análises"""

@dataclass
class Substance(Entity):
    """Entidades passivas, definidas apenas por uma quantidade"""
    level: float = 0.0

    def signal(self) -> float:
        return self.level

    def state(self) -> dict:
        return {
            "level":self.level
        } 


class PAMP(Substance):
    node_type = NodeType.PAMP

class Pathogen(Substance):
    node_type = NodeType.PATHOGEN

class Cytokine(Substance):
    node_type = NodeType.CYTOKINE

@dataclass
class Cell(Entity):
    """Entidades ativas: têm estado interno e podem agir"""
    activation: float = 0.2
    memory: float = 0.0

    def signal(self) -> float:
        return self.activation

    def state(self) -> dict:
        return {
            "activation": self.activation,
            "memory": self.memory
        }

    def apply_action(self, action: Action) -> None:
        """A célula muda a si mesma; a dinâmica muda o ambiente"""
        if action == Action.ACTIVATE:
            self.activation = min(1.0, self.activation + 0.2)
        elif action == Action.REGULATE:
            self.activation = max(0.0, self.activation - 0.2)
        self.memory = 0.9 * self.memory + 0.1 * self.activation

class Macrophage(Cell):
    node_type = NodeType.MACROPHAGE

@dataclass
class Observation:
    """O que uma unidade de célula enxerga: o_i = f(x_i, N_i)"""
    own: dict
    neighbors: dict # id -> signal