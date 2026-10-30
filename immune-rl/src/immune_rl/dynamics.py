"""Regras de dinâmica entre entidades e funções de recompensa."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .entities import Pathogen, ProCytokine, TissueDamage
from .events import Event
from .graph import ImmuneGraph


class Dynamics(ABC):
    @abstractmethod
    def apply(self, graph: ImmuneGraph) -> list[Event]:
        """Avança o ambiente um passo e devolve os eventos ocorridos."""


class SimpleDynamics(Dynamics):
    """Dinâmica reduzida do piloto, mantida para os testes unitários."""

    def __init__(self, growth=0.10, kill=0.25, decay=0.10, production=0.15):
        self.growth, self.kill = growth, kill
        self.decay, self.production = decay, production

    def apply(self, graph: ImmuneGraph) -> list[Event]:
        total_activation = sum(cell.activation for cell in graph.cells())
        for pathogen in graph.entities(Pathogen):
            pathogen.level = max(
                0.0,
                pathogen.level
                + self.growth * pathogen.level
                - self.kill * total_activation * pathogen.level,
            )

        for cytokine in graph.entities(ProCytokine):
            cytokine.level = max(
                0.0,
                (1 - self.decay) * cytokine.level + self.production * total_activation,
            )
        return []


class RewardFunction(ABC):
    @abstractmethod
    def compute(self, before: ImmuneGraph, after: ImmuneGraph) -> dict[str, float]: ...


@dataclass
class RewardParams:
    lambda_inflammation: float = 0.5
    lambda_damage: float = 0.8
    lambda_repair: float = 0.6


class ImmuneHomeostasisReward(RewardFunction):
    """Recompensa de equipe: eliminar o patógeno sem inflamar nem danificar.

    R_t = (P_{t-1} - P_t) - lambda_I * Pro_t - lambda_D * D_t
          + lambda_H * max(0, D_{t-1} - D_t)
    """

    def __init__(self, params: RewardParams | None = None):
        self.params = params or RewardParams()

    def compute(self, before, after) -> dict[str, float]:
        delta_pathogen = before.total_signal(Pathogen) - after.total_signal(Pathogen)
        inflammation = after.total_signal(ProCytokine)
        damage = after.total_signal(TissueDamage)
        healed = max(0.0, before.total_signal(TissueDamage) - damage)

        team = (
            delta_pathogen
            - self.params.lambda_inflammation * inflammation
            - self.params.lambda_damage * damage
            + self.params.lambda_repair * healed
        )
        return {cell.id: team for cell in after.cells()}


class PathogenInflammationReward(ImmuneHomeostasisReward):
    """Recompensa antiga (apenas patógeno e inflamação), mantida por compatibilidade."""

    def __init__(self, lam: float = 0.5):
        super().__init__(
            RewardParams(
                lambda_inflammation=lam,
                lambda_damage=0.0,
                lambda_repair=0.0,
            )
        )
