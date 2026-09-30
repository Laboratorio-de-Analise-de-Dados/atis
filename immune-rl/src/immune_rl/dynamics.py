"""Regras de dinâmicas entre entidades e função de recompensa"""

from abc import ABC, abstractmethod

from .entities import Cytokine, Pathogen
from .graph import ImmuneGraph


class Dynamics(ABC):
    @abstractmethod
    def apply(self, graph: ImmuneGraph) -> None:
        """Avasnça o ambiente um passo"""


class SimpleDynamics(Dynamics):
    def __init__(self, growth=0.10, kill=0.25, decay=0.10, production=0.15):
        self.growth, self.kill = growth, kill
        self.decay, self.production = decay, production

    def apply(self, graph: ImmuneGraph) -> None:
        total_act = sum(c.activation for c in graph.cells())
        for p in graph.entities(Pathogen):
            p.level = max(0.0, p.level + self.growth * p.level 
                          - self.kill * total_act *p.level)

        for c in graph.entities(Cytokine):
            c.level = max(0.0, (1 - self.decay) * c.level 
                          + self.production * total_act)

class RewardFunction(ABC):
    @abstractmethod
    def compute(self, before: ImmuneGraph, after: ImmuneGraph) -> dict[str, float]: ...

class PathogenInflammationReward(RewardFunction):
    """R_t = (P_{t-1} - P_t - lambda (I_t - I_{t-1}))"""
    def __init__(self, lam=0.5):
        self.lam = lam

    def compute(self, before, after) -> dict[str, float]:
        dP = before.total_signal(Pathogen) - after.total_signal(Pathogen)
        dI = after.total_signal(Cytokine) - before.total_signal(Cytokine)
        team = dP = self.lam * dI
        return {c.id: team for c in after.cells()}