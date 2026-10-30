from collections.abc import Callable

from .dynamics import Dynamics, RewardFunction
from .entities import (
    ACTION_LABELS,
    Action,
    AntiCytokine,
    Observation,
    Pathogen,
    ProCytokine,
    TissueDamage,
)
from .events import Event, EventKind
from .graph import ImmuneGraph


class ImmuneEnvironment:
    def __init__(
        self,
        graph_factory: Callable[[], ImmuneGraph],
        dynamics: Dynamics,
        reward_fn: RewardFunction,
        max_steps: int = 40,
    ):
        self.graph_factory = graph_factory
        self.dynamics = dynamics
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.graph: ImmuneGraph
        self.history: list[ImmuneGraph] = []
        self.events_history: list[list[Event]] = []
        self.t = 0
        self._pathogen_seen = False
        self.reset()

    def reset(self) -> dict[str, Observation]:
        self.graph = self.graph_factory()
        self.graph.time = 0
        self.history = [self.graph.snapshot()]
        self.events_history = [[]]
        self.t = 0
        self._pathogen_seen = bool(self.graph.entities(Pathogen))
        return self.observe()

    def observe(self) -> dict[str, Observation]:
        return {cell.id: self.graph.observe(cell.id) for cell in self.graph.cells()}

    def step(self, actions: dict[str, Action]):
        next_t = self.t + 1
        before = self.graph.snapshot()
        events: list[Event] = []

        for cell in self.graph.cells():
            action = Action(actions.get(cell.id, Action.NOOP))
            previous_phenotype = cell.phenotype
            cell.apply_action(action)

            if action != Action.NOOP:
                events.append(
                    Event(
                        next_t,
                        EventKind.ACTION,
                        f"{cell.id} escolheu {ACTION_LABELS[action]}",
                        (cell.id,),
                    )
                )
            if cell.phenotype != previous_phenotype:
                events.append(
                    Event(
                        next_t,
                        EventKind.PHENOTYPE,
                        f"{cell.id} mudou de {previous_phenotype.value} para {cell.phenotype.value}",
                        (cell.id,),
                    )
                )

        events.extend(self.dynamics.apply(self.graph))
        rewards = self.reward_fn.compute(before, self.graph)
        team_reward = next(iter(rewards.values()), 0.0)
        for cell in before.cells():
            rewards.setdefault(cell.id, team_reward)

        self.history.append(self.graph.snapshot())
        self.events_history.append(events)
        self.t = next_t

        pathogen = self.graph.total_signal(Pathogen)
        damage = self.graph.total_signal(TissueDamage)
        if self.graph.entities(Pathogen):
            self._pathogen_seen = True

        terminated = self._pathogen_seen and pathogen <= 0.01 and damage <= 0.05
        truncated = self.t >= self.max_steps
        if terminated:
            outcome = "patógeno eliminado e tecido recuperado"
        elif truncated:
            outcome = "tempo esgotado"
        else:
            outcome = "em andamento"

        info = {
            "P": pathogen,
            "Pro": self.graph.total_signal(ProCytokine),
            "Anti": self.graph.total_signal(AntiCytokine),
            "D": damage,
            "M1": sum(1 for cell in self.graph.cells() if cell.phenotype.value == "M1"),
            "M2": sum(1 for cell in self.graph.cells() if cell.phenotype.value == "M2"),
            "t": self.t,
            "events": events,
            "outcome": outcome,
        }
        return self.observe(), rewards, terminated, truncated, info
