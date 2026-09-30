from collections.abc import Callable

from .dynamics import Dynamics, RewardFunction
from .entities import Action, Cytokine, Observation, Pathogen
from .graph import ImmuneGraph


class ImmuneEnvironment:
    def __init__(self, graph_factory: Callable[[], ImmuneGraph],
        dynamics: Dynamics, reward_fn: RewardFunction, max_steps: int = 30):
        self.graph_factory = graph_factory
        self.dynamics = dynamics
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.graph: ImmuneGraph
        self.history: list[ImmuneGraph] = []
        self.t = 0
        self.reset()

    def reset(self) -> dict[str, Observation]:
        self.graph = self.graph_factory()
        self.history = [self.graph.snapshot()]
        self.t = 0
        return self.observe()

    def observe(self) -> dict[str, Observation]:
        return {c.id: self.graph.observe(c.id) for c in self.graph.cells()}

    def step(self, actions: dict[str, Action]):
        before = self.graph.snapshot()
        for cell_id, action in actions.items():
            self.graph.cell(cell_id).apply_action(action)
        self.dynamics.apply(self.graph)
        rewards = self.reward_fn.compute(before, self.graph)
        self.history.append(self.graph.snapshot()) # G_{t+1}
        self.t += 1

        P = self.graph.total_signal(Pathogen)
        I = self.graph.total_signal(Cytokine)
        terminated = P < 0.01 # patógeno eliminado
        truncated = self.t >= self.max_steps # fim do tempo
        return self.observe(), rewards, terminated, truncated, {
            "P": P, 
            "I": I, 
            "t": self.t
            }