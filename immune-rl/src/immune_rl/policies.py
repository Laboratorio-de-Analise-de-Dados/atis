"""Define as Políticas do agente RL"""

import random
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from .encoding import StateEncoder
from .entities import Action, Observation


class Policy(ABC):
    @abstractmethod
    def select_action(self, obs: Observation) -> Action: ...

    def update(self, obs, action, reward, next_obs, terminated) -> None:
        """Política aprendida subscreve"""

    def end_episode(self) -> None:
        pass

class RandomPolicy(Policy):
    def select_action(self, obs: Observation) -> Action:
        return random.choice(list(Action))

class FixedPolicy(Policy):
    def __init__(self, action: Action):
        self.action = action

    def select_action(self, obs: Observation) -> Action:
        return self.action 

class QLearningPolicy(Policy):
    """
    Q-learning tabular adaptado para RL multi agente. Política compartilhada, vários 
    macrófagos usam a mesma instância
    """

    def __init__(
            self, encoder: StateEncoder, alpha=0.1, gamma=0.95, epsilon=1.0, 
            epsilon_min=0.05, epsilon_decay=0.995, seed=None):
        self.encoder = encoder
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.q  = defaultdict(lambda: np.zeros(len(Action))) # Q[estado]
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: Observation) -> Action:
        s = self.encoder.encode(obs)
        if self.rng.random() < self.epsilon: # Exploration
            return Action(int(self.rng.integers(len(Action))))
        q = self.q[s] # Exploration vs Exploitation
        best = np.flatnonzero(q == q.max()) # Desempate aleatório
        return Action(int(self.rng.choice(best)))

    def update(self, obs, action, reward, next_obs, terminated) -> None:
        s, s2 = self.encoder.encode(obs), self.encoder.encode(next_obs)
        target = reward if terminated else reward + self.gamma * self.q[s2].max()
        self.q[s][action] += self.alpha * (target - self.q[s][action])

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)