import pytest
from immune_rl.encoding import StateEncoder
from immune_rl.entities import Action, NeighborView, NodeType, Observation
from immune_rl.policies import QLearningPolicy, RandomPolicy


def make_obs(activation=0.2, pathogen=1.0, cytokine=0.0):
    return Observation(
        own={"activation": activation, "memory": 0.0},
        neighbors={"p": NeighborView(NodeType.PATHOGEN, pathogen),
                   "c": NeighborView(NodeType.CYTOKINE, cytokine)},
    )
 
 
def test_random_policy_returns_a_single_action():
    # teria pego o bug random.choices (que devolve lista)
    for _ in range(20):
        assert isinstance(RandomPolicy().select_action(make_obs()), Action)
 
 
def test_encoder_bins_low_mid_high():
    enc = StateEncoder()
    assert enc.encode(make_obs(0.1, 0.1, 0.1)) == (0, 0, 0)
    assert enc.encode(make_obs(0.5, 0.5, 0.4)) == (1, 1, 1)
    assert enc.encode(make_obs(0.9, 0.9, 0.9)) == (2, 2, 2)
 
 
def test_q_update_terminal_ignores_future():
    pol = QLearningPolicy(StateEncoder(), alpha=0.5, gamma=0.9)
    obs, nxt = make_obs(), make_obs(0.6, 0.0, 0.0)
    pol.q[pol.encoder.encode(nxt)][:] = 10.0      
    pol.update(obs, Action.ACTIVATE, 1.0, nxt, terminated=True)
    assert pol.q[pol.encoder.encode(obs)][Action.ACTIVATE] == pytest.approx(0.5)
 
 
def test_q_update_bootstraps_when_not_terminal():
    pol = QLearningPolicy(StateEncoder(), alpha=0.5, gamma=0.9)
    obs, nxt = make_obs(), make_obs(0.6, 0.0, 0.0)
    pol.q[pol.encoder.encode(nxt)][:] = 10.0
    pol.update(obs, Action.ACTIVATE, 1.0, nxt, terminated=False)
    assert pol.q[pol.encoder.encode(obs)][Action.ACTIVATE] == pytest.approx(0.5 * (1.0 + 0.9 * 10.0))
 
 
def test_greedy_policy_picks_best_action():
    pol = QLearningPolicy(StateEncoder(), epsilon=0.0)
    obs = make_obs()
    pol.q[pol.encoder.encode(obs)][:] = [0.0, 5.0, 1.0]
    assert pol.select_action(obs) == Action.ACTIVATE
 
 
def test_epsilon_decays_but_not_below_minimum():
    pol = QLearningPolicy(StateEncoder(), epsilon=1.0, epsilon_min=0.5, epsilon_decay=0.1)
    pol.end_episode(); pol.end_episode()
    assert pol.epsilon == 0.5