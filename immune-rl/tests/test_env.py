import pytest
from immune_rl.dynamics import PathogenInflammationReward, SimpleDynamics
from immune_rl.entities import Action, Cytokine, Pathogen
from immune_rl.environment import ImmuneEnvironment
from immune_rl.graph import build_pilot_graph


def make_env(max_steps=30):
    return ImmuneEnvironment(build_pilot_graph, SimpleDynamics(),
                             PathogenInflammationReward(), max_steps=max_steps)
 
 
def all_do(env, action):
    return {c.id: action for c in env.graph.cells()}
 
 
def test_reset_creates_expected_world():
    env = make_env()
    assert {c.id for c in env.graph.cells()} == {"mac_A", "mac_B"}
    assert env.graph.total_signal(Pathogen) == 1.0
 
 
def test_observation_is_local():
    obs = make_env().observe()["mac_A"]
    assert set(obs.neighbors) == {"pamp", "pathogen", "cytokine"}   # não vê mac_B
 
 
def test_activate_raises_activation():
    env = make_env()
    env.step(all_do(env, Action.ACTIVATE))
    assert env.graph.cell("mac_A").activation == pytest.approx(0.4)
 
 
def test_activation_stays_between_0_and_1():
    env = make_env(max_steps=100)
    for _ in range(20):
        env.step(all_do(env, Action.ACTIVATE))
    assert env.graph.cell("mac_A").activation <= 1.0
    for _ in range(20):
        env.step(all_do(env, Action.REGULATE))
    assert env.graph.cell("mac_A").activation >= 0.0
 
 
def test_inactive_cells_let_pathogen_grow():
    env = make_env()
    env.step(all_do(env, Action.REGULATE))
    assert env.graph.total_signal(Pathogen) > 1.0
 
 
def test_active_cells_reduce_pathogen_but_cause_inflammation():
    env = make_env()
    for _ in range(5):
        env.step(all_do(env, Action.ACTIVATE))
    assert env.graph.total_signal(Pathogen) < 1.0
    assert env.graph.total_signal(Cytokine) > 0.0
 
 
def test_first_activation_gives_positive_team_reward():
    env = make_env()
    _, rewards, *_ = env.step(all_do(env, Action.ACTIVATE))
    assert rewards["mac_A"] == pytest.approx(rewards["mac_B"])   # recompensa de equipe
    assert rewards["mac_A"] > 0
 
 
def test_truncated_at_max_steps():
    env = make_env(max_steps=5)
    terminated = truncated = False
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(all_do(env, Action.NOOP))
    assert truncated and not terminated
 
 
def test_terminated_when_pathogen_eliminated():
    env = make_env(max_steps=100)
    terminated = False
    while not terminated and env.t < 100:
        _, _, terminated, _, _ = env.step(all_do(env, Action.ACTIVATE))
    assert terminated
 
 
def test_history_keeps_independent_snapshots():
    env = make_env()
    for _ in range(3):
        env.step(all_do(env, Action.ACTIVATE))
    assert len(env.history) == 4                                   # G_0 ... G_3
    assert env.history[0].total_signal(Pathogen) == 1.0           # G_0 não foi alterado depois
    assert env.history[3].total_signal(Pathogen) != 1.0