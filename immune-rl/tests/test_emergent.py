import pytest

from immune_rl.dynamics import ImmuneHomeostasisReward, RewardParams
from immune_rl.emergent import (
    ID_ANTI,
    ID_CHEMO,
    ID_DAMAGE,
    ID_PRO,
    ID_REPAIR,
    EmergentDynamics,
    EmergentParams,
    build_seed_graph,
)
from immune_rl.entities import (
    PAMP,
    AntiCytokine,
    Chemokine,
    Macrophage,
    Pathogen,
    Phenotype,
    ProCytokine,
    RepairFactor,
    TissueDamage,
)
from immune_rl.environment import ImmuneEnvironment
from immune_rl.graph import ImmuneGraph


def make_graph(*entities):
    graph = ImmuneGraph()
    for entity in entities:
        graph.add_entity(entity)
    return graph


def test_seed_graph_has_macrophage_and_intact_tissue():
    graph = build_seed_graph()
    assert [cell.id for cell in graph.cells()] == ["mac_1"]
    assert graph.total_signal(TissueDamage) == 0.0
    assert not graph.entities(Pathogen)
    assert not graph.entities(PAMP)


def test_pamp_arrival_is_recognized():
    params = EmergentParams(pamp_arrival=1, p_recognize=1.0)
    env = ImmuneEnvironment(
        build_seed_graph,
        EmergentDynamics(params, seed=0),
        ImmuneHomeostasisReward(),
        max_steps=10,
    )
    env.reset()
    env.step({})

    assert env.graph.has_entity("pamp_1")
    assert env.graph.has_edge("pamp_1", "mac_1")
    assert env.graph.entities(PAMP)


def test_m1_kills_pathogen_but_m2_does_not():
    params = EmergentParams(
        pamp_arrival=99,
        pathogen_arrival=99,
        growth=0.0,
        kill=0.5,
        decay=0.0,
        signal_decay=0.0,
        natural_healing=0.0,
    )

    m1_graph = make_graph(
        Macrophage(
            "m1",
            activation=1.0,
            phenotype=Phenotype.M1,
            pro_score=1.0,
            action_activation=0.0,
        ),
        Pathogen("pathogen", level=1.0),
        TissueDamage(ID_DAMAGE, level=0.0),
    )
    EmergentDynamics(params, seed=0).apply(m1_graph)
    assert m1_graph.entity("pathogen").level == pytest.approx(0.5)

    m2_graph = make_graph(
        Macrophage(
            "m2",
            activation=1.0,
            phenotype=Phenotype.M2,
            anti_score=1.0,
            action_activation=0.0,
        ),
        Pathogen("pathogen", level=1.0),
        TissueDamage(ID_DAMAGE, level=0.0),
    )
    EmergentDynamics(params, seed=0).apply(m2_graph)
    assert m2_graph.entity("pathogen").level == pytest.approx(1.0)


def test_pro_feedback_only_activates_already_activated_cells():
    params = EmergentParams(
        pamp_arrival=99,
        pathogen_arrival=99,
        gain=0.0,
        decay=0.0,
        pro_feedback_thr=0.2,
        pro_feedback_gain=1.0,
        bind_thr=0.1,
        unbind_thr=0.0,
        signal_decay=0.0,
        natural_healing=0.0,
    )
    graph = make_graph(
        Macrophage("low", activation=0.05, action_activation=0.0),
        Macrophage("high", activation=0.5, action_activation=0.0),
        ProCytokine(ID_PRO, level=0.5),
    )

    EmergentDynamics(params, seed=0).apply(graph)

    assert not graph.has_edge(ID_PRO, "low")
    assert graph.has_edge(ID_PRO, "high")
    assert graph.entity("low").activation == pytest.approx(0.05)
    assert graph.entity("high").activation == pytest.approx(1.0)


def test_anti_inflammatory_cytokine_inhibits_activation():
    params = EmergentParams(
        pamp_arrival=99,
        pathogen_arrival=99,
        gain=0.0,
        decay=0.0,
        anti_inhibition=0.5,
        bind_thr=0.1,
        unbind_thr=0.0,
        signal_decay=0.0,
        natural_healing=0.0,
    )
    graph = make_graph(
        Macrophage("m", activation=0.8, action_activation=0.0),
        AntiCytokine(ID_ANTI, level=0.5),
    )

    EmergentDynamics(params, seed=0).apply(graph)

    assert graph.has_edge(ID_ANTI, "m")
    assert graph.entity("m").activation == pytest.approx(0.55)


def test_chemokine_recruits_new_macrophage():
    params = EmergentParams(
        pamp_arrival=99,
        pathogen_arrival=99,
        p_recruit=1.0,
        recruit_thr=0.1,
        max_cells=3,
        signal_decay=0.0,
    )
    graph = make_graph(
        Macrophage("mac_1", action_activation=0.0),
        Chemokine(ID_CHEMO, level=1.0),
        TissueDamage(ID_DAMAGE, level=0.0),
    )

    EmergentDynamics(params, seed=0).apply(graph)

    assert len(graph.cells()) == 2
    assert graph.has_entity("mac_2")
    assert graph.has_edge(ID_CHEMO, "mac_2")


def test_repair_factor_reduces_tissue_damage():
    params = EmergentParams(
        pamp_arrival=99,
        pathogen_arrival=99,
        signal_decay=0.0,
        natural_healing=0.0,
        damage_from_pathogen=0.0,
        damage_from_inflammation=0.0,
        repair_efficiency=0.5,
        bind_thr=0.1,
        unbind_thr=0.0,
    )
    graph = make_graph(
        TissueDamage(ID_DAMAGE, level=0.2),
        RepairFactor(ID_REPAIR, level=1.0),
    )

    EmergentDynamics(params, seed=0).apply(graph)

    assert graph.has_edge(ID_REPAIR, ID_DAMAGE)
    assert graph.entity(ID_DAMAGE).level == pytest.approx(0.0)


def test_pathogen_increases_tissue_damage():
    params = EmergentParams(
        pamp_arrival=99,
        pathogen_arrival=99,
        growth=0.0,
        signal_decay=0.0,
        natural_healing=0.0,
        damage_from_pathogen=0.2,
        damage_from_inflammation=0.0,
    )
    graph = make_graph(
        Pathogen("pathogen", level=0.5),
        TissueDamage(ID_DAMAGE, level=0.0),
    )

    EmergentDynamics(params, seed=0).apply(graph)

    assert graph.entity(ID_DAMAGE).level == pytest.approx(0.1)


def test_reward_formula_combines_clearance_inflammation_damage_and_repair():
    before = make_graph(
        Pathogen("pathogen", level=1.0),
        ProCytokine(ID_PRO, level=0.2),
        TissueDamage(ID_DAMAGE, level=0.4),
        Macrophage("m"),
    )
    after = make_graph(
        Pathogen("pathogen", level=0.5),
        ProCytokine(ID_PRO, level=0.1),
        TissueDamage(ID_DAMAGE, level=0.2),
        Macrophage("m"),
    )

    reward = ImmuneHomeostasisReward(
        RewardParams(lambda_inflammation=0.5, lambda_damage=0.8, lambda_repair=0.6)
    ).compute(before, after)

    assert reward["m"] == pytest.approx(0.41)


def test_events_are_aligned_with_history():
    params = EmergentParams(pamp_arrival=1, pathogen_arrival=2, p_recognize=1.0)
    env = ImmuneEnvironment(
        build_seed_graph,
        EmergentDynamics(params, seed=0),
        ImmuneHomeostasisReward(),
        max_steps=6,
    )
    env.reset()
    for _ in range(3):
        env.step({})

    assert len(env.history) == len(env.events_history) == 4
    assert env.events_history[0] == []
    for t in range(1, 4):
        assert all(event.t == t for event in env.events_history[t])


def test_episode_does_not_end_until_tissue_recovers():
    params = EmergentParams(
        pamp_arrival=99,
        pathogen_arrival=99,
        growth=0.0,
        kill=1.0,
        damage_from_pathogen=0.0,
        damage_from_inflammation=0.0,
        natural_healing=0.0,
    )

    def damaged_factory():
        graph = make_graph(
            Macrophage(
                "m",
                activation=1.0,
                phenotype=Phenotype.M1,
                pro_score=1.0,
                action_activation=0.0,
            ),
            Pathogen("pathogen", level=0.1),
            TissueDamage(ID_DAMAGE, level=0.5),
        )
        return graph

    env = ImmuneEnvironment(
        damaged_factory,
        EmergentDynamics(params, seed=0),
        ImmuneHomeostasisReward(),
        max_steps=5,
    )
    _, _, terminated, _, info = env.step({})
    assert info["P"] == 0.0
    assert not terminated

    def healed_factory():
        graph = damaged_factory()
        graph.entity(ID_DAMAGE).level = 0.0
        return graph

    env = ImmuneEnvironment(
        healed_factory,
        EmergentDynamics(params, seed=0),
        ImmuneHomeostasisReward(),
        max_steps=5,
    )
    _, _, terminated, _, _ = env.step({})
    assert terminated
