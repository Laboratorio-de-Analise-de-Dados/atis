import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from immune_rl.dynamics import ImmuneHomeostasisReward
from immune_rl.emergent import EmergentDynamics, EmergentParams, build_seed_graph
from immune_rl.entities import Action
from immune_rl.environment import ImmuneEnvironment
from immune_rl.visualization import animate, interactive_viewer, plot_snapshots


def make_history(steps=4):
    params = EmergentParams(pamp_arrival=1, pathogen_arrival=2, p_recognize=1.0)
    env = ImmuneEnvironment(
        build_seed_graph,
        EmergentDynamics(params, seed=0),
        ImmuneHomeostasisReward(),
        max_steps=10,
    )
    env.reset()
    for _ in range(steps):
        actions = {cell.id: Action.SECRETE_PRO for cell in env.graph.cells()}
        env.step(actions)
    return env.history, env.events_history


def test_plot_snapshots_saves_figure(tmp_path):
    history, events = make_history()
    path = tmp_path / "snapshots.png"

    plot_snapshots(history, events_history=events, save=path)

    assert path.exists()
    assert path.stat().st_size > 0
    plt.close("all")


def test_animate_saves_gif(tmp_path):
    history, events = make_history(steps=3)
    path = tmp_path / "evolucao.gif"

    animation = animate(history, events_history=events, interval=100, save=path)

    assert path.exists()
    assert path.stat().st_size > 0
    plt.close(animation._fig)


def test_interactive_viewer_builds_slider():
    history, events = make_history()

    figure, slider = interactive_viewer(history, events_history=events)

    assert slider.val == 0
    assert slider.valmax == len(history) - 1
    plt.close(figure)
