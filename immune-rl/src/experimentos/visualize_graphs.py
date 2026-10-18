"""Visualiza a evolução do grafo em um episódio.

Exemplos (a partir da pasta do projeto):
  uv run python experimentos/visualize_graphs.py
  uv run python experimentos/visualize_graphs.py --times 0 3 6 -1
  uv run python experimentos/visualize_graphs.py --gif figures/evolucao.gif
  uv run python experimentos/visualize_graphs.py --viewer      # slider interativo
  uv run python experimentos/visualize_graphs.py --play        # animação em janela
"""
import argparse
import random

import matplotlib.pyplot as plt

from immune_rl.dynamics import PathogenInflammationReward, SimpleDynamics
from immune_rl.encoding import StateEncoder
from immune_rl.entities import Action
from immune_rl.environment import ImmuneEnvironment
from immune_rl.graph import build_pilot_graph
from immune_rl.policies import FixedPolicy, QLearningPolicy, RandomPolicy
from immune_rl.visualization import animate, interactive_viewer, plot_snapshots


def make_env():
    return ImmuneEnvironment(build_pilot_graph, SimpleDynamics(), PathogenInflammationReward())


def train(env, policies, episodes):
    for _ in range(episodes):
        obs = env.reset()
        while True:
            actions = {c: policies[c].select_action(o) for c, o in obs.items()}
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            for c in obs:
                policies[c].update(obs[c], actions[c], rewards[c], next_obs[c], terminated)
            obs = next_obs
            if terminated or truncated:
                break
        for p in set(policies.values()):
            p.end_episode()


def rollout(env, policies):
    """Roda 1 episódio e devolve o histórico G_0 ... G_T."""
    obs = env.reset()
    while True:
        actions = {c: policies[c].select_action(o) for c, o in obs.items()}
        obs, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            return env.history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["trained", "random", "activate"], default="trained")
    ap.add_argument("--episodes", type=int, default=600, help="episódios de treino (policy=trained)")
    ap.add_argument("--times", type=int, nargs="+", default=None, help="tempos a mostrar, ex.: 0 3 6 -1")
    ap.add_argument("--layout", choices=["layered", "spring", "circular", "shell"], default="layered")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshots", default="figures/snapshots.png", help="onde salvar a figura")
    ap.add_argument("--gif", default=None, help="salvar animação (.gif ou .mp4)")
    ap.add_argument("--viewer", action="store_true", help="abrir visualizador com slider")
    ap.add_argument("--play", action="store_true", help="abrir animação em janela")
    ap.add_argument("--show", action="store_true", help="abrir a figura de tempos em janela")
    args = ap.parse_args()

    random.seed(args.seed)
    env = make_env()
    cells = [c.id for c in env.graph.cells()]

    if args.policy == "trained":
        shared = QLearningPolicy(StateEncoder(), seed=args.seed)
        policies = {c: shared for c in cells}
        print(f"Treinando {args.episodes} episódios...")
        train(env, policies, args.episodes)
        shared.epsilon = 0.0
    elif args.policy == "random":
        policies = {c: RandomPolicy() for c in cells}
    else:
        policies = {c: FixedPolicy(Action.ACTIVATE) for c in cells}

    history = rollout(env, policies)
    print(f"Episódio com {len(history)} grafos (G_0 ... G_{len(history) - 1}).")

    plot_snapshots(history, args.times, layout=args.layout, seed=args.seed, save=args.snapshots)
    print(f"Figura salva em {args.snapshots}")

    anim = None
    if args.gif:
        animate(history, layout=args.layout, seed=args.seed, save=args.gif)
        print(f"Animação salva em {args.gif}")
    if args.play:
        anim = animate(history, layout=args.layout, seed=args.seed)   # noqa: F841 (manter referência)
    viewer = None
    if args.viewer:
        viewer = interactive_viewer(history, layout=args.layout, seed=args.seed)  # noqa: F841

    if args.show or args.play or args.viewer:
        plt.show()


if __name__ == "__main__":
    main()