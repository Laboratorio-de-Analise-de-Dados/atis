"""Visualiza a evolução do grafo em um episódio.

Exemplos (a partir da pasta do projeto):
  uv run python src/experimentos/visualize_graphs.py
  uv run python src/experimentos/visualize_graphs.py --times 0 3 6 -1
  uv run python src/experimentos/visualize_graphs.py --gif src/figures/evolucao.gif
  uv run python src/experimentos/visualize_graphs.py --viewer
  uv run python src/experimentos/visualize_graphs.py --narrate
"""

import argparse

from immune_rl.dynamics import ImmuneHomeostasisReward, SimpleDynamics
from immune_rl.emergent import EmergentDynamics, build_seed_graph
from immune_rl.encoding import EmergentStateEncoder, StateEncoder
from immune_rl.entities import Action
from immune_rl.environment import ImmuneEnvironment
from immune_rl.graph import build_pilot_graph
from immune_rl.policies import FixedPolicy, QLearningPolicy, RandomPolicy


def make_env(mode: str, seed: int, max_steps: int = 60) -> ImmuneEnvironment:
    if mode == "emergent":
        return ImmuneEnvironment(
            build_seed_graph,
            EmergentDynamics(seed=seed),
            ImmuneHomeostasisReward(),
            max_steps=max_steps,
        )
    return ImmuneEnvironment(
        build_pilot_graph,
        SimpleDynamics(),
        ImmuneHomeostasisReward(),
        max_steps=max_steps,
    )


def make_encoder(mode: str):
    return EmergentStateEncoder() if mode == "emergent" else StateEncoder()


def build_policy(kind: str, seed: int, encoder):
    if kind == "random":
        return RandomPolicy()
    if kind == "pro":
        return FixedPolicy(Action.SECRETE_PRO)
    if kind == "anti":
        return FixedPolicy(Action.SECRETE_ANTI)
    if kind == "chemo":
        return FixedPolicy(Action.SECRETE_CHEMO)
    if kind == "repair":
        return FixedPolicy(Action.SECRETE_REPAIR)
    return QLearningPolicy(encoder, seed=seed)


def ensure_policies(env, policies, kind, encoder, seed):
    for cell in env.graph.cells():
        if cell.id in policies:
            continue
        if kind == "trained":
            shared = next(
                (policy for policy in policies.values() if isinstance(policy, QLearningPolicy)),
                None,
            )
            policies[cell.id] = shared or QLearningPolicy(encoder, seed=seed)
        else:
            policies[cell.id] = build_policy(kind, seed + len(policies), encoder)


def train(env, policies, episodes, kind, encoder, seed):
    for _ in range(episodes):
        obs = env.reset()
        while True:
            ensure_policies(env, policies, kind, encoder, seed)
            actions = {
                cell_id: policies[cell_id].select_action(observation)
                for cell_id, observation in obs.items()
            }
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            for cell_id in obs:
                if cell_id in next_obs and cell_id in rewards:
                    policies[cell_id].update(
                        obs[cell_id],
                        actions[cell_id],
                        rewards[cell_id],
                        next_obs[cell_id],
                        terminated,
                    )
            obs = next_obs
            if terminated or truncated:
                break
        for policy in set(policies.values()):
            policy.end_episode()


def rollout(env, policies, kind, encoder, seed, narrate: bool = False):
    obs = env.reset()
    while True:
        ensure_policies(env, policies, kind, encoder, seed)
        actions = {
            cell_id: policies[cell_id].select_action(observation)
            for cell_id, observation in obs.items()
        }
        obs, _, terminated, truncated, info = env.step(actions)
        if narrate:
            messages = "; ".join(event.message for event in info["events"])
            print(f"t={env.t:02d} {info['outcome']}: {messages or 'sem eventos'}")
        if terminated or truncated:
            return env.history, env.events_history, info


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["emergent", "pilot"], default="emergent")
    parser.add_argument(
        "--policy",
        choices=["trained", "random", "pro", "anti", "chemo", "repair"],
        default="trained",
    )
    parser.add_argument("--episodes", type=int, default=600, help="episódios de treino")
    parser.add_argument("--max-steps", type=int, default=60, help="passos por episódio")
    parser.add_argument("--times", type=int, nargs="+", default=None)
    parser.add_argument(
        "--layout",
        choices=["layered", "spring", "circular", "shell"],
        default="layered",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshots", default="figures/snapshots.png")
    parser.add_argument("--gif", default=None, help="salvar animação (.gif ou .mp4)")
    parser.add_argument("--viewer", action="store_true", help="abrir visualizador interativo")
    parser.add_argument("--play", action="store_true", help="abrir animação em janela")
    parser.add_argument("--show", action="store_true", help="abrir a figura de snapshots")
    parser.add_argument("--narrate", action="store_true", help="imprimir eventos de cada passo")
    args = parser.parse_args()

    interactive = args.viewer or args.play or args.show
    if not interactive:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from immune_rl.visualization import animate, interactive_viewer, plot_snapshots

    env = make_env(args.mode, args.seed, args.max_steps)
    encoder = make_encoder(args.mode)
    policies = {
        cell.id: build_policy(args.policy, args.seed, encoder) for cell in env.graph.cells()
    }

    if args.policy == "trained":
        print(f"Treinando {args.episodes} episódios no modo {args.mode}...")
        train(env, policies, args.episodes, args.policy, encoder, args.seed)
        for policy in set(policies.values()):
            if isinstance(policy, QLearningPolicy):
                policy.epsilon = 0.0

    history, events_history, info = rollout(
        env, policies, args.policy, encoder, args.seed, narrate=args.narrate
    )
    print(
        f"Episódio com {len(history)} grafos; resultado: {info['outcome']}; "
        f"dano final {info['D']:.2f}; macrófagos {len(env.graph.cells())}."
    )

    plot_snapshots(
        history,
        args.times,
        events_history=events_history,
        layout=args.layout,
        seed=args.seed,
        save=args.snapshots,
    )
    print(f"Figura salva em {args.snapshots}")

    if args.gif:
        animate(
            history,
            events_history=events_history,
            layout=args.layout,
            seed=args.seed,
            save=args.gif,
        )
        print(f"Animação salva em {args.gif}")
    if args.play:
        _animation = animate(
            history, events_history=events_history, layout=args.layout, seed=args.seed
        )
    if args.viewer:
        _viewer = interactive_viewer(
            history, events_history=events_history, layout=args.layout, seed=args.seed
        )

    if interactive:
        plt.show()


if __name__ == "__main__":
    main()
