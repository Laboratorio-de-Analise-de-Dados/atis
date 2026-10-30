"""Demonstração rápida do mundo emergente."""


def main() -> None:
    """Roda um episódio demonstrativo e imprime a narração dos eventos."""
    from .dynamics import ImmuneHomeostasisReward
    from .emergent import EmergentDynamics, build_seed_graph
    from .entities import Action, Pathogen, TissueDamage
    from .environment import ImmuneEnvironment

    env = ImmuneEnvironment(
        build_seed_graph,
        EmergentDynamics(seed=0),
        ImmuneHomeostasisReward(),
        max_steps=60,
    )
    env.reset()

    while True:
        actions = {}
        pathogen = env.graph.total_signal(Pathogen)
        damage = env.graph.total_signal(TissueDamage)
        for cell in env.graph.cells():
            if pathogen > 0.01:
                actions[cell.id] = Action.SECRETE_PRO
            elif damage > 0.10:
                actions[cell.id] = Action.SECRETE_REPAIR
            else:
                actions[cell.id] = Action.SECRETE_ANTI

        _, rewards, terminated, truncated, info = env.step(actions)
        reward = next(iter(rewards.values()), 0.0)
        messages = "; ".join(event.message for event in info["events"])
        print(f"t={env.t:02d} R={reward:+.3f} {info['outcome']}: {messages or 'sem eventos'}")
        if terminated or truncated:
            break

    print(
        f"Fim: {info['outcome']} | P={info['P']:.2f} "
        f"dano={info['D']:.2f} M1={info['M1']} M2={info['M2']}"
    )
