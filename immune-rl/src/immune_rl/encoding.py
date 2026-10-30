"""Transforma Observation em estado discreto para a Q-table."""

from bisect import bisect_right

from .entities import NodeType, Observation

_PHENOTYPE_BIN = {"M0": 0, "M1": 1, "M2": 2}


class StateEncoder:
    """Encoder enxuto do piloto (ativação + patógeno + citocina)."""

    def __init__(self, own_edges=(0.33, 0.66), neighbor_edges=None):
        self.own_edges = own_edges
        self.neighbor_edges = neighbor_edges or {
            NodeType.PATHOGEN: (0.25, 0.75),
            NodeType.PRO_CYTOKINE: (0.20, 0.60),
        }

    def encode(self, obs: Observation) -> tuple:
        parts = [bisect_right(self.own_edges, obs.own["activation"])]
        for node_type, edges in self.neighbor_edges.items():
            total = sum(
                neighbor.signal
                for neighbor in obs.neighbors.values()
                if neighbor.node_type == node_type
            )
            parts.append(bisect_right(edges, total))
        return tuple(parts)


class EmergentStateEncoder:
    """Estado do mundo emergente: fenótipo, ativação, mediadores e dano."""

    neighbor_types = (
        NodeType.PAMP,
        NodeType.PATHOGEN,
        NodeType.PRO_CYTOKINE,
        NodeType.ANTI_CYTOKINE,
        NodeType.CHEMOKINE,
        NodeType.REPAIR_FACTOR,
    )

    def __init__(
        self,
        activation_edges=(0.33, 0.66),
        signal_edges=(0.20, 0.50),
        damage_edges=(0.10, 0.35),
    ):
        self.activation_edges = activation_edges
        self.signal_edges = signal_edges
        self.damage_edges = damage_edges

    def encode(self, obs: Observation) -> tuple:
        phenotype = obs.own.get("phenotype", "M0")
        parts = [
            _PHENOTYPE_BIN.get(phenotype, 0),
            bisect_right(self.activation_edges, obs.own["activation"]),
        ]
        for node_type in self.neighbor_types:
            total = sum(
                neighbor.signal
                for neighbor in obs.neighbors.values()
                if neighbor.node_type == node_type
            )
            parts.append(bisect_right(self.signal_edges, total))
        parts.append(bisect_right(self.damage_edges, obs.own.get("tissue_damage", 0.0)))
        return tuple(parts)
