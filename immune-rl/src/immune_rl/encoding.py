"""Transforma Observation em estado discreto para Q-table"""

from bisect import bisect_right

from .entities import NodeType, Observation


class StateEncoder:
    def __init__(self, own_edges=(0.33, 0.66), neighbor_edges=None):
        self.own_edges = own_edges
        self.neighbor_edges = neighbor_edges or {
            NodeType.PATHOGEN: (0.25, 0.75),
            NodeType.CYTOKINE: (0.20, 0.60),
        }

    def encode(self, obs: Observation) -> tuple:
        parts = [bisect_right(self.own_edges, obs.own["activation"])]
        for ntype, edges in self.neighbor_edges.items():
            total = sum(n.signal for n in obs.neighbors.values()
                        if n.node_type == ntype)
            parts.append(bisect_right(edges, total))
        return tuple(parts)