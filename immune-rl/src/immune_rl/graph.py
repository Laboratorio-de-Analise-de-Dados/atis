"""Grafo imunológico que encapsula o NetworkX"""

import copy
import networkx as nx
from .entities import (Entity, Cell, EdgeType, Observation, PAMP, Pathogen, Cytokine, Macrophage)

class ImmuneGraph:
    def __init__(self):
        self._g = nx.DiGraph()

    def add_entity(self, entity: Entity) -> None:
        self._g.add_node(entity.id, entity=entity)

    def connect(self, src: str, dst: str, edge_type: EdgeType, strength: float) -> None:
        self._g.add_edge(src, dst, type=edge_type, strength=strength)

    def entity(self, id: str) -> Entity:
        return self._g.nodes[id]["entity"]

    def entities(self, kind: type = Entity) -> list:
        return [d["entity"] for _, d in self._g.nodes(data=True)
                if isinstance(d["entity"], kind)]

    def cells(self) -> list[Cell]:
        return self.entities(Cell)

    def total_signal(self, kind: type) -> float:
        return sum(e.signal() for e in self.entities(kind))

    def observe(self, cell_id: str) -> Observation:
        nbrs = set(self._g.predecessors(cell_id)) | set(self._g.successors(cell_id))
        return Observation(
            own=self.entity(cell_id).state(),
            neighbors={n: self.entity(n).signal() for n in nbrs}
        )

    def snapshot(self) -> "ImmuneGraph":
        """Cópia independente para guardar G_t no histórico"""
        other = ImmuneGraph()
        other._g = copy.deepcopy(self._g)
        return other

    @property 
    def nx(self) -> nx.DiGraph:
        """Acesso ao grafo para análise e visualização"""
        return self._g


def build_pilot_graph() -> ImmuneGraph:
    """Constrói um piloto para testes"""
    g = ImmuneGraph()
    g.add_entity(PAMP("pamp", level=0.8))
    g.add_entity(Pathogen("pathogen", level=1.0))
    g.add_entity(Cytokine("cytokine", level=0.0))
    for m in ("mac_A", "mac_B"):
        g.add_entity(Macrophage(m))
        g.connect("pamp", m, EdgeType.RECOGNITION, 0.8)
        g.connect(m, "pathogen" ,EdgeType.ACTIVATION, 0.5)
        g.connect(m, "cytokine", EdgeType.PRODUCTION, 0.3)
    return g