"""Grafo imunológico que encapsula o NetworkX."""

import copy

import networkx as nx

from .entities import (
    PAMP,
    Cell,
    EdgeType,
    Entity,
    Macrophage,
    NeighborView,
    Observation,
    Pathogen,
    ProCytokine,
    TissueDamage,
)


class ImmuneGraph:
    def __init__(self) -> None:
        self._g = nx.DiGraph()
        self.time = 0

    def add_entity(self, entity: Entity) -> None:
        if self.has_entity(entity.id):
            raise ValueError(f"entidade '{entity.id}' já existe")
        self._g.add_node(entity.id, entity=entity)

    def has_entity(self, entity_id: str) -> bool:
        return entity_id in self._g

    def remove_entity(self, entity_id: str) -> None:
        self._g.remove_node(entity_id)

    def connect(self, src: str, dst: str, edge_type: EdgeType, strength: float) -> None:
        if not self.has_entity(src):
            raise KeyError(f"origem '{src}' não existe")
        if not self.has_entity(dst):
            raise KeyError(f"destino '{dst}' não existe")
        self._g.add_edge(src, dst, type=edge_type, strength=strength)

    def disconnect(self, src: str, dst: str) -> None:
        if self._g.has_edge(src, dst):
            self._g.remove_edge(src, dst)

    def has_edge(self, src: str, dst: str) -> bool:
        return self._g.has_edge(src, dst)

    def entity(self, entity_id: str) -> Entity:
        return self._g.nodes[entity_id]["entity"]

    def cell(self, cell_id: str) -> Cell:
        entity = self.entity(cell_id)
        if not isinstance(entity, Cell):
            raise TypeError(f"'{cell_id}' não é uma célula (é {type(entity).__name__})")
        return entity

    def entities(self, kind: type = Entity) -> list:
        return [
            data["entity"]
            for _, data in self._g.nodes(data=True)
            if isinstance(data["entity"], kind)
        ]

    def cells(self) -> list[Cell]:
        return self.entities(Cell)

    def total_signal(self, kind: type) -> float:
        return sum(entity.signal() for entity in self.entities(kind))

    def in_edges(self, entity_id: str) -> list[tuple[Entity, dict]]:
        return [
            (self.entity(src), data)
            for src, _, data in self._g.in_edges(entity_id, data=True)
        ]

    def out_edges(self, entity_id: str) -> list[tuple[Entity, dict]]:
        return [
            (self.entity(dst), data)
            for _, dst, data in self._g.out_edges(entity_id, data=True)
        ]

    def next_id(self, prefix: str) -> str:
        index = 1
        while self.has_entity(f"{prefix}_{index}"):
            index += 1
        return f"{prefix}_{index}"

    def observe(self, cell_id: str) -> Observation:
        center = self.entity(cell_id)
        own = center.state()
        own["tissue_damage"] = self.total_signal(TissueDamage)

        neighbors: dict[str, NeighborView] = {}
        for neighbor_id in set(self._g.predecessors(cell_id)) | set(self._g.successors(cell_id)):
            neighbor = self.entity(neighbor_id)
            edge_types: list[EdgeType] = []
            strength = 0.0
            if self._g.has_edge(neighbor_id, cell_id):
                data = self._g[neighbor_id][cell_id]
                edge_types.append(data["type"])
                strength += data.get("strength", 0.0)
            if self._g.has_edge(cell_id, neighbor_id):
                data = self._g[cell_id][neighbor_id]
                edge_types.append(data["type"])
                strength += data.get("strength", 0.0)
            neighbors[neighbor_id] = NeighborView(
                node_type=neighbor.node_type,
                signal=neighbor.signal(),
                edge_types=tuple(edge_types),
                strength=strength,
            )

        return Observation(own=own, neighbors=neighbors)

    def snapshot(self) -> "ImmuneGraph":
        """Cópia independente para guardar G_t no histórico."""
        other = ImmuneGraph()
        other._g = copy.deepcopy(self._g)
        other.time = self.time
        return other

    @property
    def nx(self) -> nx.DiGraph:
        """Acesso ao grafo para análise e visualização."""
        return self._g


def build_pilot_graph() -> ImmuneGraph:
    """Constrói o piloto simples usado nos testes unitários."""
    graph = ImmuneGraph()
    graph.add_entity(PAMP("pamp", level=0.8))
    graph.add_entity(Pathogen("pathogen", level=1.0))
    graph.add_entity(ProCytokine("cytokine", level=0.0))
    for macrophage_id in ("mac_A", "mac_B"):
        graph.add_entity(Macrophage(macrophage_id))
        graph.connect("pamp", macrophage_id, EdgeType.RECOGNITION, 0.8)
        graph.connect(macrophage_id, "pathogen", EdgeType.ACTIVATION, 0.5)
        graph.connect(macrophage_id, "cytokine", EdgeType.PRODUCTION, 0.3)
    return graph
