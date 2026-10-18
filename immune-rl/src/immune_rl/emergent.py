"""Mundo emergente: o grafo começa com UM macrófago e a estrutura surge por eventos.

Sequência (cada passo é uma REGRA, não um roteiro por tempo, exceto as chegadas):
    1. chega PAMP                      -> nó novo
    2. macrófago reconhece PAMP        -> aresta PAMP -> célula (reconhecimento)
    3. célula ativada produz           -> nós citocina e quimiocina + arestas de produção
    4. quimiocina recruta              -> nós macrófago novos (aresta de recrutamento)
    5. citocina "liga" a certas células-> aresta citocina -> célula (ativação), sorteada
    6. citocina estimula uma saída     -> nó output + aresta de estímulo
    7. chega o patógeno                -> nó novo; células ativas o engajam (interação)
    8. patógeno eliminado / sinais decaem -> nós e arestas DESAPARECEM
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .dynamics import Dynamics
from .entities import Cell, Chemokine, Cytokine, EdgeType, Macrophage, Output, PAMP, Pathogen
from .graph import ImmuneGraph


@dataclass
class EmergentParams:
    # chegadas externas (passo em que o nó aparece)
    pamp_arrival: int = 2
    pathogen_arrival: int = 8
    pamp_level: float = 0.8
    pathogen_load: float = 0.6
    # reconhecimento e ativação
    p_recognize: float = 0.7        # chance por passo de uma célula reconhecer o PAMP
    recognition_strength: float = 0.8
    gain: float = 0.20              # ativação ganha por sinal recebido
    decay: float = 0.10             # decaimento natural da ativação
    # produção
    produce_thr: float = 0.35       # ativação mínima para produzir
    cyto_prod: float = 0.25
    chemo_prod: float = 0.20
    signal_decay: float = 0.10      # decaimento de citocinas/quimiocinas
    # recrutamento
    recruit_thr: float = 0.15
    p_recruit: float = 0.5
    max_cells: int = 5
    recruit_activation: float = 0.1
    # sinalização citocina -> célula / output
    bind_thr: float = 0.20
    unbind_thr: float = 0.05
    p_bind: float = 0.4
    output_thr: float = 0.30
    output_strength: float = 0.6
    # patógeno
    engage_thr: float = 0.30
    disengage_thr: float = 0.15
    growth: float = 0.08
    kill: float = 0.30
    eradication: float = 0.02
    # PAMP decai, mas é reposto pelo patógeno
    pamp_decay: float = 0.08
    pamp_resupply: float = 0.10
    # abaixo disto o nó some do grafo
    fade_thr: float = 0.02


def build_seed_graph() -> ImmuneGraph:
    """Estado inicial: só um macrófago em repouso."""
    g = ImmuneGraph()
    g.add_entity(Macrophage("mac_1", activation=0.2))
    return g


class EmergentDynamics(Dynamics):
    def __init__(self, params: EmergentParams | None = None, seed: int = 0):
        self.p = params or EmergentParams()
        self.rng = random.Random(seed)

    def apply(self, g: ImmuneGraph) -> None:
        g.time += 1
        self._arrivals(g)
        self._recognition(g)
        self._activation(g)
        self._production(g)
        self._signaling(g)
        self._recruitment(g)
        self._pathogen(g)
        self._fade(g)

    # 1) chegadas externas ------------------------------------------------
    def _arrivals(self, g):
        p = self.p
        if g.time == p.pamp_arrival:
            g.add_entity(PAMP("pamp_1", level=p.pamp_level))
        if g.time == p.pathogen_arrival:
            g.add_entity(Pathogen("pathogen_1", level=p.pathogen_load))

    # 2) reconhecimento: surgem arestas PAMP -> célula ----------------------
    def _recognition(self, g):
        for pamp in g.entities(PAMP):
            for cell in g.cells():
                if not g.has_edge(pamp.id, cell.id) and self.rng.random() < self.p.p_recognize:
                    g.connect(pamp.id, cell.id, EdgeType.RECOGNITION, self.p.recognition_strength)

    # a ativação da célula depende dos sinais que chegam pelas ARESTAS ---------
    def _activation(self, g):
        p = self.p
        for cell in g.cells():
            drive = sum(src.signal() * a["strength"] for src, a in g.in_edges(cell.id)
                        if a["type"] in (EdgeType.RECOGNITION, EdgeType.ACTIVATION))
            cell.activation = min(1.0, max(0.0, cell.activation + p.gain * drive - p.decay * cell.activation))

    # 3) produção: surgem nós citocina/quimiocina e arestas de produção --------
    def _production(self, g):
        p = self.p
        producers = [c for c in g.cells() if c.activation > p.produce_thr]
        producer_ids = {c.id for c in producers}
        for cls, node_id, rate in ((Cytokine, "cyto_pro", p.cyto_prod),
                                    (Chemokine, "chemokine", p.chemo_prod)):
            if producers and not g.has_entity(node_id):
                g.add_entity(cls(node_id, level=0.0))
            if not g.has_entity(node_id):
                continue
            for c in g.cells():
                if c.id in producer_ids:
                    g.connect(c.id, node_id, EdgeType.PRODUCTION, c.activation)
                elif g.has_edge(c.id, node_id):
                    g.disconnect(c.id, node_id)
            node = g.entity(node_id)
            node.level = max(0.0, node.level * (1 - p.signal_decay)
                             + rate * sum(c.activation for c in producers))

    # 5/6) citocina ativa certas células (sorteio) e estimula um output ---------
    def _signaling(self, g):
        p = self.p
        for cyto in g.entities(Cytokine):
            for cell in g.cells():
                linked = g.has_edge(cyto.id, cell.id)
                if cyto.level > p.bind_thr and not linked and self.rng.random() < p.p_bind:
                    g.connect(cyto.id, cell.id, EdgeType.ACTIVATION, round(self.rng.uniform(0.3, 0.9), 2))
                elif cyto.level < p.unbind_thr and linked:
                    g.disconnect(cyto.id, cell.id)
            if cyto.level > p.output_thr and not g.has_entity("output_1"):
                g.add_entity(Output("output_1", level=0.0))
                g.connect(cyto.id, "output_1", EdgeType.STIMULATION, p.output_strength)
        for out in g.entities(Output):
            drive = sum(src.signal() * a["strength"] for src, a in g.in_edges(out.id)
                        if a["type"] == EdgeType.STIMULATION)
            out.level = 0.6 * out.level + 0.4 * drive

    # 4) recrutamento: quimiocina faz surgir macrófagos novos --------------------
    def _recruitment(self, g):
        p = self.p
        for chem in g.entities(Chemokine):
            if (chem.level > p.recruit_thr and len(g.cells()) < p.max_cells
                    and self.rng.random() < p.p_recruit):
                new_id = g.next_id("mac")
                g.add_entity(Macrophage(new_id, activation=p.recruit_activation))
                g.connect(chem.id, new_id, EdgeType.RECRUITMENT, min(1.0, chem.level))
                chem.level *= 0.8

    # 7) patógeno: células ativas o engajam; ele cresce e é morto ----------------
    def _pathogen(self, g):
        p = self.p
        for path in g.entities(Pathogen):
            for cell in g.cells():
                linked = g.has_edge(cell.id, path.id)
                if cell.activation > p.engage_thr and not linked:
                    g.connect(cell.id, path.id, EdgeType.INTERACTION, cell.activation)
                elif cell.activation <= p.disengage_thr and linked:
                    g.disconnect(cell.id, path.id)
                elif linked:
                    g.connect(cell.id, path.id, EdgeType.INTERACTION, cell.activation)  # atualiza o peso
            killing = sum(src.activation * a["strength"] for src, a in g.in_edges(path.id)
                          if isinstance(src, Cell) and a["type"] == EdgeType.INTERACTION)
            path.level = max(0.0, path.level * (1 + p.growth - p.kill * killing))
        supply = p.pamp_resupply * g.total_signal(Pathogen)
        for pamp in g.entities(PAMP):
            pamp.level = min(1.0, pamp.level * (1 - p.pamp_decay) + supply)

    # 8) o que decaiu some do grafo (com todas as suas arestas) ------------------
    def _fade(self, g):
        p = self.p
        for path in g.entities(Pathogen):
            if path.level <= p.eradication:
                g.remove_entity(path.id)      # critério de erradicação
        for kind in (PAMP, Cytokine, Chemokine, Output):
            for e in g.entities(kind):
                if e.level < p.fade_thr:
                    g.remove_entity(e.id)