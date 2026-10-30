"""Mundo emergente: o grafo começa com um macrófago e a resposta surge por eventos.

Cada passo executa regras, não um roteiro fixo. A sequência esperada é:

    1. chega o PAMP;
    2. o macrófago reconhece o PAMP;
    3. macrófagos secretam mediadores conforme a ação escolhida;
    4. citocinas pró ativam macrófagos já ativados (feedback positivo);
    5. a quimiocina recruta novos macrófagos;
    6. IL-10 regula a ativação e a produção pró-inflamatória;
    7. chega o patógeno; apenas macrófagos M1 o eliminam;
    8. TGF-beta reduz o dano tecidual causado por patógeno e inflamação;
    9. sinais decaem e nós abaixo do limiar desaparecem.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .dynamics import Dynamics
from .entities import (
    PAMP,
    Action,
    AntiCytokine,
    Cell,
    Chemokine,
    EdgeType,
    Macrophage,
    Pathogen,
    Phenotype,
    ProCytokine,
    RepairFactor,
    TissueDamage,
)
from .events import Event, EventKind, EventSeverity
from .graph import ImmuneGraph

ID_PAMP = "pamp_1"
ID_PATHOGEN = "pathogen_1"
ID_PRO = "pro_1"
ID_ANTI = "anti_1"
ID_CHEMO = "chemo_1"
ID_REPAIR = "repair_1"
ID_DAMAGE = "damage_1"

MEDIATOR_IDS = (ID_PRO, ID_ANTI, ID_CHEMO, ID_REPAIR)


@dataclass
class EmergentParams:
    # chegadas externas
    pamp_arrival: int = 2
    pathogen_arrival: int = 8
    pamp_level: float = 0.8
    pathogen_load: float = 0.6

    # reconhecimento
    p_recognize: float = 0.8
    recognition_strength: float = 0.8

    # ativação
    gain: float = 0.20
    decay: float = 0.10
    pro_feedback_thr: float = 0.10
    pro_feedback_gain: float = 0.25
    anti_inhibition: float = 0.35
    bind_thr: float = 0.10
    unbind_thr: float = 0.03

    # secreção
    pro_rate: float = 0.10
    anti_rate: float = 0.10
    chemo_rate: float = 0.08
    repair_rate: float = 0.10
    m1_bonus: float = 1.3
    m2_bonus: float = 1.3
    anti_suppression: float = 0.5
    signal_decay: float = 0.15

    # recrutamento
    recruit_thr: float = 0.15
    p_recruit: float = 0.5
    max_cells: int = 5
    recruit_activation: float = 0.10

    # patógeno
    engage_thr: float = 0.25
    disengage_thr: float = 0.10
    growth: float = 0.08
    kill: float = 0.40
    eradication: float = 0.02

    # dano e reparo
    damage_from_pathogen: float = 0.04
    damage_from_inflammation: float = 0.06
    repair_efficiency: float = 0.25
    natural_healing: float = 0.005

    # PAMP
    pamp_decay: float = 0.08
    pamp_resupply: float = 0.10

    # abaixo deste nível o nó some do grafo
    fade_thr: float = 0.02


def build_seed_graph() -> ImmuneGraph:
    """Estado inicial: um macrófago em repouso e o tecido intacto."""
    graph = ImmuneGraph()
    graph.add_entity(Macrophage("mac_1", activation=0.2, action_activation=0.0))
    graph.add_entity(TissueDamage(ID_DAMAGE, level=0.0))
    return graph


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


class EmergentDynamics(Dynamics):
    def __init__(self, params: EmergentParams | None = None, seed: int = 0):
        self.p = params or EmergentParams()
        self.rng = random.Random(seed)

    def apply(self, graph: ImmuneGraph) -> list[Event]:
        graph.time += 1
        events: list[Event] = []
        self._arrivals(graph, events)
        self._recognition(graph, events)
        self._secretion(graph, events)
        self._signaling(graph, events)
        self._activation(graph, events)
        self._recruitment(graph, events)
        self._pathogen(graph, events)
        self._damage(graph, events)
        self._fade(graph, events)
        return events

    # 1) chegadas externas -------------------------------------------------
    def _arrivals(self, graph: ImmuneGraph, events: list[Event]) -> None:
        p = self.p
        if graph.time == p.pamp_arrival and not graph.has_entity(ID_PAMP):
            graph.add_entity(PAMP(ID_PAMP, level=p.pamp_level))
            events.append(
                Event(
                    graph.time,
                    EventKind.PAMP_ARRIVAL,
                    "PAMP liberado no tecido",
                    (ID_PAMP,),
                    severity=EventSeverity.WARNING,
                )
            )
        if graph.time == p.pathogen_arrival and not graph.has_entity(ID_PATHOGEN):
            graph.add_entity(Pathogen(ID_PATHOGEN, level=p.pathogen_load))
            events.append(
                Event(
                    graph.time,
                    EventKind.PATHOGEN_ARRIVAL,
                    f"patógeno chegou com carga {p.pathogen_load:.2f}",
                    (ID_PATHOGEN,),
                    severity=EventSeverity.CRITICAL,
                )
            )

    # 2) reconhecimento ----------------------------------------------------
    def _recognition(self, graph: ImmuneGraph, events: list[Event]) -> None:
        p = self.p
        for pamp in graph.entities(PAMP):
            for cell in graph.cells():
                if graph.has_edge(pamp.id, cell.id):
                    continue
                if self.rng.random() < p.p_recognize:
                    graph.connect(pamp.id, cell.id, EdgeType.RECOGNITION, p.recognition_strength)
                    events.append(
                        Event(
                            graph.time,
                            EventKind.RECOGNITION,
                            f"{cell.id} reconheceu {pamp.id}",
                            (cell.id, pamp.id),
                            ((pamp.id, cell.id),),
                        )
                    )

    # 3) secreção escolhida pelo agente ------------------------------------
    def _secretion(self, graph: ImmuneGraph, events: list[Event]) -> None:
        p = self.p
        for kind in (ProCytokine, AntiCytokine, Chemokine, RepairFactor):
            for mediator in graph.entities(kind):
                mediator.level *= 1 - p.signal_decay

        anti_level = graph.total_signal(AntiCytokine)

        for cell in graph.cells():
            for mediator_id in MEDIATOR_IDS:
                graph.disconnect(cell.id, mediator_id)

            action = cell.last_action
            if action == Action.NOOP:
                continue

            factor = 0.2 + 0.8 * cell.activation
            if action == Action.SECRETE_PRO:
                node = self._ensure(graph, ProCytokine, ID_PRO)
                rate = p.pro_rate * factor
                if cell.phenotype == Phenotype.M1:
                    rate *= p.m1_bonus
                rate *= max(0.0, 1.0 - p.anti_suppression * anti_level)
                node.level = min(2.0, node.level + rate)
                label = "citocina pró-inflamatória"
            elif action == Action.SECRETE_ANTI:
                node = self._ensure(graph, AntiCytokine, ID_ANTI)
                rate = p.anti_rate * factor
                if cell.phenotype == Phenotype.M2:
                    rate *= p.m2_bonus
                node.level = min(2.0, node.level + rate)
                label = "citocina anti-inflamatória"
            elif action == Action.SECRETE_CHEMO:
                node = self._ensure(graph, Chemokine, ID_CHEMO)
                rate = p.chemo_rate * factor
                if cell.phenotype == Phenotype.M1:
                    rate *= p.m1_bonus
                node.level = min(2.0, node.level + rate)
                label = "quimiocina"
            else:
                node = self._ensure(graph, RepairFactor, ID_REPAIR)
                rate = p.repair_rate * factor
                if cell.phenotype == Phenotype.M2:
                    rate *= p.m2_bonus
                node.level = min(2.0, node.level + rate)
                label = "fator de reparo"

            graph.connect(cell.id, node.id, EdgeType.PRODUCTION, min(1.0, rate * 2))
            events.append(
                Event(
                    graph.time,
                    EventKind.SECRETION,
                    f"{cell.id} secretou {label} ({rate:.2f})",
                    (cell.id, node.id),
                    ((cell.id, node.id),),
                )
            )

    # 4) sinalização: pró ativa, IL-10 regula, TGF-beta repara -------------
    def _signaling(self, graph: ImmuneGraph, events: list[Event]) -> None:
        p = self.p

        for pro in graph.entities(ProCytokine):
            for cell in graph.cells():
                linked = graph.has_edge(pro.id, cell.id)
                active = cell.activation > p.pro_feedback_thr
                if pro.level > p.bind_thr and active:
                    graph.connect(pro.id, cell.id, EdgeType.ACTIVATION, p.pro_feedback_gain)
                    if not linked:
                        events.append(
                            Event(
                                graph.time,
                                EventKind.ACTIVATION,
                                f"{pro.id} passou a ativar {cell.id}",
                                (pro.id, cell.id),
                                ((pro.id, cell.id),),
                            )
                        )
                elif linked and (pro.level < p.unbind_thr or not active):
                    graph.disconnect(pro.id, cell.id)

        for anti in graph.entities(AntiCytokine):
            for cell in graph.cells():
                linked = graph.has_edge(anti.id, cell.id)
                if anti.level > p.bind_thr:
                    graph.connect(anti.id, cell.id, EdgeType.REGULATION, p.anti_inhibition)
                    if not linked:
                        events.append(
                            Event(
                                graph.time,
                                EventKind.REGULATION,
                                f"{anti.id} passou a inibir {cell.id}",
                                (anti.id, cell.id),
                                ((anti.id, cell.id),),
                            )
                        )
                elif linked and anti.level < p.unbind_thr:
                    graph.disconnect(anti.id, cell.id)

        if graph.has_entity(ID_DAMAGE):
            for repair in graph.entities(RepairFactor):
                linked = graph.has_edge(repair.id, ID_DAMAGE)
                if repair.level > p.bind_thr:
                    graph.connect(repair.id, ID_DAMAGE, EdgeType.REPAIR, p.repair_efficiency)
                    if not linked:
                        events.append(
                            Event(
                                graph.time,
                                EventKind.REPAIR,
                                f"{repair.id} passou a reparar o tecido",
                                (repair.id, ID_DAMAGE),
                                ((repair.id, ID_DAMAGE),),
                            )
                        )
                elif linked and repair.level < p.unbind_thr:
                    graph.disconnect(repair.id, ID_DAMAGE)

    # 5) ativação ----------------------------------------------------------
    def _activation(self, graph: ImmuneGraph, events: list[Event]) -> None:
        p = self.p
        for cell in graph.cells():
            recognition_drive = sum(
                source.signal() * data["strength"]
                for source, data in graph.in_edges(cell.id)
                if data["type"] == EdgeType.RECOGNITION
            )
            pro_drive = sum(
                source.signal() * data["strength"]
                for source, data in graph.in_edges(cell.id)
                if data["type"] == EdgeType.ACTIVATION and isinstance(source, ProCytokine)
            )
            if cell.activation <= p.pro_feedback_thr:
                pro_drive = 0.0
            inhibition = sum(
                source.signal() * data["strength"]
                for source, data in graph.in_edges(cell.id)
                if data["type"] == EdgeType.REGULATION
            )

            previous = cell.activation
            cell.activation = _clamp(
                previous
                + p.gain * recognition_drive
                + pro_drive
                - p.decay * previous
                - inhibition
            )
            cell.memory = 0.9 * cell.memory + 0.1 * cell.activation

            if abs(cell.activation - previous) >= 0.08:
                kind = EventKind.ACTIVATION if cell.activation > previous else EventKind.REGULATION
                verb = "ativado" if cell.activation > previous else "regulado"
                events.append(
                    Event(
                        graph.time,
                        kind,
                        f"{cell.id} foi {verb} para {cell.activation:.2f}",
                        (cell.id,),
                    )
                )

    # 6) recrutamento ------------------------------------------------------
    def _recruitment(self, graph: ImmuneGraph, events: list[Event]) -> None:
        p = self.p
        for chemokine in graph.entities(Chemokine):
            if (
                chemokine.level > p.recruit_thr
                and len(graph.cells()) < p.max_cells
                and self.rng.random() < p.p_recruit
            ):
                new_id = graph.next_id("mac")
                graph.add_entity(
                    Macrophage(
                        new_id,
                        activation=p.recruit_activation,
                        action_activation=0.0,
                    )
                )
                graph.connect(
                    chemokine.id,
                    new_id,
                    EdgeType.RECRUITMENT,
                    min(1.0, chemokine.level),
                )
                chemokine.level *= 0.8
                events.append(
                    Event(
                        graph.time,
                        EventKind.RECRUITMENT,
                        f"{new_id} foi recrutado por {chemokine.id}",
                        (new_id, chemokine.id),
                        ((chemokine.id, new_id),),
                        EventSeverity.WARNING,
                    )
                )

    # 7) patógeno ----------------------------------------------------------
    def _pathogen(self, graph: ImmuneGraph, events: list[Event]) -> None:
        p = self.p
        for pathogen in graph.entities(Pathogen):
            for cell in graph.cells():
                is_m1 = cell.phenotype == Phenotype.M1
                linked = graph.has_edge(cell.id, pathogen.id)
                if is_m1 and cell.activation > p.engage_thr:
                    graph.connect(
                        cell.id,
                        pathogen.id,
                        EdgeType.INTERACTION,
                        cell.activation,
                    )
                    if not linked:
                        events.append(
                            Event(
                                graph.time,
                                EventKind.INTERACTION,
                                f"{cell.id} (M1) engajou o patógeno",
                                (cell.id, pathogen.id),
                                ((cell.id, pathogen.id),),
                            )
                        )
                elif linked and (not is_m1 or cell.activation <= p.disengage_thr):
                    graph.disconnect(cell.id, pathogen.id)

            killing = sum(
                data["strength"]
                for source, data in graph.in_edges(pathogen.id)
                if isinstance(source, Cell)
                and source.phenotype == Phenotype.M1
                and data["type"] == EdgeType.INTERACTION
            )
            previous = pathogen.level
            pathogen.level = max(
                0.0,
                pathogen.level * (1 + p.growth - p.kill * killing),
            )
            if pathogen.level < previous:
                events.append(
                    Event(
                        graph.time,
                        EventKind.KILLING,
                        f"patógeno reduzido de {previous:.2f} para {pathogen.level:.2f}",
                        (pathogen.id,),
                        severity=EventSeverity.WARNING,
                    )
                )

        supply = p.pamp_resupply * graph.total_signal(Pathogen)
        if graph.has_entity(ID_PAMP):
            pamp = graph.entity(ID_PAMP)
            pamp.level = min(1.0, pamp.level * (1 - p.pamp_decay) + supply)
        elif graph.total_signal(Pathogen) > 0:
            graph.add_entity(PAMP(ID_PAMP, level=min(1.0, supply)))
            events.append(
                Event(
                    graph.time,
                    EventKind.PAMP_ARRIVAL,
                    "novo PAMP liberado pelo patógeno",
                    (ID_PAMP,),
                    severity=EventSeverity.WARNING,
                )
            )

    # 8) dano e reparo -----------------------------------------------------
    def _damage(self, graph: ImmuneGraph, events: list[Event]) -> None:
        p = self.p
        if not graph.has_entity(ID_DAMAGE):
            graph.add_entity(TissueDamage(ID_DAMAGE, level=0.0))
        damage = graph.entity(ID_DAMAGE)

        repair = sum(
            source.signal() * data["strength"]
            for source, data in graph.in_edges(ID_DAMAGE)
            if data["type"] == EdgeType.REPAIR
        )
        previous = damage.level
        healing = p.natural_healing if previous > 0 else 0.0
        damage.level = _clamp(
            previous
            + p.damage_from_pathogen * graph.total_signal(Pathogen)
            + p.damage_from_inflammation * graph.total_signal(ProCytokine)
            - repair
            - healing
        )

        if damage.level > previous + 1e-6:
            events.append(
                Event(
                    graph.time,
                    EventKind.DAMAGE,
                    f"dano tecidual aumentou para {damage.level:.2f}",
                    (ID_DAMAGE,),
                    severity=EventSeverity.WARNING,
                )
            )
        elif damage.level < previous - 1e-6:
            events.append(
                Event(
                    graph.time,
                    EventKind.REPAIR,
                    f"dano tecidual reduziu para {damage.level:.2f}",
                    (ID_DAMAGE,),
                )
            )

    # 9) decaimento e remoção de nós --------------------------------------
    def _fade(self, graph: ImmuneGraph, events: list[Event]) -> None:
        p = self.p

        for pathogen in list(graph.entities(Pathogen)):
            if pathogen.level <= p.eradication:
                graph.remove_entity(pathogen.id)
                events.append(
                    Event(
                        graph.time,
                        EventKind.PATHOGEN_ERADICATED,
                        "patógeno eliminado",
                        (pathogen.id,),
                        severity=EventSeverity.CRITICAL,
                    )
                )

        for kind in (PAMP, ProCytokine, AntiCytokine, Chemokine, RepairFactor):
            for entity in list(graph.entities(kind)):
                if entity.level < p.fade_thr:
                    graph.remove_entity(entity.id)
                    events.append(
                        Event(
                            graph.time,
                            EventKind.FADE,
                            f"{entity.id} desapareceu do tecido",
                            (entity.id,),
                        )
                    )

    @staticmethod
    def _ensure(graph: ImmuneGraph, kind: type, entity_id: str):
        if not graph.has_entity(entity_id):
            graph.add_entity(kind(entity_id, level=0.0))
        return graph.entity(entity_id)
