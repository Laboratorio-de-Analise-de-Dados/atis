"""Visualização da evolução do grafo imunológico.

Três modos:
  - plot_snapshots(history, times)  -> figura com os tempos escolhidos lado a lado
  - animate(history)                -> animação (janela ou arquivo .gif/.mp4)
  - interactive_viewer(history)     -> janela com slider, play/pause e eventos

Todas as funções aceitam ``events_history`` para narrar o que aconteceu em
cada passo. O layout é fixo ao longo do tempo (união de todos os grafos),
então os nós não mudam de lugar entre quadros.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy
from matplotlib.widgets import Button, Slider

from .entities import (
    AntiCytokine,
    Cell,
    Chemokine,
    EdgeType,
    NodeType,
    Pathogen,
    Phenotype,
    ProCytokine,
    RepairFactor,
    TissueDamage,
)
from .events import Event, EventKind, EventSeverity


@dataclass
class GraphStyle:
    """Aparência do grafo. Altere aqui para mudar cores, formas e rótulos."""

    node_colors: dict = field(
        default_factory=lambda: {
            NodeType.PAMP: "#D4A017",
            NodeType.PATHOGEN: "#C0392B",
            NodeType.MACROPHAGE: "#2E86C1",
            NodeType.PRO_CYTOKINE: "#E67E22",
            NodeType.ANTI_CYTOKINE: "#16A085",
            NodeType.CHEMOKINE: "#8E44AD",
            NodeType.REPAIR_FACTOR: "#27AE60",
            NodeType.DAMAGE: "#A04000",
        }
    )
    phenotype_colors: dict = field(
        default_factory=lambda: {
            Phenotype.M0: "#85C1E9",
            Phenotype.M1: "#E74C3C",
            Phenotype.M2: "#27AE60",
        }
    )
    edge_colors: dict = field(
        default_factory=lambda: {
            EdgeType.RECOGNITION: "#B7950B",
            EdgeType.ACTIVATION: "#C0392B",
            EdgeType.PRODUCTION: "#E67E22",
            EdgeType.RECRUITMENT: "#8E44AD",
            EdgeType.INTERACTION: "#922B21",
            EdgeType.REGULATION: "#16A085",
            EdgeType.REPAIR: "#27AE60",
        }
    )
    node_labels: dict = field(
        default_factory=lambda: {
            NodeType.PAMP: "PAMP",
            NodeType.PATHOGEN: "patógeno",
            NodeType.MACROPHAGE: "macrófago",
            NodeType.PRO_CYTOKINE: "citocina pró",
            NodeType.ANTI_CYTOKINE: "citocina anti",
            NodeType.CHEMOKINE: "quimiocina",
            NodeType.REPAIR_FACTOR: "fator de reparo",
            NodeType.DAMAGE: "dano tecidual",
        }
    )
    edge_labels: dict = field(
        default_factory=lambda: {
            EdgeType.RECOGNITION: "reconhecimento",
            EdgeType.ACTIVATION: "ativação",
            EdgeType.PRODUCTION: "produção",
            EdgeType.RECRUITMENT: "recrutamento",
            EdgeType.INTERACTION: "interação",
            EdgeType.REGULATION: "regulação",
            EdgeType.REPAIR: "reparo",
        }
    )
    short_labels: dict = field(
        default_factory=lambda: {
            NodeType.PAMP: "PAMP",
            NodeType.PATHOGEN: "P",
            NodeType.PRO_CYTOKINE: "pró",
            NodeType.ANTI_CYTOKINE: "anti",
            NodeType.CHEMOKINE: "quimio",
            NodeType.REPAIR_FACTOR: "reparo",
            NodeType.DAMAGE: "dano",
        }
    )
    edge_styles: dict = field(
        default_factory=lambda: {
            EdgeType.RECOGNITION: "--",
            EdgeType.ACTIVATION: "-",
            EdgeType.PRODUCTION: "-",
            EdgeType.RECRUITMENT: ":",
            EdgeType.INTERACTION: "-",
            EdgeType.REGULATION: "--",
            EdgeType.REPAIR: "-.",
        }
    )
    node_shapes: dict = field(
        default_factory=lambda: {
            NodeType.PAMP: "s",
            NodeType.PATHOGEN: "^",
            NodeType.MACROPHAGE: "o",
            NodeType.PRO_CYTOKINE: "s",
            NodeType.ANTI_CYTOKINE: "s",
            NodeType.CHEMOKINE: "D",
            NodeType.REPAIR_FACTOR: "s",
            NodeType.DAMAGE: "h",
        }
    )
    default_color: str = "#95A5A6"
    highlight_color: str = "#F1C40F"
    min_size: float = 900.0
    max_size: float = 3000.0
    edge_width_base: float = 0.8
    edge_width_scale: float = 2.5
    arc: float = 0.10
    show_values: bool = True


LAYER_ORDER = {
    NodeType.PAMP: 0,
    NodeType.PATHOGEN: 0,
    NodeType.MACROPHAGE: 1,
    NodeType.PRO_CYTOKINE: 2,
    NodeType.ANTI_CYTOKINE: 2,
    NodeType.CHEMOKINE: 2,
    NodeType.REPAIR_FACTOR: 2,
    NodeType.DAMAGE: 3,
}


# ------------------------------------------------------------------ layout
def compute_layout(history, layout: str = "layered", seed: int = 0) -> dict:
    """Posições FIXAS para todos os tempos, usando a união dos grafos."""
    union = nx.DiGraph()
    for graph in history:
        union.add_nodes_from(graph.nx.nodes)
        union.add_edges_from(graph.nx.edges)
        for node, data in graph.nx.nodes(data=True):
            union.nodes[node]["layer"] = LAYER_ORDER.get(data["entity"].node_type, 1)

    if layout == "layered":
        return _layered_layout(union)
    if layout == "spring":
        return nx.spring_layout(union, seed=seed, iterations=300)
    if layout == "circular":
        return nx.circular_layout(union)
    if layout == "shell":
        return nx.shell_layout(union)
    raise ValueError(
        f"layout desconhecido: {layout!r} (use 'layered', 'spring', 'circular' ou 'shell')"
    )


def _layered_layout(union: nx.DiGraph) -> dict:
    layers: dict[int, list[str]] = defaultdict(list)
    for node, data in union.nodes(data=True):
        layers[data.get("layer", 1)].append(node)

    positions = {}
    for layer, nodes in sorted(layers.items()):
        ordered = sorted(nodes)
        count = len(ordered)
        for index, node in enumerate(ordered):
            y = (count - 1) / 2 - index
            positions[node] = (layer * 1.4, y * 1.15)
    return positions


def _limits(pos: dict, pad: float = 0.35, top_extra: float = 0.18):
    if not pos:
        return (-1.0, 1.0), (-1.0, 1.0)
    xs = [point[0] for point in pos.values()]
    ys = [point[1] for point in pos.values()]
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad + top_extra)


def _vmax_by_type(history) -> dict:
    vmax: dict = {}
    for graph in history:
        for _, data in graph.nx.nodes(data=True):
            entity = data["entity"]
            vmax[entity.node_type] = max(vmax.get(entity.node_type, 1e-9), entity.signal())
    return vmax


# --------------------------------------------------------------- utilidades
def _node_color(entity, style: GraphStyle) -> str:
    if isinstance(entity, Cell):
        return style.phenotype_colors.get(
            entity.phenotype, style.node_colors.get(entity.node_type, style.default_color)
        )
    return style.node_colors.get(entity.node_type, style.default_color)


def _node_shape(entity, style: GraphStyle) -> str:
    return style.node_shapes.get(entity.node_type, "o")


def _node_label(entity, style: GraphStyle) -> str:
    if isinstance(entity, Cell):
        return f"{entity.id}\n{entity.phenotype.value} {entity.signal():.2f}"
    if isinstance(entity, TissueDamage):
        return f"dano\n{entity.signal():.2f}"
    return f"{entity.id}\n{entity.signal():.2f}"


def _summary(graph, style: GraphStyle) -> str:
    cells = graph.cells()
    parts = []
    if cells:
        m1 = sum(1 for cell in cells if cell.phenotype == Phenotype.M1)
        m2 = sum(1 for cell in cells if cell.phenotype == Phenotype.M2)
        parts.append(f"cél {len(cells)} · M1 {m1} · M2 {m2}")

    signals = []
    for kind in (Pathogen, ProCytokine, AntiCytokine, Chemokine, RepairFactor, TissueDamage):
        total = graph.total_signal(kind)
        if total > 1e-9 or kind is TissueDamage:
            label = style.short_labels.get(kind.node_type, kind.node_type.value)
            signals.append(f"{label} {total:.2f}")
    if signals:
        parts.append(" · ".join(signals))
    return "\n".join(parts)


_HEADLINE_PRIORITY = (
    EventKind.PATHOGEN_ERADICATED,
    EventKind.RECRUITMENT,
    EventKind.RECOGNITION,
    EventKind.PATHOGEN_ARRIVAL,
    EventKind.PAMP_ARRIVAL,
    EventKind.KILLING,
    EventKind.DAMAGE,
    EventKind.REPAIR,
    EventKind.SECRETION,
    EventKind.ACTION,
)


def _headline(events: list[Event] | None) -> str:
    if not events:
        return "sem eventos relevantes"
    for kind in _HEADLINE_PRIORITY:
        for event in events:
            if event.kind == kind:
                return event.message
    return events[0].message


def _event_text(events: list[Event] | None, max_events: int = 5) -> str:
    if not events:
        return "sem eventos relevantes"
    lines = []
    for event in events[:max_events]:
        prefix = "!" if event.severity != EventSeverity.INFO else "-"
        lines.append(f"{prefix} {event.message}")
    remaining = len(events) - max_events
    if remaining > 0:
        lines.append(f"... +{remaining} eventos")
    return "\n".join(lines)


def _draw_self_loops(ax, pos, loops, sizes, style: GraphStyle) -> None:
    """Laços (arestas de um nó para ele mesmo), desenhados acima do nó."""
    figure = ax.figure
    for node, edge_type, strength in loops:
        radius = math.sqrt(sizes[node]) / 2

        def at(angle_deg):
            angle = math.radians(angle_deg)
            return offset_copy(
                ax.transData,
                fig=figure,
                units="points",
                x=radius * math.cos(angle),
                y=radius * math.sin(angle),
            )

        ax.annotate(
            "",
            xy=pos[node],
            xycoords=at(60),
            xytext=pos[node],
            textcoords=at(120),
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3,rad=-2.0",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=16,
                color=style.edge_colors.get(edge_type, style.default_color),
                lw=style.edge_width_base + style.edge_width_scale * strength,
            ),
            annotation_clip=False,
        )


def _normalize_times(times, n: int) -> list[int]:
    """Aceita None (6 tempos igualmente espaçados), índices e negativos (-1 = último)."""
    if times is None:
        k = min(6, n)
        return sorted({round(i * (n - 1) / max(k - 1, 1)) for i in range(k)})
    out = []
    for t in times:
        if not -n <= t < n:
            raise IndexError(f"t={t} fora do intervalo válido [0, {n - 1}]")
        out.append(t if t >= 0 else n + t)
    return out


# ---------------------------------------------------------------- desenho
def draw_graph(
    graph,
    ax,
    pos,
    *,
    vmax_by_type=None,
    style: GraphStyle | None = None,
    title: str | None = None,
    limits=None,
    events: list[Event] | None = None,
    max_events: int = 5,
    show_edge_labels: bool = False,
):
    """Desenha um grafo em um eixo, destacando os eventos do passo."""
    style = style or GraphStyle()
    vmax_by_type = vmax_by_type or _vmax_by_type([graph])
    network = graph.nx
    ax.clear()

    nodes = list(network.nodes)
    entities = {node: network.nodes[node]["entity"] for node in nodes}

    sizes = {}
    for node in nodes:
        entity = entities[node]
        maximum = max(vmax_by_type.get(entity.node_type, 1e-9), 1e-9)
        fraction = min(max(entity.signal() / maximum, 0.0), 1.0)
        sizes[node] = style.min_size + (style.max_size - style.min_size) * fraction
    size_list = [sizes[node] for node in nodes]

    # arestas, agrupadas por tipo
    by_type: dict = defaultdict(list)
    loops = []
    for source, target, data in network.edges(data=True):
        if source == target:
            loops.append((source, data.get("type"), data.get("strength", 0.5)))
        else:
            by_type[data.get("type")].append(
                (source, target, data.get("strength", 0.5))
            )

    for edge_type, edges in by_type.items():
        nx.draw_networkx_edges(
            network,
            pos,
            ax=ax,
            edgelist=[(source, target) for source, target, _ in edges],
            width=[
                style.edge_width_base + style.edge_width_scale * strength
                for *_, strength in edges
            ],
            edge_color=style.edge_colors.get(edge_type, style.default_color),
            style=style.edge_styles.get(edge_type, "-"),
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            connectionstyle=f"arc3,rad={style.arc}",
            nodelist=nodes,
            node_size=size_list,
            alpha=0.85,
        )

    if events:
        highlighted_edges = [
            edge
            for event in events
            for edge in event.edges
            if network.has_edge(*edge)
        ]
        if highlighted_edges:
            nx.draw_networkx_edges(
                network,
                pos,
                ax=ax,
                edgelist=highlighted_edges,
                width=4.0,
                edge_color=style.highlight_color,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=18,
                connectionstyle=f"arc3,rad={style.arc}",
                nodelist=nodes,
                node_size=size_list,
                alpha=0.9,
            )

    # nós, agrupados por forma e coloridos por tipo/fenótipo
    groups: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        groups[_node_shape(entities[node], style)].append(node)
    for shape, members in groups.items():
        nx.draw_networkx_nodes(
            network,
            pos,
            ax=ax,
            nodelist=members,
            node_size=[sizes[node] for node in members],
            node_color=[_node_color(entities[node], style) for node in members],
            node_shape=shape,
            edgecolors="black",
            linewidths=1.0,
        )

    if events:
        highlighted_nodes = [
            node for event in events for node in event.nodes if node in network
        ]
        highlighted_nodes = list(dict.fromkeys(highlighted_nodes))
        if highlighted_nodes:
            ax.scatter(
                [pos[node][0] for node in highlighted_nodes],
                [pos[node][1] for node in highlighted_nodes],
                s=[sizes[node] * 1.7 for node in highlighted_nodes],
                facecolors="none",
                edgecolors=style.highlight_color,
                linewidths=2.5,
                zorder=5,
            )

    labels = {
        node: _node_label(entities[node], style) if style.show_values else node
        for node in nodes
    }
    nx.draw_networkx_labels(
        network,
        pos,
        labels=labels,
        ax=ax,
        font_size=8,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.78),
    )

    if show_edge_labels:
        edge_labels = {
            (source, target): f"{data.get('strength', 0):.2f}"
            for source, target, data in network.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            network,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=6,
            label_pos=0.35,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
        )

    _draw_self_loops(ax, pos, loops, sizes, style)

    xlim, ylim = limits or _limits(pos)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()

    head = title or f"t = {graph.time}"
    ax.set_title(f"{head} · {_headline(events)}\n{_summary(graph, style)}", fontsize=8.0)

    if events is not None:
        ax.text(
            0.99,
            0.98,
            _event_text(events, max_events),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9),
            zorder=10,
        )


def _legend_handles(history, style: GraphStyle) -> list:
    node_types = {}
    edge_types = []
    phenotypes = set()
    for graph in history:
        for _, data in graph.nx.nodes(data=True):
            entity = data["entity"]
            node_types[entity.node_type] = True
            if isinstance(entity, Cell):
                phenotypes.add(entity.phenotype)
        for *_, data in graph.nx.edges(data=True):
            edge_type = data.get("type")
            if edge_type not in edge_types:
                edge_types.append(edge_type)

    handles = []
    for node_type in node_types:
        if node_type == NodeType.MACROPHAGE and phenotypes:
            continue
        handles.append(
            Line2D(
                [],
                [],
                marker=style.node_shapes.get(node_type, "o"),
                linestyle="",
                markerfacecolor=style.node_colors.get(node_type, style.default_color),
                markeredgecolor="black",
                markersize=9,
                label=style.node_labels.get(node_type, node_type.value),
            )
        )
    for phenotype in sorted(phenotypes, key=lambda item: item.value):
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor=style.phenotype_colors.get(phenotype, style.default_color),
                markeredgecolor="black",
                markersize=8,
                label=f"macrófago {phenotype.value}",
            )
        )
    for edge_type in edge_types:
        handles.append(
            Line2D(
                [],
                [],
                color=style.edge_colors.get(edge_type, style.default_color),
                linewidth=2.5,
                linestyle=style.edge_styles.get(edge_type, "-"),
                label=style.edge_labels.get(edge_type, getattr(edge_type, "value", str(edge_type))),
            )
        )
    return handles


def _draw_timeline(history, ax, *, current: int | None = None, style: GraphStyle | None = None):
    style = style or GraphStyle()
    steps = list(range(len(history)))
    pathogen = [graph.total_signal(Pathogen) for graph in history]
    pro = [graph.total_signal(ProCytokine) for graph in history]
    anti = [graph.total_signal(AntiCytokine) for graph in history]
    damage = [graph.total_signal(TissueDamage) for graph in history]
    m1 = [sum(1 for cell in graph.cells() if cell.phenotype == Phenotype.M1) for graph in history]
    m2 = [sum(1 for cell in graph.cells() if cell.phenotype == Phenotype.M2) for graph in history]

    ax.clear()
    ax.plot(steps, pathogen, color=style.node_colors[NodeType.PATHOGEN], label="patógeno", lw=2)
    ax.plot(steps, pro, color=style.node_colors[NodeType.PRO_CYTOKINE], label="citocina pró", lw=2)
    ax.plot(steps, anti, color=style.node_colors[NodeType.ANTI_CYTOKINE], label="citocina anti", lw=2)
    ax.plot(steps, damage, color=style.node_colors[NodeType.DAMAGE], label="dano", lw=2)
    ax.step(steps, m1, where="mid", color=style.phenotype_colors[Phenotype.M1], label="M1", lw=1.5)
    ax.step(steps, m2, where="mid", color=style.phenotype_colors[Phenotype.M2], label="M2", lw=1.5)

    ax.set_xlim(0, max(1, len(history) - 1))
    maximum = max(pathogen + pro + anti + damage + m1 + m2 + [1.0])
    ax.set_ylim(0, maximum * 1.15)
    ax.set_xlabel("t")
    ax.set_ylabel("sinal / contagem")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=7, loc="upper right", frameon=False)
    if current is not None:
        ax.axvline(current, color="black", linestyle="--", alpha=0.55, lw=1.5)


# ------------------------------------------------------------------ tempos
def plot_snapshots(
    history,
    times=None,
    *,
    events_history=None,
    ncols=3,
    layout="layered",
    seed=0,
    pos=None,
    style=None,
    save=None,
):
    """Mostra os tempos escolhidos lado a lado, com a narração de cada um."""
    style = style or GraphStyle()
    times = _normalize_times(times, len(history))
    pos = pos or compute_layout(history, layout, seed)
    vmax, limits = _vmax_by_type(history), _limits(pos)

    ncols = min(ncols, len(times))
    nrows = math.ceil(len(times) / ncols)
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows + 0.8), squeeze=False
    )
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, t in zip(axes.flat, times):
        events = events_history[t] if events_history and t < len(events_history) else None
        draw_graph(
            history[t],
            ax,
            pos,
            vmax_by_type=vmax,
            style=style,
            title=f"t = {t}",
            limits=limits,
            events=events,
            max_events=3,
        )

    handles = _legend_handles(history, style)
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=min(7, len(handles)),
        frameon=False,
        fontsize=8,
    )
    figure.suptitle("Evolução do grafo imunológico", fontsize=12)
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save, dpi=160, bbox_inches="tight")
    return figure


# ----------------------------------------------------------------- 2) animação
def animate(
    history,
    *,
    events_history=None,
    interval=700,
    layout="layered",
    seed=0,
    pos=None,
    style=None,
    figsize=(8.5, 8.5),
    save=None,
    fps=None,
    repeat=True,
):
    """Anima G_0 ... G_T, com linha do tempo e narração dos eventos."""
    style = style or GraphStyle()
    pos = pos or compute_layout(history, layout, seed)
    vmax, limits = _vmax_by_type(history), _limits(pos)
    last = len(history) - 1

    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.18)
    ax = figure.add_subplot(grid[0])
    ax_timeline = figure.add_subplot(grid[1])

    handles = _legend_handles(history, style)
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(7, len(handles)),
        frameon=False,
        fontsize=7,
    )
    figure.subplots_adjust(bottom=0.11, top=0.95, left=0.03, right=0.97)

    def update(index):
        events = events_history[index] if events_history and index < len(events_history) else None
        draw_graph(
            history[index],
            ax,
            pos,
            vmax_by_type=vmax,
            style=style,
            title=f"t = {index} / {last}",
            limits=limits,
            events=events,
            max_events=5,
        )
        _draw_timeline(history, ax_timeline, current=index, style=style)
        return []

    animation = FuncAnimation(
        figure,
        update,
        frames=len(history),
        interval=interval,
        repeat=repeat,
        blit=False,
    )
    if save:
        _save_animation(animation, save, fps or 1000 / interval)
    return animation


def _save_animation(animation, path, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        animation.save(path, writer=PillowWriter(fps=fps))
    elif path.suffix.lower() == ".mp4":
        try:
            animation.save(path, writer="ffmpeg", fps=fps)
        except (RuntimeError, ValueError) as err:
            raise RuntimeError(
                f"Não foi possível gerar .mp4 (ffmpeg instalado?). Use .gif. ({err})"
            )
    else:
        raise ValueError("Extensão não suportada: use .gif ou .mp4")


# ------------------------------------------------------------- 3) interativo
def interactive_viewer(
    history,
    *,
    events_history=None,
    layout="layered",
    seed=0,
    pos=None,
    style=None,
    figsize=(9.0, 9.0),
    interval=900,
):
    """Janela com slider, play/pause e painel de eventos. Retorna (fig, slider)."""
    if len(history) < 2:
        raise ValueError("O histórico precisa ter pelo menos 2 grafos.")

    style = style or GraphStyle()
    pos = pos or compute_layout(history, layout, seed)
    vmax, limits = _vmax_by_type(history), _limits(pos)
    last = len(history) - 1

    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(3, 1, height_ratios=[3.3, 0.9, 0.2], hspace=0.22)
    ax = figure.add_subplot(grid[0])
    ax_timeline = figure.add_subplot(grid[1])
    figure.add_subplot(grid[2]).set_axis_off()

    handles = _legend_handles(history, style)
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=min(7, len(handles)),
        frameon=False,
        fontsize=7,
    )

    slider = Slider(figure.add_axes((0.18, 0.045, 0.55, 0.028)), "t", 0, last, valinit=0, valstep=1)
    button = Button(figure.add_axes((0.78, 0.035, 0.12, 0.05)), "play")

    def render(index):
        events = events_history[index] if events_history and index < len(events_history) else None
        draw_graph(
            history[index],
            ax,
            pos,
            vmax_by_type=vmax,
            style=style,
            title=f"t = {index} / {last}",
            limits=limits,
            events=events,
            max_events=5,
        )
        _draw_timeline(history, ax_timeline, current=index, style=style)

    def on_change(value):
        render(int(value))
        figure.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            slider.set_val(min(slider.val + 1, last))
        elif event.key == "left":
            slider.set_val(max(slider.val - 1, 0))
        elif event.key == " ":
            toggle_play()

    state = {"playing": False}
    timer = figure.canvas.new_timer(interval=interval)

    def advance():
        if slider.val >= last:
            state["playing"] = False
            timer.stop()
            button.label.set_text("play")
            return
        slider.set_val(min(slider.val + 1, last))

    timer.add_callback(advance)

    def toggle_play(event=None):
        state["playing"] = not state["playing"]
        if state["playing"]:
            timer.start()
            button.label.set_text("pausar")
        else:
            timer.stop()
            button.label.set_text("play")
        figure.canvas.draw_idle()

    slider.on_changed(on_change)
    button.on_clicked(toggle_play)
    figure.canvas.mpl_connect("key_press_event", on_key)
    render(0)
    return figure, slider


def plot_timeseries(history, *, style=None, save=None):
    """Figura separada com a linha do tempo completa."""
    style = style or GraphStyle()
    figure, ax = plt.subplots(figsize=(9, 3))
    _draw_timeline(history, ax, current=None, style=style)
    ax.set_title("Sinais do episódio")
    figure.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save, dpi=160, bbox_inches="tight")
    return figure
