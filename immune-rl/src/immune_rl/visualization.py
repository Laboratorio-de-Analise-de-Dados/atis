"""Visualização da evolução do grafo imunológico.

Três modos:
  - plot_snapshots(history, times)   -> figura com os tempos escolhidos lado a lado
  - animate(history)                 -> animação (janela ou arquivo .gif/.mp4)
  - interactive_viewer(history)      -> janela com slider (e setas ← → do teclado)

Depende apenas de: graph.nx (DiGraph), entity.node_type, entity.signal() e das
arestas com atributos 'type' (EdgeType) e 'strength'.
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
from matplotlib.widgets import Slider

from .entities import Cell, EdgeType, NodeType


@dataclass
class GraphStyle:
    """Aparência do grafo. Altere aqui para mudar cores, tamanhos e rótulos."""
    node_colors: dict = field(default_factory=lambda: {
        NodeType.PAMP: "#E6A817",
        NodeType.PATHOGEN: "#C0392B",
        NodeType.MACROPHAGE: "#2E86C1",
        NodeType.CYTOKINE: "#27AE60",
    })
    edge_colors: dict = field(default_factory=lambda: {
        EdgeType.RECOGNITION: "#B7950B",
        EdgeType.ACTIVATION: "#922B21",
        EdgeType.PRODUCTION: "#1E8449",
    })
    node_labels: dict = field(default_factory=lambda: {
        NodeType.PAMP: "PAMP",
        NodeType.PATHOGEN: "patógeno",
        NodeType.MACROPHAGE: "macrófago",
        NodeType.CYTOKINE: "citocina",
    })
    edge_labels: dict = field(default_factory=lambda: {
        EdgeType.RECOGNITION: "reconhecimento",
        EdgeType.ACTIVATION: "ativação",
        EdgeType.PRODUCTION: "produção",
    })
    default_color: str = "#888888"
    min_size: float = 700.0
    max_size: float = 2800.0
    edge_width_base: float = 0.8
    edge_width_scale: float = 3.0
    arc: float = 0.12            # curvatura das arestas (separa A→B de B→A)
    show_values: bool = True


# ----------------------------------------------------------------- utilitários
# Em qual COLUNA cada tipo de nó fica no layout "layered" (edite para reorganizar).
LAYER_ORDER = {
    NodeType.PAMP: 0,
    NodeType.MACROPHAGE: 1,
    NodeType.PATHOGEN: 2,
    NodeType.CYTOKINE: 2,
}


def compute_layout(history, layout: str = "layered", seed: int = 0) -> dict:
    """Posições FIXAS para todos os tempos (os nós não 'pulam' entre quadros).

    Usa a união de todos os grafos do histórico, então nós/arestas que só
    aparecem depois (recrutamento, novas relações) já têm lugar reservado.

    layouts: 'layered' (colunas por tipo, determinístico; padrão),
             'spring' (forças; nós simétricos podem se sobrepor),
             'circular', 'shell'.
    """
    union = nx.DiGraph()
    for g in history:
        union.add_nodes_from(g.nx.nodes)
        union.add_edges_from(g.nx.edges)
        for n, d in g.nx.nodes(data=True):
            union.nodes[n]["layer"] = LAYER_ORDER.get(d["entity"].node_type, 1)
    if layout == "layered":
        return nx.multipartite_layout(union, subset_key="layer", align="vertical")
    if layout == "spring":
        return nx.spring_layout(union, seed=seed, iterations=300)
    if layout == "circular":
        return nx.circular_layout(union)
    if layout == "shell":
        return nx.shell_layout(union)
    raise ValueError(f"layout desconhecido: {layout!r} (use 'layered', 'spring', 'circular' ou 'shell')")


def _vmax_by_type(history) -> dict:
    """Maior sinal de cada tipo ao longo do tempo: define a escala dos tamanhos."""
    vmax: dict = {}
    for g in history:
        for _, d in g.nx.nodes(data=True):
            e = d["entity"]
            vmax[e.node_type] = max(vmax.get(e.node_type, 1e-9), e.signal())
    return vmax


def _limits(pos: dict, pad: float = 0.3, top_extra: float = 0.15):
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad + top_extra)


def _draw_self_loops(ax, pos, loops, sizes, style):
    """Laços (arestas de um nó para ele mesmo, ex.: sinalização autócrina).

    Desenhados acima do nó, com tamanho em PONTOS relativo ao raio do nó, para não
    depender da escala dos dados (o laço do networkx cresce com o tamanho do nó e
    sai da figura).
    """
    fig = ax.figure
    for node, etype, strength in loops:
        radius = math.sqrt(sizes[node]) / 2   # raio do nó em pontos

        def at(angle_deg):
            a = math.radians(angle_deg)
            return offset_copy(ax.transData, fig=fig, units="points",
                               x=radius * math.cos(a), y=radius * math.sin(a))

        ax.annotate(
            "", xy=pos[node], xycoords=at(60), xytext=pos[node], textcoords=at(120),
            arrowprops=dict(
                arrowstyle="-|>", connectionstyle="arc3,rad=-2.0", shrinkA=0, shrinkB=0,
                mutation_scale=16, color=style.edge_colors.get(etype, style.default_color),
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


def _summary(graph, style: GraphStyle) -> str:
    groups = defaultdict(list)
    for _, d in graph.nx.nodes(data=True):
        groups[d["entity"].node_type].append(d["entity"])
    parts = []
    for nt, ents in groups.items():
        name = style.node_labels.get(nt, nt.value)
        if isinstance(ents[0], Cell):
            parts.append(f"{name} (média) {sum(e.signal() for e in ents) / len(ents):.2f}")
        else:
            parts.append(f"{name} {sum(e.signal() for e in ents):.2f}")
    return " · ".join(parts)


def _legend_handles(history, style: GraphStyle) -> list:
    kinds, etypes = {}, []
    for g in history:
        for _, d in g.nx.nodes(data=True):
            kinds[d["entity"].node_type] = isinstance(d["entity"], Cell)
        for *_, d in g.nx.edges(data=True):
            if d.get("type") not in etypes:
                etypes.append(d.get("type"))
    handles = []
    for nt, is_cell in kinds.items():
        handles.append(Line2D([], [], marker="o" if is_cell else "s", linestyle="",
                              markerfacecolor=style.node_colors.get(nt, style.default_color),
                              markeredgecolor="black", markersize=9,
                              label=style.node_labels.get(nt, nt.value)))
    for et in etypes:
        handles.append(Line2D([], [], color=style.edge_colors.get(et, style.default_color),
                              linewidth=2.5,
                              label=style.edge_labels.get(et, getattr(et, "value", str(et)))))
    return handles


# ---------------------------------------------------------------- desenho base
def draw_graph(graph, ax, pos, *, vmax_by_type=None, style=None, title=None, limits=None):
    """Desenha UM grafo em um eixo.

    Círculos = células, quadrados = substâncias. O TAMANHO do nó é proporcional
    ao seu sinal (ativação ou nível) relativo ao máximo daquele tipo no histórico.
    A LARGURA da aresta é proporcional ao seu 'strength'.
    """
    style = style or GraphStyle()
    vmax_by_type = vmax_by_type or _vmax_by_type([graph])
    G = graph.nx
    ax.clear()

    nodes = list(G.nodes)
    ents = {n: G.nodes[n]["entity"] for n in nodes}
    sizes = {}
    for n in nodes:
        frac = min(max(ents[n].signal() / vmax_by_type[ents[n].node_type], 0.0), 1.0)
        sizes[n] = style.min_size + (style.max_size - style.min_size) * frac
    size_list = [sizes[n] for n in nodes]

    # arestas (atrás dos nós), agrupadas por tipo para dar uma cor a cada
    by_type, loops = defaultdict(list), []
    for u, v, d in G.edges(data=True):
        if u == v:   # aresta autócrina: desenhada por _draw_self_loops
            loops.append((u, d.get("type"), d.get("strength", 0.5)))
        else:
            by_type[d.get("type")].append((u, v, d.get("strength", 0.5)))
    for et, edges in by_type.items():
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v) for u, v, _ in edges],
            width=[style.edge_width_base + style.edge_width_scale * s for *_, s in edges],
            edge_color=style.edge_colors.get(et, style.default_color),
            arrows=True, arrowstyle="-|>", arrowsize=16,
            connectionstyle=f"arc3,rad={style.arc}",
            nodelist=nodes, node_size=size_list,
        )

    # nós, agrupados por tipo (cor) e por célula/substância (forma)
    groups = defaultdict(list)
    for n in nodes:
        groups[ents[n].node_type].append(n)
    for nt, members in groups.items():
        shape = "o" if isinstance(ents[members[0]], Cell) else "s"
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=members, node_size=[sizes[n] for n in members],
            node_color=style.node_colors.get(nt, style.default_color),
            node_shape=shape, edgecolors="black", linewidths=1.0,
        )

    labels = {n: f"{n}\n{ents[n].signal():.2f}" if style.show_values else n for n in nodes}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax, font_size=8,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
    )

    _draw_self_loops(ax, pos, loops, sizes, style)

    xlim, ylim = limits or _limits(pos)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    head = title or ""
    ax.set_title(f"{head}\n{_summary(graph, style)}".strip(), fontsize=9)


# ------------------------------------------------------------------ 1) tempos
def plot_snapshots(history, times=None, *, ncols=3, layout="layered", seed=0, pos=None,
                   style=None, save=None):
    """Mostra os tempos escolhidos lado a lado.

    times: lista de índices, ex. [0, 3, 6, -1] (-1 = último). None = 6 igualmente espaçados.
    """
    style = style or GraphStyle()
    times = _normalize_times(times, len(history))
    pos = pos or compute_layout(history, layout, seed)
    vmax, limits = _vmax_by_type(history), _limits(pos)

    ncols = min(ncols, len(times))
    nrows = math.ceil(len(times) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.4 * nrows + 0.7), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, t in zip(axes.flat, times):
        draw_graph(history[t], ax, pos, vmax_by_type=vmax, style=style,
                   title=f"t = {t}", limits=limits)

    handles = _legend_handles(history, style)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------- 2) animação
def animate(history, *, interval=700, layout="layered", seed=0, pos=None, style=None,
            figsize=(7.5, 6.5), save=None, fps=None, repeat=True):
    """Anima G_0 ... G_T.

    interval: ms por quadro. save: caminho .gif (Pillow) ou .mp4 (exige ffmpeg).
    IMPORTANTE: guarde o retorno em uma variável (anim = animate(...)); se o objeto
    for descartado, a animação para.
    """
    style = style or GraphStyle()
    pos = pos or compute_layout(history, layout, seed)
    vmax, limits = _vmax_by_type(history), _limits(pos)
    last = len(history) - 1

    fig, ax = plt.subplots(figsize=figsize)
    handles = _legend_handles(history, style)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.1, top=0.9)

    def update(i):
        draw_graph(history[i], ax, pos, vmax_by_type=vmax, style=style,
                   title=f"t = {i} / {last}", limits=limits)
        return []

    anim = FuncAnimation(fig, update, frames=len(history), interval=interval,
                         repeat=repeat, blit=False)
    if save:
        _save_animation(anim, save, fps or 1000 / interval)
    return anim


def _save_animation(anim, path, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        anim.save(path, writer=PillowWriter(fps=fps))
    elif path.suffix.lower() == ".mp4":
        try:
            anim.save(path, writer="ffmpeg", fps=fps)
        except (RuntimeError, ValueError) as err:
            raise RuntimeError(f"Não foi possível gerar .mp4 (ffmpeg instalado?). Use .gif. ({err})")
    else:
        raise ValueError("Extensão não suportada: use .gif ou .mp4")


# ------------------------------------------------------------- 3) interativo
def interactive_viewer(history, *, layout="layered", seed=0, pos=None, style=None, figsize=(8, 7.2)):
    """Janela com slider de tempo (e teclas ← →). Retorna (fig, slider).

    Guarde os dois retornos: fig, slider = interactive_viewer(h); depois plt.show().
    """
    if len(history) < 2:
        raise ValueError("O histórico precisa ter pelo menos 2 grafos.")
    style = style or GraphStyle()
    pos = pos or compute_layout(history, layout, seed)
    vmax, limits = _vmax_by_type(history), _limits(pos)
    last = len(history) - 1

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.22, top=0.9)
    handles = _legend_handles(history, style)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.09),
               ncol=len(handles), frameon=False, fontsize=9)
    slider = Slider(fig.add_axes((0.2, 0.04, 0.6, 0.03)), "t", 0, last, valinit=0, valstep=1)

    def render(t):
        draw_graph(history[t], ax, pos, vmax_by_type=vmax, style=style,
                   title=f"t = {t} / {last}", limits=limits)

    def on_change(value):
        render(int(value))
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            slider.set_val(min(slider.val + 1, last))
        elif event.key == "left":
            slider.set_val(max(slider.val - 1, 0))

    slider.on_changed(on_change)
    fig.canvas.mpl_connect("key_press_event", on_key)
    render(0)
    return fig, slider