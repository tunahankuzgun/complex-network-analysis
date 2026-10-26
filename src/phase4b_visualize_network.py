"""
Phase 4 (addendum) - Network visualization.

Draws the song network with:
  - edges colored by similarity weight (darker/warmer = more alike)
  - a slightly more spread-out spring layout, so nodes sit less on top of
    each other
  - the same graph under four layouts, so the picture is not an artifact
    of one algorithm
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

GENRE_COLORS = {
    "pop": "#e41a1c",
    "rap": "#377eb8",
    "rock": "#4daf4a",
    "latin": "#984ea3",
    "r&b": "#ff7f00",
    "edm": "#a65628",
}

SEED = 6


def load_graph(path: str = "data/song_network.gexf") -> nx.Graph:
    return nx.read_gexf(path)


def giant_component(G: nx.Graph) -> nx.Graph:
    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes).copy()


def edge_weights(G: nx.Graph) -> np.ndarray:
    return np.array([float(data.get("weight", 0.0)) for _, _, data in G.edges(data=True)])


def draw_on_ax(G: nx.Graph, pos, ax, title: str, node_scale: float = 3.0):
    weights = edge_weights(G)
    degrees = dict(G.degree())
    node_sizes = [3 + node_scale * degrees[n] for n in G.nodes()]
    node_colors = [GENRE_COLORS.get(G.nodes[n].get("playlist_genre"), "grey") for n in G.nodes()]

    collection = nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=weights,
        edge_cmap=plt.cm.YlOrRd,
        edge_vmin=float(weights.min()),
        edge_vmax=float(weights.max()),
        alpha=0.45,
        width=0.45,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.9, linewidths=0
    )
    ax.set_title(title)
    ax.axis("off")
    return collection


def layout_positions(G: nx.Graph) -> dict:
    """
    Four layouts. Kamada–Kawai and spectral want a connected graph, so this
    is meant to be called on the giant component.
    """
    return {
        "Spring (spread)": nx.spring_layout(G, seed=SEED, k=0.62, iterations=70),
        "Kamada–Kawai": nx.kamada_kawai_layout(G),
        "Spectral": nx.spectral_layout(G),
        "Circular": nx.circular_layout(G),
    }


def draw_network(G: nx.Graph, path: str = "figures/network_overview.png"):
    # k=0.62 (was 0.25) pushes neighbors apart a bit without blowing the
    # figure apart. Smaller nodes do the rest of the overlap fix.
    pos = nx.spring_layout(G, seed=SEED, k=0.62, iterations=70)

    fig, ax = plt.subplots(figsize=(12, 12))
    collection = draw_on_ax(
        G,
        pos,
        ax,
        f"Song similarity network (n={G.number_of_nodes()}, m={G.number_of_edges()})\n"
        "node size = degree, node color = genre, edge color = similarity",
        node_scale=4.0,
    )
    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GENRE_COLORS.items()]
    ax.legend(handles=legend_handles, title="Genre", loc="upper right")
    if collection is not None:
        fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.02, label="Edge similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def draw_layouts(G: nx.Graph, path: str = "figures/layout_comparison.png"):
    giant = giant_component(G)
    positions = layout_positions(giant)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    collections = []
    for ax, (name, pos) in zip(axes.ravel(), positions.items()):
        collections.append(draw_on_ax(giant, pos, ax, name, node_scale=2.2))

    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GENRE_COLORS.items()]
    fig.legend(handles=legend_handles, title="Genre", loc="upper right")
    fig.suptitle(
        f"Same giant component, four layouts (n={giant.number_of_nodes()})\n"
        "edge color = similarity weight"
    )
    # One colorbar is enough; every panel uses the same weight range.
    if collections[0] is not None:
        fig.colorbar(collections[0], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Edge similarity")
    fig.savefig(path, dpi=140)
    print(f"Saved {path}")


if __name__ == "__main__":
    G = load_graph()
    draw_network(G)
    draw_layouts(G)
