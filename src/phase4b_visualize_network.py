"""
Phase 4 (addendum) - Basic network visualization.

Produces a node-link picture of the network (the "hairball"), colored by
genre and sized by degree, so there's a visual sense of the graph's shape
before community detection in Phase 6 (which makes a second, more
informative version colored by detected community vs. true genre).
"""

import networkx as nx
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


def load_graph(path: str = "data/song_network.gexf") -> nx.Graph:
    return nx.read_gexf(path)


def draw_network(G: nx.Graph, path: str = "figures/network_overview.png"):
    pos = nx.spring_layout(G, seed=42, k=0.25, iterations=50)

    degrees = dict(G.degree())
    node_sizes = [10 + 8 * degrees[n] for n in G.nodes()]
    node_colors = [GENRE_COLORS.get(G.nodes[n].get("playlist_genre"), "grey") for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.5, edge_color="grey")
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.85, linewidths=0
    )

    legend_handles = [
        mpatches.Patch(color=c, label=g) for g, c in GENRE_COLORS.items()
    ]
    ax.legend(handles=legend_handles, title="Genre", loc="upper right")
    ax.set_title(f"Song similarity network (n={G.number_of_nodes()}, m={G.number_of_edges()})\nnode size = degree, color = genre")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    G = load_graph()
    draw_network(G)
