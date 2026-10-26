"""
Phase 6 - Community detection.

Detects communities in the network purely from its structure (the
algorithm doesn't know genre labels), then checks how well those
communities line up with the real playlist_genre labels. If they align
well, audio-feature similarity alone recovers genre structure; if not,
that's evidence genres aren't purely defined by these 9 features.
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from networkx.algorithms.community import louvain_communities, modularity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

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


def detect_communities(G: nx.Graph, seed: int = 6):
    communities = louvain_communities(G, weight="weight", seed=seed)
    communities = sorted(communities, key=len, reverse=True)
    mod = modularity(G, communities, weight="weight")
    return communities, mod


def assign_labels(G: nx.Graph, communities) -> pd.DataFrame:
    node_to_community = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = idx

    rows = []
    for n in G.nodes():
        rows.append(
            {
                "track_id": n,
                "track_name": G.nodes[n].get("track_name"),
                "track_artist": G.nodes[n].get("track_artist"),
                "playlist_genre": G.nodes[n].get("playlist_genre"),
                "playlist_subgenre": G.nodes[n].get("playlist_subgenre"),
                "community": node_to_community[n],
            }
        )
    return pd.DataFrame(rows)


def evaluate_against_genre(df: pd.DataFrame):
    nmi = normalized_mutual_info_score(df["playlist_genre"], df["community"])
    ari = adjusted_rand_score(df["playlist_genre"], df["community"])
    contingency = pd.crosstab(df["community"], df["playlist_genre"])
    return nmi, ari, contingency


def draw_comparison(G: nx.Graph, df: pd.DataFrame, path: str = "figures/community_vs_genre.png"):
    pos = nx.spring_layout(G, seed=6, k=0.55, iterations=60)

    node_order = list(G.nodes())
    community_of = dict(zip(df["track_id"], df["community"]))
    genre_of = dict(zip(df["track_id"], df["playlist_genre"]))

    n_communities = df["community"].nunique()
    cmap = plt.get_cmap("tab20", n_communities)
    community_colors = [cmap(community_of[n]) for n in node_order]
    genre_colors = [GENRE_COLORS.get(genre_of[n], "grey") for n in node_order]

    fig, axes = plt.subplots(1, 2, figsize=(20, 11))

    nx.draw_networkx_edges(G, pos, ax=axes[0], alpha=0.1, width=0.5, edge_color="grey")
    nx.draw_networkx_nodes(G, pos, ax=axes[0], node_size=20, node_color=community_colors, linewidths=0)
    axes[0].set_title(f"Detected communities (Louvain, {n_communities} communities)")
    axes[0].axis("off")

    nx.draw_networkx_edges(G, pos, ax=axes[1], alpha=0.1, width=0.5, edge_color="grey")
    nx.draw_networkx_nodes(G, pos, ax=axes[1], node_size=20, node_color=genre_colors, linewidths=0)
    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in GENRE_COLORS.items()]
    axes[1].legend(handles=legend_handles, title="Genre", loc="upper right")
    axes[1].set_title("True genre labels")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    G = load_graph()
    communities, mod = detect_communities(G)
    print(f"Number of communities detected: {len(communities)}")
    print(f"Modularity: {mod:.4f}")
    print("Community sizes:", [len(c) for c in communities])

    df = assign_labels(G, communities)
    df.to_csv("data/community_assignments.csv", index=False)

    nmi, ari, contingency = evaluate_against_genre(df)
    print(f"\nNormalized Mutual Information (community vs genre): {nmi:.4f}")
    print(f"Adjusted Rand Index (community vs genre): {ari:.4f}")

    print("\n=== Contingency table (rows=community, columns=genre) ===")
    print(contingency)

    print("\n=== Dominant genre per community ===")
    for comm_id in sorted(df["community"].unique()):
        sub = df[df["community"] == comm_id]
        top_genre = sub["playlist_genre"].value_counts().idxmax()
        top_share = sub["playlist_genre"].value_counts(normalize=True).max()
        print(f"Community {comm_id} (n={len(sub)}): dominant genre = {top_genre} ({top_share:.1%})")

    draw_comparison(G, df)
