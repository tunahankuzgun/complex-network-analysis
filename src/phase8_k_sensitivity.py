"""
k-sensitivity and within-community analysis.

1. Build mutual kNN graphs for k = 5, 8, 10, 12, 15 and compare basic stats.
2. For three of those graphs (k = 5, 10, 15), run Louvain and describe what
   is going on inside the communities: size, density, clustering, and how
   mixed the genre labels are.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities, modularity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import phase3_build_network as net

K_VALUES = [5, 8, 10, 12, 15]
COMMUNITY_K_VALUES = [5, 10, 15]
SEED = 6
MIN_COMMUNITY_SIZE = 15


def basic_stats(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    giant = G.subgraph(components[0])
    return {
        "n_nodes": n,
        "n_edges": m,
        "density": nx.density(G),
        "mean_degree": float(np.mean(degrees)),
        "max_degree": int(max(degrees)),
        "n_isolates": sum(d == 0 for d in degrees),
        "n_components": len(components),
        "giant_fraction": len(components[0]) / n,
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "giant_avg_path_length": nx.average_shortest_path_length(giant),
    }


def community_rows(G: nx.Graph, k: int) -> tuple[dict, pd.DataFrame, list]:
    communities = louvain_communities(G, weight="weight", seed=SEED)
    communities = sorted(communities, key=len, reverse=True)
    mod = modularity(G, communities, weight="weight")

    labels = []
    genres = []
    for node in G.nodes():
        genres.append(G.nodes[node].get("playlist_genre"))
    node_to_comm = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = idx
    labels = [node_to_comm[node] for node in G.nodes()]

    summary = {
        "k": k,
        "n_communities": len(communities),
        "n_communities_size_ge_15": sum(len(c) >= MIN_COMMUNITY_SIZE for c in communities),
        "modularity": mod,
        "nmi_vs_genre": normalized_mutual_info_score(genres, labels),
        "ari_vs_genre": adjusted_rand_score(genres, labels),
    }

    rows = []
    for idx, comm in enumerate(communities):
        if len(comm) < MIN_COMMUNITY_SIZE:
            continue
        sub = G.subgraph(comm)
        genre_counts = pd.Series(
            [G.nodes[n].get("playlist_genre") for n in comm]
        ).value_counts(normalize=True)
        rows.append(
            {
                "k": k,
                "community": idx,
                "size": len(comm),
                "internal_edges": sub.number_of_edges(),
                "density": nx.density(sub),
                "avg_clustering": nx.average_clustering(sub),
                "dominant_genre": genre_counts.index[0],
                "dominant_genre_share": float(genre_counts.iloc[0]),
                "n_genres": int((genre_counts > 0).sum()),
            }
        )
    return summary, pd.DataFrame(rows), communities


def plot_stat_comparison(stats: pd.DataFrame, path: str = "figures/k_comparison_stats.png"):
    metrics = [
        ("n_edges", "Edges"),
        ("mean_degree", "Mean degree"),
        ("density", "Density"),
        ("avg_clustering", "Avg. clustering"),
        ("n_components", "Components"),
        ("giant_fraction", "Giant component fraction"),
        ("n_isolates", "Isolates"),
        ("giant_avg_path_length", "Avg. path length (giant)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.plot(stats["k"], stats[col], marker="o", color="steelblue")
        ax.set_xticks(stats["k"])
        ax.set_xlabel("k")
        ax.set_title(title)
    fig.suptitle("Mutual kNN graphs: basic stats at different k")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def plot_community_networks(graphs: dict, community_sets: dict, path: str = "figures/k_community_networks.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, k in zip(axes, COMMUNITY_K_VALUES):
        G = graphs[k]
        communities = community_sets[k]
        node_to_comm = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = idx
        # Only the larger communities get a color. Singletons stay grey.
        cmap = plt.cm.tab20
        colors = []
        for n in G.nodes():
            cid = node_to_comm[n]
            if len(communities[cid]) < MIN_COMMUNITY_SIZE:
                colors.append("#d0d0d0")
            else:
                colors.append(cmap(cid % 20))
        pos = nx.spring_layout(G, seed=SEED, k=0.55, iterations=40)
        weights = [float(data.get("weight", 0.0)) for _, _, data in G.edges(data=True)]
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color=weights, edge_cmap=plt.cm.YlOrRd,
            alpha=0.25, width=0.3,
        )
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=8, node_color=colors, linewidths=0)
        ax.set_title(f"k = {k}")
        ax.axis("off")
    fig.suptitle("Three kNN graphs, colored by Louvain community (grey = community < 15 songs)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"Saved {path}")


def plot_within_community(within: pd.DataFrame, path: str = "figures/within_community_stats.png"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    for ax, k in zip(axes, COMMUNITY_K_VALUES):
        sub = within[within["k"] == k].head(8).iloc[::-1]
        labels = [f"{int(r.community)}: {r.dominant_genre}" for r in sub.itertuples()]
        ax.barh(labels, sub["density"].values, color="steelblue")
        ax.set_title(f"k = {k}")
        ax.set_xlabel("Internal density")
    fig.suptitle("Largest communities: internal density (label = id: dominant genre)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    sample = pd.read_csv("data/sampled_tracks.csv")

    graphs = {}
    stat_rows = []
    for k in K_VALUES:
        print(f"Building k={k}...")
        G = net.build_knn_graph(sample, k=k, mutual=True)
        graphs[k] = G
        row = basic_stats(G)
        row["k"] = k
        stat_rows.append(row)
        print(
            f"  edges={row['n_edges']}  clustering={row['avg_clustering']:.3f}  "
            f"components={row['n_components']}  path={row['giant_avg_path_length']:.2f}"
        )

    stats = pd.DataFrame(stat_rows)[
        ["k", "n_nodes", "n_edges", "density", "mean_degree", "max_degree",
         "n_isolates", "n_components", "giant_fraction", "avg_clustering",
         "transitivity", "giant_avg_path_length"]
    ]
    stats.to_csv("data/k_comparison_stats.csv", index=False)
    print("\n=== Basic stats by k ===")
    print(stats.round(4).to_string(index=False))
    plot_stat_comparison(stats)

    summaries = []
    within_frames = []
    community_sets = {}
    for k in COMMUNITY_K_VALUES:
        print(f"\nCommunities for k={k}...")
        summary, within, communities = community_rows(graphs[k], k)
        summaries.append(summary)
        within_frames.append(within)
        community_sets[k] = communities
        print(
            f"  communities={summary['n_communities']}  modularity={summary['modularity']:.3f}  "
            f"NMI={summary['nmi_vs_genre']:.3f}  ARI={summary['ari_vs_genre']:.3f}"
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv("data/k_community_summary.csv", index=False)
    within_df = pd.concat(within_frames, ignore_index=True)
    within_df.to_csv("data/within_community_stats.csv", index=False)

    print("\n=== Community detection across k ===")
    print(summary_df.round(4).to_string(index=False))
    print("\n=== Within-community stats (size >= 15) ===")
    print(within_df.round(3).to_string(index=False))

    plot_community_networks(graphs, community_sets)
    plot_within_community(within_df)
