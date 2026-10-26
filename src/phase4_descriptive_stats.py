"""
Phase 4 - Descriptive network statistics.

Characterizes the basic structure of the song-similarity network: size,
degree distribution, clustering, connectivity, and small-world-ness.

Note on isolates: songs with no mutual top-k match stay in the graph.
They're included in degree/clustering statistics (degree=0, clustering=0),
but excluded from metrics that require connectivity (average shortest path
length, diameter, small-world comparison). Those are computed on the giant
connected component only.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = "figures"


def load_graph(path: str = "data/song_network.gexf") -> nx.Graph:
    return nx.read_gexf(path)


def basic_stats(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    giant = G.subgraph(components[0]).copy()

    stats = {
        "n_nodes": n,
        "n_edges": m,
        "density": density,
        "n_components": len(components),
        "giant_component_size": len(components[0]),
        "giant_component_fraction": len(components[0]) / n,
        "average_clustering_full": nx.average_clustering(G),
        "transitivity_full": nx.transitivity(G),
    }
    return stats, giant


def degree_distribution_plot(G: nx.Graph, path: str = f"{FIGURES_DIR}/degree_distribution.png"):
    degrees = [d for _, d in G.degree()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(degrees, bins=range(0, max(degrees) + 2), color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Number of songs")
    axes[0].set_title("Degree distribution")

    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1
    xs = sorted(d for d in degree_counts if d > 0)
    ys = [degree_counts[d] for d in xs]
    axes[1].scatter(xs, ys, color="firebrick")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Degree (log)")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_title("Degree distribution (log-log)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def giant_component_path_stats(giant: nx.Graph) -> dict:
    avg_path_length = nx.average_shortest_path_length(giant, weight=None)
    diameter = nx.diameter(giant)
    avg_clustering = nx.average_clustering(giant)
    return {
        "giant_avg_path_length": avg_path_length,
        "giant_diameter": diameter,
        "giant_avg_clustering": avg_clustering,
    }


def small_world_comparison(giant: nx.Graph, n_random: int = 10) -> dict:
    """
    Compare the giant component against equivalent random graphs (same n,
    same number of edges) to compute the small-world coefficient sigma:

        sigma = (C / C_rand) / (L / L_rand)

    sigma >> 1 indicates small-world structure (high clustering relative to
    a random graph, without a correspondingly large increase in path length).

    Uses an Erdos-Renyi G(n, m) random graph with the same n and number of
    edges, averaged over n_random trials to get stable estimates of
    C_rand and L_rand. Only connected random graphs are used for L_rand
    (giant component of the random graph, like for our real network).
    """
    n = giant.number_of_nodes()
    m = giant.number_of_edges()

    C_rands, L_rands = [], []
    for seed in range(n_random):
        R = nx.gnm_random_graph(n, m, seed=seed)
        R_components = sorted(nx.connected_components(R), key=len, reverse=True)
        R_giant = R.subgraph(R_components[0]).copy()
        C_rands.append(nx.average_clustering(R))
        L_rands.append(nx.average_shortest_path_length(R_giant))

    C_rand = np.mean(C_rands)
    L_rand = np.mean(L_rands)

    C = nx.average_clustering(giant)
    L = nx.average_shortest_path_length(giant)

    sigma = (C / C_rand) / (L / L_rand)

    return {
        "C_real": C,
        "L_real": L,
        "C_random": C_rand,
        "L_random": L_rand,
        "sigma": sigma,
    }


def plot_er_metric_comparison(sw_stats: dict, path: str = f"{FIGURES_DIR}/er_comparison_metrics.png"):
    """Bar chart of clustering and path length against an Erdős–Rényi graph."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    labels = ["Our network", "Erdős–Rényi"]
    colors = ["steelblue", "darkorange"]

    c_vals = [sw_stats["C_real"], sw_stats["C_random"]]
    l_vals = [sw_stats["L_real"], sw_stats["L_random"]]
    axes[0].bar(labels, c_vals, color=colors)
    axes[0].set_ylabel("Average clustering coefficient")
    axes[0].set_title("Clustering")
    for x, val in enumerate(c_vals):
        axes[0].text(x, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, l_vals, color=colors)
    axes[1].set_ylabel("Average path length")
    axes[1].set_title("Path length")
    for x, val in enumerate(l_vals):
        axes[1].text(x, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, max(c_vals) * 1.18)
    axes[1].set_ylim(0, max(l_vals) * 1.18)

    fig.suptitle(
        f"Giant component vs Erdős–Rényi G(n, m)   |   small-world σ = {sw_stats['sigma']:.1f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def plot_er_degree_overlay(
    G: nx.Graph, path: str = f"{FIGURES_DIR}/er_degree_distribution.png", n_random: int = 20, seed: int = 6
):
    """
    Overlay our degree distribution on Erdős–Rényi graphs with the same
    number of nodes and edges. The ER curve is the mean over n_random draws.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    real = np.bincount([d for _, d in G.degree()])

    er_counts = []
    max_deg = len(real) - 1
    for i in range(n_random):
        R = nx.gnm_random_graph(n, m, seed=seed + i)
        counts = np.bincount([d for _, d in R.degree()])
        max_deg = max(max_deg, len(counts) - 1)
        er_counts.append(counts)

    real = np.pad(real, (0, max_deg + 1 - len(real)))
    er_mat = np.vstack([np.pad(c, (0, max_deg + 1 - len(c))) for c in er_counts])
    er_mean = er_mat.mean(axis=0)

    xs = np.arange(max_deg + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(xs, real, color="steelblue", alpha=0.75, label="Our network")
    ax.plot(xs, er_mean, color="darkorange", marker="o", linewidth=2, label=f"Erdős–Rényi mean (n={n_random})")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of songs")
    ax.set_title("Degree distribution: our network vs Erdős–Rényi G(n, m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    G = load_graph()
    stats, giant = basic_stats(G)
    print("=== Basic stats (full graph) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    degree_distribution_plot(G)

    print("\n=== Path-based stats (giant component only) ===")
    path_stats = giant_component_path_stats(giant)
    for k, v in path_stats.items():
        print(f"{k}: {v}")

    print("\n=== Small-world comparison (giant component vs random graphs) ===")
    sw_stats = small_world_comparison(giant)
    for k, v in sw_stats.items():
        print(f"{k}: {v:.4f}")

    plot_er_metric_comparison(sw_stats)
    plot_er_degree_overlay(G)
