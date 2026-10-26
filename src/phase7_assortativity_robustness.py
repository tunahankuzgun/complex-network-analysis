"""
Phase 7 - Assortativity & robustness (stretch goals).

1. Assortativity: quantifies two things seen qualitatively in Phase 4/6 -
   (a) do high-degree songs tend to connect to other high-degree songs?
   (b) do same-genre songs preferentially connect to each other? (this
       directly quantifies the "genre mixing" seen in Phase 6)
2. Robustness/percolation: simulates removing songs from the network
   either randomly (simulates "random failure") or by removing the
   highest-degree song first each time (simulates a "targeted attack" on
   hub songs), and tracks how the giant connected component shrinks.
   Classic complex-network resilience analysis.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def load_graph(path: str = "data/song_network.gexf") -> nx.Graph:
    return nx.read_gexf(path)


def assortativity_report(G: nx.Graph) -> dict:
    degree_assort = nx.degree_assortativity_coefficient(G)
    genre_assort = nx.attribute_assortativity_coefficient(G, "playlist_genre")
    return {
        "degree_assortativity": degree_assort,
        "genre_assortativity": genre_assort,
    }


def targeted_attack(G: nx.Graph, checkpoint_step: int = 25, max_fraction: float = 0.9):
    """Repeatedly remove the current highest-degree node; recompute degree each time (adaptive attack)."""
    H = G.copy()
    n = G.number_of_nodes()
    max_removals = int(max_fraction * n)

    removed_fracs = [0.0]
    giant_fracs = [1.0]
    removed = 0
    while removed < max_removals and H.number_of_nodes() > 0:
        degrees = dict(H.degree())
        target = max(degrees, key=degrees.get)
        H.remove_node(target)
        removed += 1
        if removed % checkpoint_step == 0:
            components = nx.connected_components(H)
            giant_size = max((len(c) for c in components), default=0)
            removed_fracs.append(removed / n)
            giant_fracs.append(giant_size / n)
    return removed_fracs, giant_fracs


def random_failure(
    G: nx.Graph, checkpoint_step: int = 25, max_fraction: float = 0.9,
    n_trials: int = 10, seed: int = 6,
):
    """Remove nodes in random order, averaged over several trials."""
    n = G.number_of_nodes()
    max_removals = int(max_fraction * n)
    rng = np.random.default_rng(seed)

    all_giant_fracs = []
    removed_fracs_ref = None
    for trial in range(n_trials):
        H = G.copy()
        order = rng.permutation(list(G.nodes()))
        removed_fracs = [0.0]
        giant_fracs = [1.0]
        removed = 0
        for node in order[:max_removals]:
            H.remove_node(node)
            removed += 1
            if removed % checkpoint_step == 0:
                components = nx.connected_components(H)
                giant_size = max((len(c) for c in components), default=0)
                removed_fracs.append(removed / n)
                giant_fracs.append(giant_size / n)
        all_giant_fracs.append(giant_fracs)
        removed_fracs_ref = removed_fracs

    avg_giant_fracs = np.mean(all_giant_fracs, axis=0)
    return removed_fracs_ref, avg_giant_fracs


def plot_robustness(random_result, targeted_result, path: str = "figures/robustness.png"):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(random_result[0], random_result[1], label="Random failure (avg of 10 trials)", color="steelblue")
    ax.plot(targeted_result[0], targeted_result[1], label="Targeted attack (remove highest-degree first)", color="firebrick")
    ax.set_xlabel("Fraction of nodes removed")
    ax.set_ylabel("Giant component size (fraction of original N)")
    ax.set_title("Network robustness: random failure vs. targeted attack")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    G = load_graph()

    print("=== Assortativity ===")
    assort = assortativity_report(G)
    for k, v in assort.items():
        print(f"{k}: {v:.4f}")

    print("\nRunning targeted attack simulation...")
    targeted_result = targeted_attack(G)

    print("Running random failure simulation (10 trials)...")
    random_result = random_failure(G)

    plot_robustness(random_result, targeted_result)
