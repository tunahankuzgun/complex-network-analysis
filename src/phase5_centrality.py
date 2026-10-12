"""
Phase 5 - Centrality analysis.

Computes standard centrality measures to find:
  (a) which songs are most "central" in the similarity network, and what
      that means here (NOT popularity - centrality is purely structural)
  (b) whether structural centrality correlates with track_popularity (a
      real-world, non-network metric) - are "typical/generic-sounding"
      songs more popular, or are outliers?

Note on weights: edges carry two attributes, 'weight' (similarity, higher =
more alike) and 'distance' (euclidean distance, lower = more alike).
  - For path-based measures (betweenness, closeness), lower cost should
    mean "closer", so 'distance' is used as the edge cost.
  - For measures based on strength of connection (eigenvector, PageRank),
    higher should mean "more strongly connected", so 'weight' is used.
"""

import networkx as nx
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


def load_graph(path: str = "data/song_network.gexf") -> nx.Graph:
    return nx.read_gexf(path)


def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    degree_c = nx.degree_centrality(G)
    strength = dict(G.degree(weight="weight"))  # weighted degree ("strength")
    betweenness_c = nx.betweenness_centrality(G, weight="distance", normalized=True)
    closeness_c = nx.closeness_centrality(G, distance="distance")
    eigenvector_c = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    pagerank_c = nx.pagerank(G, weight="weight")

    rows = []
    for n in G.nodes():
        data = G.nodes[n]
        rows.append(
            {
                "track_id": n,
                "track_name": data.get("track_name"),
                "track_artist": data.get("track_artist"),
                "playlist_genre": data.get("playlist_genre"),
                "track_popularity": data.get("track_popularity"),
                "degree_centrality": degree_c[n],
                "strength": strength[n],
                "betweenness_centrality": betweenness_c[n],
                "closeness_centrality": closeness_c[n],
                "eigenvector_centrality": eigenvector_c[n],
                "pagerank": pagerank_c[n],
            }
        )
    return pd.DataFrame(rows)


def print_top_n(df: pd.DataFrame, metric: str, n: int = 10):
    print(f"\n--- Top {n} by {metric} ---")
    top = df.sort_values(metric, ascending=False).head(n)
    for _, row in top.iterrows():
        print(
            f"{row[metric]:.4f}  {row['track_name']} - {row['track_artist']} "
            f"[{row['playlist_genre']}] (popularity={row['track_popularity']})"
        )


def correlation_with_popularity(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        pear_r, pear_p = pearsonr(df[metric], df["track_popularity"])
        spear_r, spear_p = spearmanr(df[metric], df["track_popularity"])
        rows.append(
            {
                "metric": metric,
                "pearson_r": pear_r,
                "pearson_p": pear_p,
                "spearman_r": spear_r,
                "spearman_p": spear_p,
            }
        )
    return pd.DataFrame(rows)


def plot_centrality_vs_popularity(df: pd.DataFrame, metric: str, path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[metric], df["track_popularity"], alpha=0.35, s=15, color="teal")
    ax.set_xlabel(metric)
    ax.set_ylabel("track_popularity")
    ax.set_title(f"{metric} vs. track_popularity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


METRICS = [
    "degree_centrality",
    "strength",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
    "pagerank",
]


if __name__ == "__main__":
    G = load_graph()
    df = compute_centralities(G)
    df.to_csv("data/centrality_scores.csv", index=False)

    for metric in METRICS:
        print_top_n(df, metric, n=10)

    print("\n=== Correlation with track_popularity ===")
    corr_df = correlation_with_popularity(df, METRICS)
    print(corr_df.to_string(index=False))

    best_metric = corr_df.iloc[corr_df["pearson_r"].abs().idxmax()]["metric"]
    plot_centrality_vs_popularity(df, best_metric, f"figures/{best_metric}_vs_popularity.png")
