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

import sys

import networkx as nx
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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
                "track_popularity": float(data.get("track_popularity")),
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


def plot_all_centrality_figures(df: pd.DataFrame, metrics: list):
    """Histogram of every centrality, and each one against track_popularity."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        ax.hist(df[metric], bins=40, color="steelblue", edgecolor="white")
        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel("Songs")
    fig.suptitle("Centrality distributions")
    fig.tight_layout()
    fig.savefig("figures/centrality_histograms.png", dpi=150)
    print("Saved figures/centrality_histograms.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        ax.scatter(df[metric], df["track_popularity"], alpha=0.3, s=10, color="teal")
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_ylabel("track popularity")
    fig.suptitle("Every centrality vs track popularity")
    fig.tight_layout()
    fig.savefig("figures/centrality_vs_popularity.png", dpi=150)
    print("Saved figures/centrality_vs_popularity.png")


def _heatmap(matrix: pd.DataFrame, path: str, title: str):
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [c.replace("_", "\n") for c in matrix.columns]
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.columns)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def correlation_tables(df: pd.DataFrame, metrics: list):
    """
    Numeric correlations among centralities and popularity, plus a
    genre breakdown (genre is a label, so it is a table of means and of
    within-genre correlations, not a column in the Pearson matrix).
    """
    cols = metrics + ["track_popularity"]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")

    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    pearson.to_csv("data/centrality_correlation_pearson.csv")
    spearman.to_csv("data/centrality_correlation_spearman.csv")
    print("\n=== Pearson correlation (centralities + popularity) ===")
    print(pearson.round(3).to_string())
    print("\n=== Spearman correlation (centralities + popularity) ===")
    print(spearman.round(3).to_string())
    _heatmap(pearson, "figures/centrality_correlation_pearson.png", "Pearson correlation")
    _heatmap(spearman, "figures/centrality_correlation_spearman.png", "Spearman correlation")

    by_genre = df.groupby("playlist_genre")[cols].mean().sort_index()
    by_genre.to_csv("data/centrality_by_genre.csv")
    print("\n=== Mean centrality by genre ===")
    print(by_genre.round(4).to_string())

    # Within-genre correlation with popularity. Same question as the global
    # correlation, asked separately inside each genre.
    rows = []
    for genre, sub in df.groupby("playlist_genre"):
        row = {"playlist_genre": genre, "n": len(sub)}
        for metric in metrics:
            r, p = pearsonr(sub[metric], sub["track_popularity"])
            row[f"{metric}_r"] = r
            row[f"{metric}_p"] = p
        rows.append(row)
    genre_corr = pd.DataFrame(rows).sort_values("playlist_genre")
    genre_corr.to_csv("data/centrality_popularity_corr_by_genre.csv", index=False)
    print("\n=== Pearson r with track_popularity, by genre ===")
    r_cols = ["playlist_genre", "n"] + [c for c in genre_corr.columns if c.endswith("_r")]
    print(genre_corr[r_cols].round(3).to_string(index=False))


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
    plot_all_centrality_figures(df, METRICS)
    correlation_tables(df, METRICS)
