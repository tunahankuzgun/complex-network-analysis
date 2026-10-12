"""
Phase 3 - Build the song-similarity network.

Standardizes the 9 audio features, finds each song's k nearest neighbors,
and builds a weighted, undirected networkx graph.
"""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import phase1_load_and_clean as clean_module

K = 10  # number of nearest neighbors per song
MUTUAL = True  # True = mutual kNN (edge only if both songs list each other), False = union kNN


def standardize_features(df: pd.DataFrame) -> np.ndarray:
    """Z-score standardize the audio feature columns. Returns an (n, 9) array."""
    scaler = StandardScaler()
    return scaler.fit_transform(df[clean_module.FEATURE_COLUMNS].values)


def build_knn_graph(
    df: pd.DataFrame, k: int = K, mutual: bool = MUTUAL
) -> nx.Graph:
    """
    Build a k-NN similarity graph.

    - Nodes are track_ids, with metadata (name, artist, genre, subgenre,
      popularity) attached as node attributes.
    - Edge weight = similarity = 1 / (1 + euclidean_distance), so closer
      songs (smaller distance) get a weight closer to 1, and dissimilar
      ones approach 0. This keeps weights positive and bounded, which is
      convenient for weighted centrality/clustering metrics later.

    Self-matches are excluded explicitly rather than assumed to be the
    first returned neighbor: a few songs in this dataset are exact
    audio-feature duplicates (e.g. re-releases under a different
    track_id), which ties them at distance 0 and can displace the true
    self-match from position 0.
    """
    X = standardize_features(df)

    # n_neighbors=k+1 so there's still room for k real neighbors after
    # dropping the self-match.
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    distances, indices = nbrs.kneighbors(X)

    neighbors = []  # neighbors[i] = list of (j, distance), closest first, excluding self
    for i in range(len(df)):
        pairs = [(j, d) for j, d in zip(indices[i], distances[i]) if j != i]
        neighbors.append(pairs[:k])
    neighbor_sets = [{j for j, _ in pairs} for pairs in neighbors]

    G = nx.Graph()
    track_ids = df["track_id"].values
    for _, row in df.reset_index(drop=True).iterrows():
        G.add_node(
            row["track_id"],
            track_name=row["track_name"],
            track_artist=row["track_artist"],
            track_popularity=row["track_popularity"],
            playlist_genre=row["playlist_genre"],
            playlist_subgenre=row["playlist_subgenre"],
        )

    for i, pairs in enumerate(neighbors):
        for j, dist in pairs:
            if mutual and i not in neighbor_sets[j]:
                continue
            if G.has_edge(track_ids[i], track_ids[j]):
                continue
            G.add_edge(track_ids[i], track_ids[j], weight=1.0 / (1.0 + dist), distance=dist)

    return G


if __name__ == "__main__":
    sample = pd.read_csv("data/sampled_tracks.csv")
    G = build_knn_graph(sample)

    print(f"mutual={MUTUAL}, k={K}")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    degrees = [d for _, d in G.degree()]
    print("Degree: min=%d max=%d mean=%.2f" % (min(degrees), max(degrees), np.mean(degrees)))

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    print("Number of connected components:", len(components))
    print("Giant component size:", len(components[0]), f"({100*len(components[0])/G.number_of_nodes():.1f}% of nodes)")

    isolates = list(nx.isolates(G))
    print("Isolated nodes:", len(isolates))

    nx.write_gexf(G, "data/song_network.gexf")
    print("\nSaved graph to data/song_network.gexf")
