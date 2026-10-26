# Song Similarity Network - Complex Network Analysis Project

A course project modeling a real-world complex system as a network and analyzing its
structure with standard complex-network-analysis techniques (degree distribution,
small-world testing, centrality, community detection, assortativity, and robustness
simulation).

## The system being modeled

**Nodes are songs. An edge connects two songs if they sound alike**, based on Spotify's
audio features (danceability, energy, loudness, speechiness, acousticness,
instrumentalness, liveness, valence, tempo). The result is a "sonic landscape" of
popular music, where tightly-connected neighborhoods correspond to sonically similar
songs, and we can ask how well that structure lines up with human-assigned genre labels.

## Data source

[TidyTuesday Spotify Songs dataset](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-01-21/readme.md)
(originally collected via the [`spotifyr`](https://www.rcharlie.com/spotifyr/) package
by Charlie Thompson, Josiah Parry, Donal Phipps, and Tom Wolff; genre curation by
Kaylin Pavlik). See `data/readme.md` for the full data dictionary.

- 32,833 track entries → 28,356 unique tracks after cleaning
- 6 genres: pop, rap, rock, latin, r&b, edm
- 2,500-track stratified sample used to build the network (see Methodology)

## Methodology / pipeline

| Script | What it does |
|---|---|
| `src/phase1_load_and_clean.py` | Load the CSV, deduplicate tracks (keep each track's most common genre label), select the 9 audio features |
| `src/phase2_sample.py` | Draw a 2,500-song sample, stratified proportionally by genre |
| `src/phase3_build_network.py` | Standardize features, build a mutual k-Nearest-Neighbors graph (k=10), weighted by similarity |
| `src/phase4_descriptive_stats.py` | Degree distribution, density, clustering, path length, small-world coefficient, Erdős–Rényi comparison plots |
| `src/phase4b_visualize_network.py` | Network drawing: edges colored by similarity, plus four layouts |
| `src/phase5_centrality.py` | All centrality distributions, each vs popularity, correlation heatmaps, genre tables |
| `src/phase6_community_detection.py` | Louvain community detection vs. true genre labels (NMI, ARI) |
| `src/phase7_assortativity_robustness.py` | Degree/genre assortativity; robustness under random failure vs. targeted attack |
| `src/phase8_k_sensitivity.py` | Basic stats for k = 5, 8, 10, 12, 15; within-community analysis for k = 5, 10, 15 |

Full rationale for each design decision (sampling size, k-NN vs. threshold, mutual vs.
union rule, handling of isolated nodes, etc.) is documented as docstrings/comments in
each script.

### Running it

```bash
pip install -r requirements.txt
python src/phase1_load_and_clean.py
python src/phase2_sample.py
python src/phase3_build_network.py
python src/phase4_descriptive_stats.py
python src/phase4b_visualize_network.py
python src/phase5_centrality.py
python src/phase6_community_detection.py
python src/phase7_assortativity_robustness.py
python src/phase8_k_sensitivity.py
```

Each script reads the previous step's output from `data/` and writes its own
outputs there, plus any figures to `figures/`.

## Extra figures

| Figure | What it shows |
|---|---|
| `figures/er_comparison_metrics.png` | Clustering and path length vs an Erdős–Rényi graph with the same size |
| `figures/er_degree_distribution.png` | Degree distribution overlaid on Erdős–Rényi graphs |
| `figures/network_overview.png` | Full network; node color = genre, edge color = similarity |
| `figures/layout_comparison.png` | Same giant component drawn with spring, Kamada–Kawai, spectral, and circular layouts |
| `figures/centrality_histograms.png` | Distribution of every centrality |
| `figures/centrality_vs_popularity.png` | Every centrality against `track_popularity` |
| `figures/centrality_correlation_pearson.png` | Pearson correlation of all centralities and popularity |
| `figures/centrality_correlation_spearman.png` | Spearman correlation of the same metrics |
| `figures/k_comparison_stats.png` | Basic stats for k = 5, 8, 10, 12, 15 |
| `figures/k_community_networks.png` | k = 5, 10, 15 networks colored by community |
| `figures/within_community_stats.png` | Internal density of the largest communities in those three graphs |

Genre breakdowns are tables, not plots: `data/centrality_by_genre.csv` and `data/centrality_popularity_corr_by_genre.csv`.

## Summary of findings

We built a network of 2,500 songs where each song is connected to others that sound
similar, based on its audio features (danceability, energy, tempo, etc.) rather than
its playlist or genre label. The resulting network is shown in **Figure 1**
(`figures/network_overview.png`), songs cluster into neighborhoods of similar sound,
with EDM standing out as a visually distinct island, while pop, rock, latin, and r&b
overlap heavily in the middle.

Structurally, the network shows a classic **small-world pattern**: songs clump into
tight local groups (clustering coefficient ~130x higher than a random network of the
same size, see **Figure 2**, `figures/degree_distribution.png`, for the degree
distribution behind this), yet any two songs are still typically only about 8 "similarity
hops" apart. When we let an algorithm detect communities purely from the network's
structure, without telling it any genre labels, it did find real clusters, but they only
loosely matched the official genre categories (**Figure 3**,
`figures/community_vs_genre.png`), again, with EDM as the one genre that is sonically
distinctive, while the rest blend together. This suggests genre is more of a cultural/
marketing label than something fully determined by these 9 audio measurements.

We also checked whether being a structurally "typical" song (one with lots of similar
neighbors) makes a song more popular. It doesn't, there is essentially no correlation
between any centrality measure and `track_popularity` (**Figure 4**,
`figures/closeness_centrality_vs_popularity.png`), meaning popularity is driven by
factors outside of how the song sounds relative to its neighbors (e.g. artist fame,
marketing, playlist placement).

Finally, we tested how tough the network is to damage (**Figure 5**,
`figures/robustness.png`). Removing songs at random barely affects it, the network
stays mostly connected even after removing most songs randomly. But specifically
removing the most "typical/well-connected" songs first causes the network to collapse
rapidly after only about 30-40% of nodes are removed. This "robust to random failure,
fragile to targeted attack" pattern is the same signature seen in real-world networks
like the internet and power grids, a small set of hub-like songs is structurally
load-bearing for keeping the whole network connected.
