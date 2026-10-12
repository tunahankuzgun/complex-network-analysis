"""
Phase 1 - Load & clean the Spotify songs data.

Produces a clean dataframe with exactly one row per unique track_id, keeping
the columns needed for the network (identity + genre labels + the 9 audio
features), with no missing values in the feature columns.
"""

import pandas as pd

DATA_PATH = "data/spotify_songs.csv"

# The 9 numeric audio features we'll use to measure song similarity.
# (left out: key, mode -> categorical, treated differently if you want to add them later;
#  duration_ms -> about length, not "sound", so not included for now)
FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Columns we want to keep around as node metadata (not used for similarity,
# but useful for labeling, interpretation, and validating communities later).
METADATA_COLUMNS = [
    "track_id",
    "track_name",
    "track_artist",
    "track_popularity",
    "playlist_genre",
    "playlist_subgenre",
]


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV."""
    return pd.read_csv(path)


def dedupe_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per unique track_id.

    Strategy chosen: for each track_id, keep the row whose playlist_genre is
    the most frequent ("mode") genre for that track across all the playlists
    it appears in. This is more robust than "first occurrence" because it
    isn't sensitive to row order in the original CSV, and it reflects the
    genre the track is *most strongly associated with* across the dataset
    rather than an arbitrary single playlist. Audio features and track-level
    metadata (name, artist, popularity) are identical across duplicate rows
    for the same track_id, so we just need to pick which playlist_genre /
    playlist_subgenre to keep.

    Ties in the mode (e.g. a track appearing in exactly one 'pop' and one
    'rock' playlist) are broken by keeping whichever row pandas encounters
    first for that (track_id, genre) pair - acceptable for our purposes, and
    documented here for the report.
    """
    # Determine the most common playlist_genre per track_id.
    dominant_genre = (
        df.groupby("track_id")["playlist_genre"]
        .agg(lambda s: s.mode().iloc[0])
        .rename("dominant_genre")
    )

    # Keep only rows whose playlist_genre matches that track's dominant genre,
    # then drop any remaining duplicates (e.g. the track appears in two
    # different 'pop' playlists - subgenre might differ, so just take the
    # first such row).
    merged = df.merge(dominant_genre, on="track_id")
    matching = merged[merged["playlist_genre"] == merged["dominant_genre"]]
    deduped = matching.drop_duplicates(subset="track_id", keep="first").drop(
        columns="dominant_genre"
    )
    return deduped.reset_index(drop=True)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only METADATA_COLUMNS + FEATURE_COLUMNS."""
    return df[METADATA_COLUMNS + FEATURE_COLUMNS].copy()


def check_missing(df: pd.DataFrame) -> None:
    """
    Print how many missing values exist per column.
    This is a check, not a transformation - just report what you find.
    """
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print("Missing values per column:\n", missing)


def clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Run the full Phase 1 pipeline and return the cleaned dataframe."""
    df = load_raw(path)
    df = dedupe_tracks(df)
    df = select_columns(df)
    check_missing(df)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df


if __name__ == "__main__":
    cleaned = clean()
    print("Shape:", cleaned.shape)
    print(cleaned.head())
    print("\nUnique tracks:", cleaned["track_id"].nunique())
    print("\nGenre counts:\n", cleaned["playlist_genre"].value_counts())
