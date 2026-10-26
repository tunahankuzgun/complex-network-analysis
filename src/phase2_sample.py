"""
Phase 2 - Stratified sample of tracks for the network.

Draws a sample of SAMPLE_SIZE songs from the cleaned ~28k tracks,
proportionally stratified by playlist_genre, so the sample's genre mix
matches the full dataset's genre mix. Fixed random seed for reproducibility.
"""

import pandas as pd

import phase1_load_and_clean as clean_module

SAMPLE_SIZE = 2500
RANDOM_SEED = 6


def stratified_sample(
    df: pd.DataFrame, n: int = SAMPLE_SIZE, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Proportionally sample n rows from df, stratified by playlist_genre.

    Each genre's share of the sample matches its share of the full dataset
    (rounded to whole songs), e.g. if 'edm' is ~19% of the data, it will be
    ~19% of the sample too.
    """
    genre_fracs = df["playlist_genre"].value_counts(normalize=True)
    sampled_parts = []
    for genre, frac in genre_fracs.items():
        genre_df = df[df["playlist_genre"] == genre]
        n_genre = round(n * frac)
        sampled_parts.append(
            genre_df.sample(n=min(n_genre, len(genre_df)), random_state=seed)
        )
    sample = pd.concat(sampled_parts).sample(frac=1, random_state=seed)  # shuffle
    return sample.reset_index(drop=True)


if __name__ == "__main__":
    cleaned = clean_module.clean()
    sample = stratified_sample(cleaned)
    print("Sample shape:", sample.shape)
    print("\nGenre share in full data:\n", cleaned["playlist_genre"].value_counts(normalize=True))
    print("\nGenre share in sample:\n", sample["playlist_genre"].value_counts(normalize=True))
    sample.to_csv("data/sampled_tracks.csv", index=False)
    print("\nSaved to data/sampled_tracks.csv")
