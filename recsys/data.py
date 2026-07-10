"""Data loading for MovieLens ratings + links.

Loads ``ml-latest-small`` (or a bundled fixture) into a sparse user-item
matrix plus a ``movieId -> tmdbId`` mapping. Fails fast with manual-download
instructions if the dataset is unavailable (FAIL-6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.config import (
    DATA_SUBDIR,
    FIXTURE_DATA_DIR,
    MOVIELENS_DOWNLOAD_INSTRUCTIONS,
)

logger = logging.getLogger(__name__)

_RATINGS_COLS = ["userId", "movieId", "rating", "timestamp"]
_MOVIES_COLS = ["movieId", "title", "genres"]
_LINKS_COLS = ["movieId", "imdbId", "tmdbId"]


@dataclass
class RatingsData:
    """Container for loaded ratings, movies, links, and the sparse matrix."""

    ratings: pd.DataFrame
    movies: pd.DataFrame
    links: pd.DataFrame
    user_item: sp.csr_matrix
    movie_ids: list[int]
    user_ids: list[int]
    movie_id_to_idx: dict[int, int] = field(default_factory=dict)
    user_id_to_idx: dict[int, int] = field(default_factory=dict)

    @property
    def tmdb_map(self) -> dict[int, int]:
        """``{movieId: tmdbId}`` excluding rows where tmdbId is NaN."""
        valid = self.links.dropna(subset=["tmdbId"])
        return dict(zip(valid["movieId"].astype(int), valid["tmdbId"].astype(int)))

    def rated_by_user(self, user_id: int) -> set[int]:
        """Set of movieIds the user has rated (in this data)."""
        if user_id not in self.user_id_to_idx:
            return set()
        row = self.user_item[self.user_id_to_idx[user_id]]
        return {self.movie_ids[i] for i in row.indices}

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_movies(self) -> int:
        return len(self.movie_ids)

    @property
    def n_ratings(self) -> int:
        return self.ratings.shape[0]


def _resolve_data_dir(fixture: bool = False) -> Path:
    """Return the data directory to use, or raise FAIL-6."""
    if fixture:
        d = FIXTURE_DATA_DIR
        if not d.exists():
            raise FileNotFoundError(
                f"Fixture dataset not found at {d}. The bundle may be corrupted."
            )
        return d
    d = DATA_SUBDIR
    if not d.exists():
        raise FileNotFoundError(
            f"MovieLens dataset not found at {d}.\n"
            f"Manual download required (FAIL-6):\n  {MOVIELENS_DOWNLOAD_INSTRUCTIONS}"
        )
    return d


def load_ratings(fixture: bool = False, data_dir: Path | None = None) -> RatingsData:
    """Load MovieLens ratings, movies, and links.

    Args:
        fixture: If True, load the bundled fixture dataset.
        data_dir: Override the data directory (for testing).

    Raises:
        FileNotFoundError: If the dataset is missing (FAIL-6).
    """
    if data_dir is not None:
        d = data_dir
    else:
        d = _resolve_data_dir(fixture)

    ratings_path = d / "ratings.csv"
    movies_path = d / "movies.csv"
    links_path = d / "links.csv"

    for p, name in [(ratings_path, "ratings.csv"), (movies_path, "movies.csv"),
                     (links_path, "links.csv")]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file {name} not found in {d} (FAIL-6).\n"
                f"  {MOVIELENS_DOWNLOAD_INSTRUCTIONS}"
            )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    links = pd.read_csv(links_path)

    # Ensure expected columns
    for col in _RATINGS_COLS:
        if col not in ratings.columns:
            raise ValueError(f"ratings.csv missing column '{col}' (FAIL-6)")
    for col in _LINKS_COLS:
        if col not in links.columns:
            raise ValueError(f"links.csv missing column '{col}' (FAIL-6)")

    # Sort and index
    ratings = ratings.sort_values(["userId", "movieId"]).reset_index(drop=True)

    # Build ordered id lists (only movies that appear in ratings)
    rated_movie_ids = sorted(ratings["movieId"].unique())
    user_ids = sorted(ratings["userId"].unique())

    movie_id_to_idx = {mid: i for i, mid in enumerate(rated_movie_ids)}
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    # Build sparse user-item matrix
    rows = ratings["userId"].map(user_id_to_idx).values
    cols = ratings["movieId"].map(movie_id_to_idx).values
    vals = ratings["rating"].values.astype(np.float64)
    user_item = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user_ids), len(rated_movie_ids)),
    )

    logger.info(
        "Loaded %d ratings, %d users, %d movies from %s",
        len(ratings), len(user_ids), len(rated_movie_ids), d,
    )

    return RatingsData(
        ratings=ratings,
        movies=movies,
        links=links,
        user_item=user_item,
        movie_ids=rated_movie_ids,
        user_ids=user_ids,
        movie_id_to_idx=movie_id_to_idx,
        user_id_to_idx=user_id_to_idx,
    )
