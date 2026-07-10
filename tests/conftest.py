"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recsys.config import FIXTURE_DATA_DIR, FIXTURE_TMDB_CACHE
from recsys.data import load_ratings
from recsys.split import holdout_split
from recsys.enrichment import enrich
from recsys.tmdb import TMDBClient


@pytest.fixture(scope="session")
def fixture_data():
    """Loaded fixture RatingsData."""
    return load_ratings(data_dir=FIXTURE_DATA_DIR)


@pytest.fixture(scope="session")
def train_test(fixture_data):
    """Train/test split of the fixture data."""
    return holdout_split(fixture_data.ratings, test_fraction=0.2, seed=42)


@pytest.fixture(scope="session")
def train_ratings_data(fixture_data, train_test):
    """RatingsData rebuilt from the train split (with sparse matrix)."""
    from recsys.data import RatingsData
    import scipy.sparse as sp
    import numpy as np

    train_df, _ = train_test
    movie_ids = sorted(train_df["movieId"].unique())
    user_ids = sorted(train_df["userId"].unique())
    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    rows = train_df["userId"].map(user_id_to_idx).values
    cols = train_df["movieId"].map(movie_id_to_idx).values
    vals = train_df["rating"].values.astype(np.float64)
    user_item = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
    )
    return RatingsData(
        ratings=train_df,
        movies=fixture_data.movies,
        links=fixture_data.links,
        user_item=user_item,
        movie_ids=movie_ids,
        user_ids=user_ids,
        movie_id_to_idx=movie_id_to_idx,
        user_id_to_idx=user_id_to_idx,
    )


@pytest.fixture(scope="session")
def content_features(fixture_data):
    """TMDB content features for fixture movies (from pre-seeded cache)."""
    client = TMDBClient(api_key=None, cache_dir=FIXTURE_TMDB_CACHE)
    return enrich(fixture_data.movie_ids, fixture_data.tmdb_map, client)


@pytest.fixture
def small_ratings():
    """Tiny ratings DataFrame for hand-verifiable tests."""
    return pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 1},
        {"userId": 1, "movieId": 2, "rating": 4.0, "timestamp": 2},
        {"userId": 1, "movieId": 3, "rating": 2.0, "timestamp": 3},
        {"userId": 2, "movieId": 1, "rating": 4.0, "timestamp": 1},
        {"userId": 2, "movieId": 2, "rating": 5.0, "timestamp": 2},
        {"userId": 2, "movieId": 3, "rating": 3.0, "timestamp": 3},
        {"userId": 3, "movieId": 1, "rating": 3.0, "timestamp": 1},
        {"userId": 3, "movieId": 3, "rating": 5.0, "timestamp": 2},
    ])
