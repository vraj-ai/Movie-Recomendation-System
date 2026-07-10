"""Tests for the data module (ticket #7)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recsys.data import load_ratings, RatingsData
from recsys.config import FIXTURE_DATA_DIR


class TestLoadRatings:
    def test_loads_fixture_data(self):
        data = load_ratings(data_dir=FIXTURE_DATA_DIR)
        assert isinstance(data, RatingsData)
        assert data.n_ratings > 0
        assert data.n_users > 0
        assert data.n_movies > 0

    def test_ratings_columns(self, fixture_data):
        for col in ("userId", "movieId", "rating", "timestamp"):
            assert col in fixture_data.ratings.columns

    def test_movies_columns(self, fixture_data):
        for col in ("movieId", "title", "genres"):
            assert col in fixture_data.movies.columns

    def test_links_columns(self, fixture_data):
        for col in ("movieId", "imdbId", "tmdbId"):
            assert col in fixture_data.links.columns

    def test_sparse_matrix_shape(self, fixture_data):
        assert fixture_data.user_item.shape == (
            fixture_data.n_users,
            fixture_data.n_movies,
        )

    def test_sparse_matrix_values(self, fixture_data):
        # Check that the sparse matrix has the right number of non-zero entries
        assert fixture_data.user_item.nnz == fixture_data.n_ratings

    def test_tmdb_map(self, fixture_data):
        tmdb = fixture_data.tmdb_map
        assert len(tmdb) > 0
        # All values should be ints
        for movie_id, tmdb_id in tmdb.items():
            assert isinstance(movie_id, int)
            assert isinstance(tmdb_id, int)

    def test_rated_by_user(self, fixture_data):
        user_id = fixture_data.user_ids[0]
        rated = fixture_data.rated_by_user(user_id)
        assert len(rated) > 0
        # All rated movies should be in movie_ids
        for mid in rated:
            assert mid in fixture_data.movie_ids

    def test_rated_by_unknown_user(self, fixture_data):
        rated = fixture_data.rated_by_user(99999)
        assert rated == set()

    def test_movie_id_to_idx_consistency(self, fixture_data):
        for mid, idx in fixture_data.movie_id_to_idx.items():
            assert fixture_data.movie_ids[idx] == mid

    def test_user_id_to_idx_consistency(self, fixture_data):
        for uid, idx in fixture_data.user_id_to_idx.items():
            assert fixture_data.user_ids[idx] == uid


class TestFailFast:
    def test_missing_dataset_fails_fast(self, tmp_path):
        """FAIL-6: missing dataset raises FileNotFoundError with instructions."""
        with pytest.raises(FileNotFoundError, match="FAIL-6"):
            load_ratings(data_dir=tmp_path / "nonexistent")

    def test_missing_ratings_csv_fails_fast(self, tmp_path):
        """FAIL-6: partial dataset raises FileNotFoundError."""
        (tmp_path / "movies.csv").write_text("movieId,title,genres\n1,A,Comedy\n")
        (tmp_path / "links.csv").write_text("movieId,imdbId,tmdbId\n1,1,1\n")
        with pytest.raises(FileNotFoundError, match="ratings.csv"):
            load_ratings(data_dir=tmp_path)

    def test_fixture_flag_loads_fixture(self):
        data = load_ratings(fixture=True)
        assert data.n_ratings > 0

    def test_fixture_data_has_10_movies(self, fixture_data):
        assert fixture_data.n_movies == 10

    def test_fixture_data_has_10_users(self, fixture_data):
        assert fixture_data.n_users == 10
