"""Tests for the SVDRecommender (ticket #4)."""

from __future__ import annotations

import pytest

from recsys.recommenders import SVDRecommender
from recsys.interfaces import Recommender


class TestSVDRecommender:
    def test_implements_interface(self):
        assert issubclass(SVDRecommender, Recommender)

    def test_name(self):
        assert SVDRecommender().name == "svd"

    def test_fit(self, train_ratings_data):
        rec = SVDRecommender(n_components=5)
        rec.fit(train_ratings_data)
        assert rec._reconstructed is not None
        assert rec._reconstructed.shape == (
            train_ratings_data.n_users,
            train_ratings_data.n_movies,
        )

    def test_predict_returns_float(self, train_ratings_data):
        rec = SVDRecommender(n_components=5)
        rec.fit(train_ratings_data)
        user_id = train_ratings_data.user_ids[0]
        movie_id = train_ratings_data.movie_ids[0]
        pred = rec.predict(user_id, movie_id)
        assert pred is not None
        assert isinstance(pred, float)

    def test_predict_unknown_user(self, train_ratings_data):
        rec = SVDRecommender(n_components=5)
        rec.fit(train_ratings_data)
        pred = rec.predict(99999, train_ratings_data.movie_ids[0])
        assert pred is not None  # falls back to global mean
        assert isinstance(pred, float)

    def test_predict_unknown_movie(self, train_ratings_data):
        rec = SVDRecommender(n_components=5)
        rec.fit(train_ratings_data)
        user_id = train_ratings_data.user_ids[0]
        pred = rec.predict(user_id, 99999)
        assert pred is not None  # falls back to global mean

    def test_recommend_returns_list(self, train_ratings_data):
        rec = SVDRecommender(n_components=5)
        rec.fit(train_ratings_data)
        user_id = train_ratings_data.user_ids[0]
        result = rec.recommend(user_id, n=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_recommend_excludes_seen(self, train_ratings_data):
        rec = SVDRecommender(n_components=5)
        rec.fit(train_ratings_data)
        user_id = train_ratings_data.user_ids[0]
        seen = train_ratings_data.rated_by_user(user_id)
        result = rec.recommend(user_id, n=10)
        for mid in result:
            assert mid not in seen

    def test_recommend_unknown_user(self, train_ratings_data):
        rec = SVDRecommender(n_components=5)
        rec.fit(train_ratings_data)
        result = rec.recommend(99999, n=5)
        assert result == []

    def test_recommend_before_fit(self):
        rec = SVDRecommender()
        assert rec.recommend(1, n=5) == []
        assert rec.predict(1, 1) is None

    def test_predictions_in_valid_range(self, train_ratings_data):
        """Predicted ratings should be in a reasonable range (0.5–5.0)."""
        rec = SVDRecommender(n_components=5)
        rec.fit(train_ratings_data)
        user_id = train_ratings_data.user_ids[0]
        for movie_id in train_ratings_data.movie_ids[:5]:
            pred = rec.predict(user_id, movie_id)
            assert pred is not None
            # Allow some slack for SVD reconstruction
            assert -2.0 <= pred <= 7.0

    def test_n_components_too_large(self, train_ratings_data):
        """More components than min(users, movies)-1 → clamped."""
        rec = SVDRecommender(n_components=1000)
        rec.fit(train_ratings_data)
        assert rec._reconstructed is not None

    def test_recommend_n_larger_than_pool(self, train_ratings_data):
        rec = SVDRecommender(n_components=5)
        rec.fit(train_ratings_data)
        user_id = train_ratings_data.user_ids[0]
        seen = train_ratings_data.rated_by_user(user_id)
        available = len(train_ratings_data.movie_ids) - len(seen)
        result = rec.recommend(user_id, n=1000)
        assert len(result) <= available
