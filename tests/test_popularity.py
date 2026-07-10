"""Tests for the PopularityRecommender (ticket #7)."""

from __future__ import annotations

import pytest

from recsys.recommenders import PopularityRecommender
from recsys.interfaces import Recommender


class TestPopularityRecommender:
    def test_implements_interface(self):
        assert issubclass(PopularityRecommender, Recommender)

    def test_name(self):
        assert PopularityRecommender().name == "popularity"

    def test_fit_sets_scores(self, train_ratings_data):
        rec = PopularityRecommender()
        rec.fit(train_ratings_data)
        assert len(rec._scores) > 0

    def test_recommend_returns_list(self, train_ratings_data):
        rec = PopularityRecommender()
        rec.fit(train_ratings_data)
        user_id = train_ratings_data.user_ids[0]
        result = rec.recommend(user_id, n=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_recommend_excludes_seen(self, train_ratings_data):
        """Recommended movies should not include movies the user already rated."""
        rec = PopularityRecommender()
        rec.fit(train_ratings_data)
        user_id = train_ratings_data.user_ids[0]
        seen = train_ratings_data.rated_by_user(user_id)
        result = rec.recommend(user_id, n=10)
        for mid in result:
            assert mid not in seen

    def test_recommend_unknown_user(self, train_ratings_data):
        """Unknown user still gets popularity-based recommendations."""
        rec = PopularityRecommender()
        rec.fit(train_ratings_data)
        result = rec.recommend(99999, n=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_popularity_ordering(self, train_ratings_data):
        """More popular movies should be recommended first for unknown users."""
        rec = PopularityRecommender(min_votes=1, prior=3.0)
        rec.fit(train_ratings_data)
        result = rec.recommend(99999, n=10)
        # Scores should be in descending order
        scores = [rec._scores[mid] for mid in result]
        assert scores == sorted(scores, reverse=True)

    def test_predict_returns_none(self, train_ratings_data):
        """Popularity does not predict ratings."""
        rec = PopularityRecommender()
        rec.fit(train_ratings_data)
        assert rec.predict(1, 1) is None

    def test_recommend_n_larger_than_pool(self, train_ratings_data):
        """Asking for more than available returns what's available."""
        rec = PopularityRecommender()
        rec.fit(train_ratings_data)
        user_id = train_ratings_data.user_ids[0]
        seen = train_ratings_data.rated_by_user(user_id)
        available = len(train_ratings_data.movie_ids) - len(seen)
        result = rec.recommend(user_id, n=1000)
        assert len(result) <= available

    def test_empty_recommend_before_fit(self):
        rec = PopularityRecommender()
        assert rec.recommend(1, n=5) == []
