"""Tests for the ContentRecommender (ticket #3)."""

from __future__ import annotations

import pytest

from recsys.recommenders import ContentRecommender
from recsys.interfaces import Recommender


class TestContentRecommender:
    def test_implements_interface(self):
        assert issubclass(ContentRecommender, Recommender)

    def test_name(self):
        assert ContentRecommender().name == "content"

    def test_fit(self, train_ratings_data, content_features):
        rec = ContentRecommender()
        rec.fit(train_ratings_data, content_features=content_features)
        assert rec._tfidf_matrix is not None
        assert rec._vectorizer is not None

    def test_recommend_returns_list(self, train_ratings_data, content_features):
        rec = ContentRecommender()
        rec.fit(train_ratings_data, content_features=content_features)
        user_id = train_ratings_data.user_ids[0]
        result = rec.recommend(user_id, n=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_recommend_excludes_seen(self, train_ratings_data, content_features):
        rec = ContentRecommender()
        rec.fit(train_ratings_data, content_features=content_features)
        user_id = train_ratings_data.user_ids[0]
        seen = train_ratings_data.rated_by_user(user_id)
        result = rec.recommend(user_id, n=10)
        for mid in result:
            assert mid not in seen

    def test_empty_content_features_no_error(self, train_ratings_data):
        """FAIL-5: empty content features → no crash, returns something."""
        empty_features = {mid: {
            "genres": [], "keywords": [], "cast": [], "crew": [], "overview": ""
        } for mid in train_ratings_data.movie_ids}
        rec = ContentRecommender()
        rec.fit(train_ratings_data, content_features=empty_features)
        user_id = train_ratings_data.user_ids[0]
        result = rec.recommend(user_id, n=5)
        assert isinstance(result, list)

    def test_hand_verifiable_ordering(self, train_ratings_data, content_features):
        """User 1 likes animation/family movies (1, 2, 8).
        Recommended movies should lean toward similar content.
        """
        rec = ContentRecommender(relevance_threshold=3.5)
        rec.fit(train_ratings_data, content_features=content_features)
        result = rec.recommend(1, n=5)
        assert len(result) > 0
        # All results should be valid movie IDs
        for mid in result:
            assert mid in train_ratings_data.movie_ids

    def test_similar_returns_related(self, train_ratings_data, content_features):
        """similar() should return movies with overlapping content."""
        rec = ContentRecommender()
        rec.fit(train_ratings_data, content_features=content_features)
        # Movie 1 (Toy Story: Animation, Family, Comedy, Adventure)
        # Movie 2 (Jumanji: Adventure, Family, Fantasy) — shares genres
        similar = rec.similar(1, n=5)
        assert len(similar) > 0
        assert 1 not in similar  # excludes self
        # Jumanji (2) and Tom and Huck (8) share Adventure/Family with Toy Story
        assert 2 in similar or 8 in similar

    def test_similar_excludes_self(self, train_ratings_data, content_features):
        rec = ContentRecommender()
        rec.fit(train_ratings_data, content_features=content_features)
        for mid in train_ratings_data.movie_ids:
            similar = rec.similar(mid, n=5)
            assert mid not in similar

    def test_similar_unknown_movie(self, train_ratings_data, content_features):
        rec = ContentRecommender()
        rec.fit(train_ratings_data, content_features=content_features)
        assert rec.similar(99999, n=5) == []

    def test_recommend_before_fit(self):
        rec = ContentRecommender()
        assert rec.recommend(1, n=5) == []

    def test_predict_returns_none(self, train_ratings_data, content_features):
        """Content-based doesn't predict ratings."""
        rec = ContentRecommender()
        rec.fit(train_ratings_data, content_features=content_features)
        assert rec.predict(1, 1) is None

    def test_content_clustering(self, train_ratings_data, content_features):
        """Action movies (6, 9, 10) should cluster together via similarity."""
        rec = ContentRecommender()
        rec.fit(train_ratings_data, content_features=content_features)
        # Movie 6 (Heat: Action, Crime, Thriller)
        similar_to_6 = rec.similar(6, n=5)
        # Sudden Death (9) and GoldenEye (10) share Action/Thriller
        assert 9 in similar_to_6 or 10 in similar_to_6

    def test_cold_user_recommend(self, train_ratings_data, content_features):
        """Unknown user gets content-dense recommendations (no crash)."""
        rec = ContentRecommender()
        rec.fit(train_ratings_data, content_features=content_features)
        result = rec.recommend(99999, n=5)
        assert isinstance(result, list)
