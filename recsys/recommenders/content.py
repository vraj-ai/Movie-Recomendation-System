"""Content-based recommender: TF-IDF + cosine over TMDB features."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recsys.interfaces import Recommender

logger = logging.getLogger(__name__)


def _features_to_text(features: dict) -> str:
    """Concatenate TMDB content features into a single text blob."""
    parts: list[str] = []
    for key in ("genres", "keywords", "cast", "crew", "overview"):
        val = features.get(key, "")
        if isinstance(val, list):
            parts.append(" ".join(str(v) for v in val))
        else:
            parts.append(str(val))
    return " ".join(parts).strip()


class ContentRecommender(Recommender):
    """TF-IDF over concatenated TMDB features + cosine similarity.

    A user profile is the centroid of TF-IDF vectors of movies the user
    rated at/above the relevance threshold. Unseen movies are ranked by
    cosine similarity to this profile.
    """

    name = "content"

    def __init__(self, relevance_threshold: float = 3.5):
        self._threshold = relevance_threshold
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._movie_ids: list[int] = []
        self._movie_id_to_idx: dict[int, int] = {}
        self._train = None
        self._cosine_cache = None

    def fit(self, train, content_features: dict | None = None) -> None:
        if content_features is None:
            content_features = {}

        # Build text corpus for all movies in training ratings
        movie_ids = sorted(train.ratings["movieId"].unique())
        self._movie_ids = movie_ids
        self._movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

        corpus = []
        for mid in movie_ids:
            feats = content_features.get(mid, {})
            corpus.append(_features_to_text(feats))

        self._vectorizer = TfidfVectorizer(stop_words="english")
        try:
            self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        except ValueError:
            # Empty vocabulary (all documents are empty/stop-words) →
            # create a zero matrix so the model degrades gracefully (FAIL-4/FAIL-5)
            self._tfidf_matrix = None
        self._train = train
        logger.info("ContentRecommender fit on %d movies", len(movie_ids))

    def _user_profile(self, user_id: int) -> np.ndarray | None:
        """Centroid of TF-IDF vectors for the user's liked (train) movies."""
        if self._tfidf_matrix is None:
            return None
        ratings = self._train.ratings
        user_ratings = ratings[ratings["userId"] == user_id]
        liked = user_ratings[user_ratings["rating"] >= self._threshold]
        indices = [
            self._movie_id_to_idx[mid]
            for mid in liked["movieId"]
            if mid in self._movie_id_to_idx
        ]
        if not indices:
            # Fall back to all rated movies
            indices = [
                self._movie_id_to_idx[mid]
                for mid in user_ratings["movieId"]
                if mid in self._movie_id_to_idx
            ]
        if not indices:
            return None
        profile = self._tfidf_matrix[indices].mean(axis=0)
        return np.asarray(profile).ravel()

    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        if self._tfidf_matrix is None:
            if self._train is None:
                return []  # not fitted yet
            # No content features — fall back to recommending unseen movies
            # in arbitrary order (popularity would be better, but we don't
            # have that signal here). This is the FAIL-5 degradation path.
            seen = self._train.rated_by_user(user_id)
            return [mid for mid in self._movie_ids if mid not in seen][:n]

        profile = self._user_profile(user_id)
        if profile is None:
            # Cold user: return by TF-IDF norm (most content-dense movies)
            norms = np.asarray(self._tfidf_matrix.sum(axis=1)).ravel()
            order = np.argsort(norms)[::-1]
            seen = self._train.rated_by_user(user_id)
            result = [self._movie_ids[i] for i in order if self._movie_ids[i] not in seen]
            return result[:n]

        sims = cosine_similarity(profile.reshape(1, -1), self._tfidf_matrix).ravel()
        seen = self._train.rated_by_user(user_id)
        order = np.argsort(sims)[::-1]
        result = [self._movie_ids[i] for i in order if self._movie_ids[i] not in seen]
        return result[:n]

    def similar(self, movie_id: int, n: int = 10) -> list[int]:
        """Return *n* movies most similar to *movie_id* by cosine similarity."""
        if self._tfidf_matrix is None or movie_id not in self._movie_id_to_idx:
            return []
        idx = self._movie_id_to_idx[movie_id]
        vec = self._tfidf_matrix[idx]
        sims = cosine_similarity(vec, self._tfidf_matrix).ravel()
        sims[idx] = -1.0  # exclude self
        order = np.argsort(sims)[::-1]
        return [self._movie_ids[i] for i in order[:n]]
