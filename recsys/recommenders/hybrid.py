"""Hybrid recommender: weighted blend of content + CF (SVD) scores."""

from __future__ import annotations

import logging

import numpy as np

from recsys.interfaces import Recommender
from recsys.recommenders.content import ContentRecommender
from recsys.recommenders.svd import SVDRecommender

logger = logging.getLogger(__name__)


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1]; if all values equal, return uniform."""
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


class HybridRecommender(Recommender):
    """Weighted blend of normalized content-based and SVD scores.

    ``content_weight`` in [0, 1]: 0 = pure CF, 1 = pure content.
    Scores from each sub-model are min-max normalized before blending so
    the two signals are on a comparable scale.
    """

    name = "hybrid"

    def __init__(
        self,
        content_weight: float = 0.5,
        n_components: int = 20,
        relevance_threshold: float = 3.5,
        random_state: int = 42,
    ):
        if not 0 <= content_weight <= 1:
            raise ValueError(f"content_weight must be in [0, 1], got {content_weight}")
        self._content_weight = content_weight
        self._content = ContentRecommender(relevance_threshold=relevance_threshold)
        self._svd = SVDRecommender(
            n_components=n_components, random_state=random_state
        )
        self._train = None
        self._content_scores_cache: dict[int, np.ndarray] = {}

    def fit(self, train, content_features: dict | None = None) -> None:
        self._train = train
        self._content_scores_cache.clear()
        self._content.fit(train, content_features=content_features)
        self._svd.fit(train, content_features=content_features)
        logger.info("HybridRecommender fit (content_weight=%.2f)", self._content_weight)

    def _content_scores(self, user_id: int) -> np.ndarray:
        """Get cosine similarity scores for all movies for this user."""
        if self._content._tfidf_matrix is None:
            return np.zeros(len(self._train.movie_ids))

        if user_id in self._content_scores_cache:
            return self._content_scores_cache[user_id]

        from sklearn.metrics.pairwise import cosine_similarity

        profile = self._content._user_profile(user_id)
        if profile is None:
            scores = np.zeros(len(self._train.movie_ids))
        else:
            scores = cosine_similarity(
                profile.reshape(1, -1), self._content._tfidf_matrix
            ).ravel()
        self._content_scores_cache[user_id] = scores
        return scores

    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        if self._train is None:
            return []

        u_idx = self._train.user_id_to_idx.get(user_id)
        if u_idx is None:
            return []

        # SVD scores
        svd_scores = self._svd._reconstructed[u_idx].copy()
        # Content scores
        content_scores = self._content_scores(user_id)

        # Normalize both to [0, 1]
        svd_norm = _minmax_normalize(svd_scores)
        content_norm = _minmax_normalize(content_scores)

        # Blend
        w = self._content_weight
        blended = w * content_norm + (1 - w) * svd_norm

        # Mask seen items
        seen = self._train.rated_by_user(user_id)
        for mid in seen:
            m_idx = self._train.movie_id_to_idx.get(mid)
            if m_idx is not None:
                blended[m_idx] = -np.inf

        order = np.argsort(blended)[::-1]
        # Filter out masked (seen) items, then take top-n
        return [
            self._train.movie_ids[i]
            for i in order
            if blended[i] > -np.inf
        ][:n]

    def predict(self, user_id: int, movie_id: int) -> float | None:
        return self._svd.predict(user_id, movie_id)
