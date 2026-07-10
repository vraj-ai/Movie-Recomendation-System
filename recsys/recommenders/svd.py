"""SVD (matrix factorization) recommender via TruncatedSVD."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import TruncatedSVD

from recsys.interfaces import Recommender

logger = logging.getLogger(__name__)


class SVDRecommender(Recommender):
    """Latent-factor matrix factorization of the user–movie ratings matrix.

    Uses ``TruncatedSVD`` to decompose the (mean-centered) user–item matrix
    into low-rank factors. Predicts unseen ratings by reconstructing the
    matrix from the factors, and ranks for top-N.
    """

    name = "svd"

    def __init__(self, n_components: int = 20, random_state: int = 42):
        self._n_components = n_components
        self._random_state = random_state
        self._svd: TruncatedSVD | None = None
        self._user_factors = None
        self._item_factors = None
        self._reconstructed = None
        self._global_mean = 0.0
        self._train = None

    def fit(self, train, content_features=None) -> None:
        self._train = train
        ui = train.user_item.astype(np.float64)
        self._global_mean = float(ui.data.mean())

        # Mean-center the matrix
        ui_centered = ui.copy()
        ui_centered.data -= self._global_mean

        n_comp = min(self._n_components, min(ui.shape) - 1)
        if n_comp < 1:
            n_comp = 1

        self._svd = TruncatedSVD(
            n_components=n_comp, random_state=self._random_state
        )
        self._user_factors = self._svd.fit_transform(ui_centered)
        self._item_factors = self._svd.components_
        # Reconstruct the predicted ratings matrix
        self._reconstructed = (
            self._user_factors @ self._item_factors + self._global_mean
        )
        logger.info(
            "SVDRecommender fit: %d components, matrix shape %s",
            n_comp, self._reconstructed.shape,
        )

    def _predict_idx(self, user_idx: int, movie_idx: int) -> float:
        return float(self._reconstructed[user_idx, movie_idx])

    def predict(self, user_id: int, movie_id: int) -> float | None:
        if self._reconstructed is None:
            return None
        u_idx = self._train.user_id_to_idx.get(user_id)
        m_idx = self._train.movie_id_to_idx.get(movie_id)
        if u_idx is None or m_idx is None:
            return self._global_mean  # fall back to global mean for unknowns
        return self._predict_idx(u_idx, m_idx)

    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        if self._reconstructed is None:
            return []
        u_idx = self._train.user_id_to_idx.get(user_id)
        if u_idx is None:
            return []
        scores = self._reconstructed[u_idx].copy()
        seen = self._train.rated_by_user(user_id)
        for mid in seen:
            m_idx = self._train.movie_id_to_idx.get(mid)
            if m_idx is not None:
                scores[m_idx] = -np.inf
        order = np.argsort(scores)[::-1]
        # Filter out masked (seen) items, then take top-n
        return [
            self._train.movie_ids[i]
            for i in order
            if scores[i] > -np.inf
        ][:n]
