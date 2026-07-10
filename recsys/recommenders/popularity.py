"""Popularity baseline recommender.

Non-personalized: ranks movies by a weighted rating signal (IMDB-style
Bayesian average). Every personalized approach must beat this floor.
"""

from __future__ import annotations

import logging

import numpy as np

from recsys.interfaces import Recommender

logger = logging.getLogger(__name__)


class PopularityRecommender(Recommender):
    """Ranks movies by Bayesian-average rating (non-personalized).

    The score blends mean rating with a prior so that a movie rated 5.0 by
    one user doesn't outrank a movie rated 4.8 by 500 users.
    """

    name = "popularity"

    def __init__(self, min_votes: int = 5, prior: float = 3.0):
        self._min_votes = min_votes
        self._prior = prior
        self._scores: dict[int, float] = {}
        self._train: object | None = None

    def fit(self, train, content_features=None) -> None:
        ratings = train.ratings
        global_mean = float(ratings["rating"].mean())
        stats = (
            ratings.groupby("movieId")["rating"]
            .agg(["mean", "count"])
            .reset_index()
        )
        # Bayesian weighted average
        m = self._min_votes
        c = global_mean
        stats["score"] = (
            (stats["count"] / (stats["count"] + m)) * stats["mean"]
            + (m / (stats["count"] + m)) * c
        )
        self._scores = dict(zip(stats["movieId"].astype(int), stats["score"]))
        self._train = train
        logger.info("PopularityRecommender fit on %d movies", len(self._scores))

    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        if not self._scores:
            return []
        seen = self._train.rated_by_user(user_id) if self._train else set()
        candidates = [(mid, sc) for mid, sc in self._scores.items() if mid not in seen]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in candidates[:n]]
