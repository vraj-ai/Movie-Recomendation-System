"""Common Recommender interface — the model seam."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recsys.data import RatingsData


class Recommender(ABC):
    """Base contract every recommender implements.

    All four approaches (popularity, content, SVD, hybrid) satisfy this
    interface so ``evaluate`` treats them uniformly.
    """

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        train: "RatingsData",
        content_features: dict | None = None,
    ) -> None:
        """Train on the given ratings (and optional TMDB content features)."""

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        """Return a ranked list of *n* movieIds for *user_id*."""

    def predict(self, user_id: int, movie_id: int) -> float | None:
        """Predict a numeric rating, or ``None`` if unsupported."""
        return None

    def similar(self, movie_id: int, n: int = 10) -> list[int]:
        """Return *n* movieIds similar to *movie_id*, or empty if unsupported."""
        return []
