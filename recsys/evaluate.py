"""Evaluation metrics: Precision@K, Recall@K, NDCG@K, RMSE, MAE."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from recsys.interfaces import Recommender
    from recsys.data import RatingsData


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-*k* recommendations that are relevant."""
    if k <= 0:
        return 0.0
    topk = recommended[:k]
    if not topk:
        return 0.0
    hits = sum(1 for m in topk if m in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items that appear in the top-*k*."""
    if not relevant:
        return 0.0
    topk = recommended[:k]
    hits = sum(1 for m in topk if m in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at *k*.

    DCG = sum(rel_i / log2(i+2)) for i in 0..k-1
    IDCG = DCG of the ideal ordering (all relevant items first)
    NDCG = DCG / IDCG
    """
    if k <= 0 or not relevant:
        return 0.0
    topk = recommended[:k]

    dcg = 0.0
    for i, movie_id in enumerate(topk):
        if movie_id in relevant:
            dcg += 1.0 / math.log2(i + 2)

    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def rmse(predictions: list[float], actuals: list[float]) -> float:
    """Root Mean Square Error between predicted and actual ratings."""
    if not predictions:
        return 0.0
    preds = np.array(predictions, dtype=float)
    acts = np.array(actuals, dtype=float)
    return float(np.sqrt(np.mean((preds - acts) ** 2)))


def mae(predictions: list[float], actuals: list[float]) -> float:
    """Mean Absolute Error between predicted and actual ratings."""
    if not predictions:
        return 0.0
    preds = np.array(predictions, dtype=float)
    acts = np.array(actuals, dtype=float)
    return float(np.mean(np.abs(preds - acts)))


def evaluate_recommender(
    recommender: "Recommender",
    train: "RatingsData",
    test_ratings,
    k: int = 10,
    relevance_threshold: float = 3.5,
    n: int = 10,
) -> dict:
    """Evaluate *recommender* on ranking + (optionally) rating metrics.

    Args:
        recommender: A fitted ``Recommender``.
        train: Training ``RatingsData`` (already fit).
        test_ratings: DataFrame with ``userId, movieId, rating``.
        k: K for Precision/Recall/NDCG.
        relevance_threshold: Rating >= this is a "hit".
        n: How many items to recommend per user.

    Returns:
        Dict with ``precision_at_k``, ``recall_at_k``, ``ndcg_at_k``,
        and optionally ``rmse``, ``mae`` if the recommender supports ``predict``.
    """
    from recsys.split import relevant_items

    rel_map = relevant_items(test_ratings, relevance_threshold)
    test_users = test_ratings["userId"].unique()

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []

    for user_id in test_users:
        user_id_cast = int(user_id)
        rel = rel_map.get(user_id_cast, set())
        # Recommend items not in the user's training set
        seen = train.rated_by_user(user_id_cast)
        recs = recommender.recommend(user_id_cast, n=max(n, k))
        recs = [m for m in recs if m not in seen]
        precisions.append(precision_at_k(recs, rel, k))
        recalls.append(recall_at_k(recs, rel, k))
        ndcgs.append(ndcg_at_k(recs, rel, k))

    result: dict = {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }

    # Rating-prediction metrics (only for models that implement predict)
    preds: list[float] = []
    acts: list[float] = []
    for row in test_ratings.itertuples(index=False):
        p = recommender.predict(int(row.userId), int(row.movieId))
        if p is not None:
            preds.append(p)
            acts.append(float(row.rating))
    if preds:
        result["rmse"] = rmse(preds, acts)
        result["mae"] = mae(preds, acts)

    return result
