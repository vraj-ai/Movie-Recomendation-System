"""Per-user hold-out splitter for ratings."""

from __future__ import annotations

import pandas as pd
import numpy as np


def holdout_split(
    ratings: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings per-user into train and hold-out test sets.

    For each user, a ``test_fraction`` of their ratings is held out for test.
    Users with fewer than 2 ratings keep all ratings in train.

    Guarantees:
        - Train and test are **disjoint** per user (no rating in both).
        - Every test user appears in train (so the model has seen them).

    Args:
        ratings: DataFrame with columns ``userId, movieId, rating, timestamp``.
        test_fraction: Fraction of each user's ratings to hold out (0–1).
        seed: RNG seed for reproducibility.

    Returns:
        ``(train_df, test_df)`` — same columns as input.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    if ratings.empty:
        raise ValueError("ratings DataFrame is empty; cannot split")

    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for user_id, group in ratings.groupby("userId"):
        n = len(group)
        if n < 2:
            train_parts.append(group)
            continue
        n_test = max(1, int(round(n * test_fraction)))
        n_test = min(n_test, n - 1)  # keep at least 1 in train
        shuffled = group.sample(frac=1.0, random_state=rng.integers(2**31))
        test_parts.append(shuffled.iloc[:n_test])
        train_parts.append(shuffled.iloc[n_test:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def relevant_items(
    test_ratings: pd.DataFrame,
    threshold: float = 3.5,
) -> dict[int, set[int]]:
    """Return ``{userId: {movieId, ...}}`` for test ratings at/above *threshold*."""
    relevant = test_ratings[test_ratings["rating"] >= threshold]
    return (
        relevant.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )
