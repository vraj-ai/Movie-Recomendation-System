"""Tests for the per-user hold-out splitter (ticket #7)."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.split import holdout_split, relevant_items


class TestHoldoutSplit:
    def test_train_test_disjoint(self, fixture_data):
        """No rating appears in both train and test."""
        train, test = holdout_split(fixture_data.ratings, test_fraction=0.2)
        # Merge keys for comparison
        train_keys = set(zip(train["userId"], train["movieId"]))
        test_keys = set(zip(test["userId"], test["movieId"]))
        assert train_keys.isdisjoint(test_keys)

    def test_every_test_user_in_train(self, fixture_data):
        """Every user in test must also appear in train."""
        train, test = holdout_split(fixture_data.ratings, test_fraction=0.2)
        train_users = set(train["userId"])
        test_users = set(test["userId"])
        assert test_users.issubset(train_users)

    def test_per_user_split(self, fixture_data):
        """Each user has ratings in both train and test (if they have >= 2)."""
        train, test = holdout_split(fixture_data.ratings, test_fraction=0.2)
        for user_id in fixture_data.ratings["userId"].unique():
            user_train = train[train["userId"] == user_id]
            user_test = test[test["userId"] == user_id]
            user_total = len(fixture_data.ratings[fixture_data.ratings["userId"] == user_id])
            # Users with >= 2 ratings should have at least 1 in test
            if user_total >= 2:
                assert len(user_train) >= 1
                assert len(user_test) >= 1

    def test_test_fraction_respected(self):
        """Test set is approximately test_fraction of total."""
        df = pd.DataFrame([
            {"userId": i // 10, "movieId": i % 10, "rating": 4.0, "timestamp": i}
            for i in range(100)
        ])
        train, test = holdout_split(df, test_fraction=0.2)
        # 10 users, each with 10 ratings → ~2 per user in test = ~20 total
        assert 15 <= len(test) <= 25

    def test_single_rating_user_all_train(self):
        """Users with 1 rating keep it in train."""
        df = pd.DataFrame([
            {"userId": 1, "movieId": 1, "rating": 4.0, "timestamp": 1},
            {"userId": 1, "movieId": 2, "rating": 3.0, "timestamp": 2},
            {"userId": 2, "movieId": 1, "rating": 5.0, "timestamp": 1},
        ])
        train, test = holdout_split(df, test_fraction=0.2)
        assert len(test[test["userId"] == 2]) == 0
        assert len(train[train["userId"] == 2]) == 1

    def test_reproducible(self, fixture_data):
        """Same seed produces same split."""
        t1, te1 = holdout_split(fixture_data.ratings, seed=42)
        t2, te2 = holdout_split(fixture_data.ratings, seed=42)
        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_frame_equal(te1, te2)

    def test_invalid_fraction_raises(self):
        df = pd.DataFrame([
            {"userId": 1, "movieId": 1, "rating": 4.0, "timestamp": 1},
        ])
        with pytest.raises(ValueError):
            holdout_split(df, test_fraction=0.0)
        with pytest.raises(ValueError):
            holdout_split(df, test_fraction=1.0)

    def test_empty_ratings_raises_clear_error(self):
        """Red-team: empty ratings must raise a clear ValueError, not a
        confusing 'No objects to concatenate' from pd.concat([])."""
        empty = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
        with pytest.raises(ValueError, match="ratings"):
            holdout_split(empty, test_fraction=0.2)


class TestRelevantItems:
    def test_threshold_filtering(self):
        df = pd.DataFrame([
            {"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 1},
            {"userId": 1, "movieId": 20, "rating": 3.0, "timestamp": 2},
            {"userId": 2, "movieId": 10, "rating": 3.5, "timestamp": 1},
        ])
        rel = relevant_items(df, threshold=3.5)
        assert rel[1] == {10}
        assert rel[2] == {10}

    def test_no_relevant_items(self):
        df = pd.DataFrame([
            {"userId": 1, "movieId": 10, "rating": 2.0, "timestamp": 1},
        ])
        rel = relevant_items(df, threshold=3.5)
        assert 1 not in rel or rel[1] == set()
