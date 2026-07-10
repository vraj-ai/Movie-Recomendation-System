"""Tests for RMSE/MAE metrics (ticket #4)."""

from __future__ import annotations

import math

import pytest

from recsys.evaluate import rmse, mae, evaluate_recommender


class TestRMSE:
    def test_perfect_predictions(self):
        preds = [3.0, 4.0, 5.0]
        acts = [3.0, 4.0, 5.0]
        assert rmse(preds, acts) == pytest.approx(0.0)

    def test_known_values(self):
        preds = [3.0, 4.0, 5.0]
        acts = [2.0, 5.0, 3.0]
        # errors: 1, -1, 2 → squared: 1, 1, 4 → mean: 2 → sqrt: ~1.414
        assert rmse(preds, acts) == pytest.approx(math.sqrt(2.0))

    def test_single_value(self):
        preds = [4.0]
        acts = [3.0]
        assert rmse(preds, acts) == pytest.approx(1.0)

    def test_empty(self):
        assert rmse([], []) == 0.0

    def test_all_same_error(self):
        preds = [1.0, 2.0, 3.0]
        acts = [2.0, 3.0, 4.0]  # all off by 1
        assert rmse(preds, acts) == pytest.approx(1.0)


class TestMAE:
    def test_perfect_predictions(self):
        preds = [3.0, 4.0, 5.0]
        acts = [3.0, 4.0, 5.0]
        assert mae(preds, acts) == pytest.approx(0.0)

    def test_known_values(self):
        preds = [3.0, 4.0, 5.0]
        acts = [2.0, 5.0, 3.0]
        # |errors|: 1, 1, 2 → mean: 4/3
        assert mae(preds, acts) == pytest.approx(4.0 / 3.0)

    def test_single_value(self):
        preds = [4.0]
        acts = [3.0]
        assert mae(preds, acts) == pytest.approx(1.0)

    def test_empty(self):
        assert mae([], []) == 0.0

    def test_rmse_gte_mae(self):
        """RMSE >= MAE always holds (RMS >= mean of abs)."""
        preds = [1.0, 5.0, 3.0]
        acts = [3.0, 3.0, 3.0]
        assert rmse(preds, acts) >= mae(preds, acts)


class TestEvaluateWithRMSE:
    def test_svd_produces_rmse_mae(self, train_ratings_data, train_test):
        """evaluate_recommender includes rmse/mae for SVD."""
        from recsys.recommenders import SVDRecommender

        _, test_df = train_test
        rec = SVDRecommender(n_components=5)
        rec.fit(train_ratings_data)

        metrics = evaluate_recommender(rec, train_ratings_data, test_df, k=5)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0
        assert metrics["rmse"] >= metrics["mae"]  # RMSE >= MAE invariant

    def test_popularity_no_rmse(self, train_ratings_data, train_test):
        """Popularity (no predict) → no rmse/mae."""
        from recsys.recommenders import PopularityRecommender

        _, test_df = train_test
        rec = PopularityRecommender()
        rec.fit(train_ratings_data)

        metrics = evaluate_recommender(rec, train_ratings_data, test_df, k=5)
        assert "rmse" not in metrics
        assert "mae" not in metrics
