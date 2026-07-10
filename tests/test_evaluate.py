"""Tests for evaluation metrics: Precision@K, Recall@K, NDCG@K (ticket #7)."""

from __future__ import annotations

import math

import pytest

from recsys.evaluate import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    evaluate_recommender,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        recs = [1, 2, 3, 4, 5]
        rel = {1, 2, 3, 4, 5}
        assert precision_at_k(recs, rel, k=5) == 1.0

    def test_none_relevant(self):
        recs = [1, 2, 3, 4, 5]
        rel = {99}
        assert precision_at_k(recs, rel, k=5) == 0.0

    def test_partial(self):
        recs = [1, 2, 3, 4, 5]
        rel = {1, 3, 5}
        # 3 hits out of 5
        assert precision_at_k(recs, rel, k=5) == pytest.approx(0.6)

    def test_k_larger_than_recs(self):
        recs = [1, 2]
        rel = {1, 2}
        # k=5 but only 2 recs → 2/5
        assert precision_at_k(recs, rel, k=5) == pytest.approx(0.4)

    def test_empty_recs(self):
        assert precision_at_k([], {1, 2}, k=5) == 0.0

    def test_k_zero(self):
        assert precision_at_k([1, 2], {1}, k=0) == 0.0


class TestRecallAtK:
    def test_all_relevant_in_top_k(self):
        recs = [1, 2, 3]
        rel = {1, 2, 3}
        assert recall_at_k(recs, rel, k=3) == 1.0

    def test_partial_recall(self):
        recs = [1, 2, 3]
        rel = {1, 2, 3, 4, 5}
        # 3 out of 5 relevant found
        assert recall_at_k(recs, rel, k=3) == pytest.approx(0.6)

    def test_no_relevant(self):
        assert recall_at_k([1, 2], set(), k=5) == 0.0

    def test_empty_recs(self):
        assert recall_at_k([], {1, 2}, k=5) == 0.0


class TestNDCGAtK:
    def test_ideal_ordering(self):
        """All relevant items first → NDCG = 1.0."""
        recs = [1, 2, 3]
        rel = {1, 2, 3}
        assert ndcg_at_k(recs, rel, k=3) == pytest.approx(1.0)

    def test_worst_ordering(self):
        """Relevant items at the end."""
        recs = [4, 5, 1, 2, 3]
        rel = {1, 2, 3}
        # DCG = 1/log2(4) + 1/log2(5) + 1/log2(6)
        # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4)
        dcg = 1 / math.log2(4) + 1 / math.log2(5) + 1 / math.log2(6)
        idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        expected = dcg / idcg
        assert ndcg_at_k(recs, rel, k=5) == pytest.approx(expected)

    def test_no_relevant(self):
        assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0

    def test_empty_recs(self):
        assert ndcg_at_k([], {1, 2}, k=5) == 0.0

    def test_single_relevant_first(self):
        recs = [1, 2, 3]
        rel = {1}
        # DCG = 1/log2(2) = 1.0; IDCG = 1.0 → NDCG = 1.0
        assert ndcg_at_k(recs, rel, k=3) == pytest.approx(1.0)

    def test_single_relevant_last(self):
        recs = [2, 3, 1]
        rel = {1}
        # DCG = 1/log2(4); IDCG = 1/log2(2) = 1.0
        expected = (1 / math.log2(4)) / 1.0
        assert ndcg_at_k(recs, rel, k=3) == pytest.approx(expected)


class TestEvaluateRecommender:
    def test_returns_all_metrics(self, train_ratings_data, train_test):
        """evaluate_recommender returns precision, recall, ndcg."""
        from recsys.recommenders import PopularityRecommender

        _, test_df = train_test
        rec = PopularityRecommender()
        rec.fit(train_ratings_data)

        metrics = evaluate_recommender(rec, train_ratings_data, test_df, k=5)
        assert "precision_at_k" in metrics
        assert "recall_at_k" in metrics
        assert "ndcg_at_k" in metrics
        assert 0.0 <= metrics["precision_at_k"] <= 1.0
        assert 0.0 <= metrics["recall_at_k"] <= 1.0
        assert 0.0 <= metrics["ndcg_at_k"] <= 1.0

    def test_no_rmse_for_non_predictor(self, train_ratings_data, train_test):
        """Popularity (no predict) should not have rmse/mae."""
        from recsys.recommenders import PopularityRecommender

        _, test_df = train_test
        rec = PopularityRecommender()
        rec.fit(train_ratings_data)

        metrics = evaluate_recommender(rec, train_ratings_data, test_df, k=5)
        assert "rmse" not in metrics
        assert "mae" not in metrics
