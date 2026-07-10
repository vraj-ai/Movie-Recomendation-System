"""Tests for the HybridRecommender (ticket #5)."""

from __future__ import annotations

import pytest

from recsys.recommenders import HybridRecommender, ContentRecommender, SVDRecommender
from recsys.interfaces import Recommender


class TestHybridRecommender:
    def test_implements_interface(self):
        assert issubclass(HybridRecommender, Recommender)

    def test_name(self):
        assert HybridRecommender().name == "hybrid"

    def test_fit(self, train_ratings_data, content_features):
        rec = HybridRecommender(n_components=5)
        rec.fit(train_ratings_data, content_features=content_features)
        assert rec._train is not None
        assert rec._content._tfidf_matrix is not None
        assert rec._svd._reconstructed is not None

    def test_recommend_returns_list(self, train_ratings_data, content_features):
        rec = HybridRecommender(n_components=5)
        rec.fit(train_ratings_data, content_features=content_features)
        user_id = train_ratings_data.user_ids[0]
        result = rec.recommend(user_id, n=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_recommend_excludes_seen(self, train_ratings_data, content_features):
        rec = HybridRecommender(n_components=5)
        rec.fit(train_ratings_data, content_features=content_features)
        user_id = train_ratings_data.user_ids[0]
        seen = train_ratings_data.rated_by_user(user_id)
        result = rec.recommend(user_id, n=10)
        for mid in result:
            assert mid not in seen

    def test_predict_delegates_to_svd(self, train_ratings_data, content_features):
        """Hybrid.predict delegates to SVD."""
        rec = HybridRecommender(n_components=5)
        rec.fit(train_ratings_data, content_features=content_features)
        user_id = train_ratings_data.user_ids[0]
        movie_id = train_ratings_data.movie_ids[0]
        pred = rec.predict(user_id, movie_id)
        assert pred is not None
        assert isinstance(pred, float)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            HybridRecommender(content_weight=-0.1)
        with pytest.raises(ValueError):
            HybridRecommender(content_weight=1.1)

    def test_weight_extremes(self, train_ratings_data, content_features):
        """weight=0 → pure CF, weight=1 → pure content."""
        user_id = train_ratings_data.user_ids[0]
        n = 5

        # Pure CF
        rec_cf = HybridRecommender(content_weight=0.0, n_components=5)
        rec_cf.fit(train_ratings_data, content_features=content_features)
        cf_result = rec_cf.recommend(user_id, n=n)

        # Pure content
        rec_content = HybridRecommender(content_weight=1.0, n_components=5)
        rec_content.fit(train_ratings_data, content_features=content_features)
        content_result = rec_content.recommend(user_id, n=n)

        # Both should return valid results
        assert len(cf_result) > 0
        assert len(content_result) > 0

    def test_degrades_without_content(self, train_ratings_data):
        """FAIL-5: hybrid degrades correctly when content features are absent."""
        empty_features = {mid: {
            "genres": [], "keywords": [], "cast": [], "crew": [], "overview": ""
        } for mid in train_ratings_data.movie_ids}
        rec = HybridRecommender(n_components=5)
        rec.fit(train_ratings_data, content_features=empty_features)
        user_id = train_ratings_data.user_ids[0]
        result = rec.recommend(user_id, n=5)
        assert isinstance(result, list)  # no crash

    def test_recommend_before_fit(self):
        rec = HybridRecommender()
        assert rec.recommend(1, n=5) == []

    def test_unknown_user(self, train_ratings_data, content_features):
        rec = HybridRecommender(n_components=5)
        rec.fit(train_ratings_data, content_features=content_features)
        assert rec.recommend(99999, n=5) == []

    def test_blended_result_differs_from_pure(self, train_ratings_data, content_features):
        """A 50/50 blend should generally differ from pure CF or pure content."""
        user_id = train_ratings_data.user_ids[0]
        n = 5

        rec_blend = HybridRecommender(content_weight=0.5, n_components=5)
        rec_blend.fit(train_ratings_data, content_features=content_features)
        blended = rec_blend.recommend(user_id, n=n)

        rec_cf = HybridRecommender(content_weight=0.0, n_components=5)
        rec_cf.fit(train_ratings_data, content_features=content_features)
        cf = rec_cf.recommend(user_id, n=n)

        rec_ct = HybridRecommender(content_weight=1.0, n_components=5)
        rec_ct.fit(train_ratings_data, content_features=content_features)
        ct = rec_ct.recommend(user_id, n=n)

        # At least one of the pure approaches should differ from the blend
        # (they can't all be identical unless content and CF perfectly agree)
        assert blended != cf or blended != ct


class TestRefitCacheStaleness:
    """Red-team: hybrid must not serve stale per-user content scores after a refit."""

    def _build_rd(self, ratings_df, movies, links, movie_subset):
        import scipy.sparse as sp
        import numpy as np
        from recsys.data import RatingsData

        sub = ratings_df[ratings_df["movieId"].isin(movie_subset)]
        mids = sorted(sub["movieId"].unique())
        uids = sorted(sub["userId"].unique())
        m2i = {m: i for i, m in enumerate(mids)}
        u2i = {u: i for i, u in enumerate(uids)}
        r = sub["userId"].map(u2i).values
        c = sub["movieId"].map(m2i).values
        v = sub["rating"].values.astype(np.float64)
        ui = sp.csr_matrix((v, (r, c)), shape=(len(uids), len(mids)))
        return RatingsData(
            ratings=sub, movies=movies, links=links, user_item=ui,
            movie_ids=mids, user_ids=uids, movie_id_to_idx=m2i, user_id_to_idx=u2i,
        )

    def test_refit_with_smaller_pool_does_not_crash(
        self, train_ratings_data, content_features
    ):
        """Refitting on a smaller movie pool must not reuse stale cached scores
        of the wrong length (shape-mismatch crash)."""
        rec = HybridRecommender(n_components=5)
        rec.fit(train_ratings_data, content_features=content_features)
        user_id = train_ratings_data.user_ids[0]
        first = rec.recommend(user_id, n=5)
        assert isinstance(first, list)

        # Refit on a strict subset — fewer movies, different shape
        subset_ids = train_ratings_data.movie_ids[:5]
        sub_rd = self._build_rd(
            train_ratings_data.ratings, train_ratings_data.movies,
            train_ratings_data.links, subset_ids,
        )
        sub_cf = {k: content_features[k] for k in subset_ids if k in content_features}
        rec.fit(sub_rd, content_features=sub_cf)

        # Must not crash with a broadcast error from a stale cache entry
        result = rec.recommend(user_id, n=5)
        assert isinstance(result, list)
        # All results must come from the NEW (smaller) pool only
        for mid in result:
            assert mid in sub_rd.movie_ids


class TestFourWayTable:
    def test_all_four_in_run(self, tmp_path):
        """run --fixture produces a comparison table with all four approaches."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "recsys.cli", "run", "--fixture", "--out", str(tmp_path)],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"run failed:\n{result.stderr}"

        md = (tmp_path / "comparison.md").read_text()
        assert "popularity" in md.lower()
        assert "content" in md.lower()
        assert "svd" in md.lower()
        assert "hybrid" in md.lower()

        csv_text = (tmp_path / "comparison.csv").read_text()
        lines = csv_text.strip().split("\n")
        assert len(lines) >= 5  # header + 4 approaches
