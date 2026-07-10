"""Tests for the enrichment step (ticket #2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recsys.enrichment import enrich
from recsys.tmdb import TMDBClient, _empty_features
from recsys.config import FIXTURE_TMDB_CACHE


class TestEnrich:
    def test_enrich_returns_dict(self, fixture_data):
        client = TMDBClient(api_key=None, cache_dir=FIXTURE_TMDB_CACHE)
        features = enrich(fixture_data.movie_ids, fixture_data.tmdb_map, client)
        assert isinstance(features, dict)
        assert len(features) == len(fixture_data.movie_ids)

    def test_all_movies_have_features(self, fixture_data, content_features):
        """Every movie gets a features dict (possibly empty)."""
        for mid in fixture_data.movie_ids:
            assert mid in content_features
            feats = content_features[mid]
            for key in ("genres", "keywords", "cast", "crew", "overview"):
                assert key in feats

    def test_known_movie_has_content(self, content_features):
        """Movie 1 (Toy Story, tmdbId=862) has content from cache."""
        feats = content_features[1]
        assert len(feats["genres"]) > 0
        assert "Animation" in feats["genres"]
        assert len(feats["cast"]) > 0
        assert "Tom Hanks" in feats["cast"]
        assert len(feats["overview"]) > 0

    def test_no_tmdb_id_gets_empty_features(self):
        """FAIL-5: movie with no tmdbId → empty features."""
        client = TMDBClient(api_key=None, cache_dir=Path("/tmp/nonexistent_cache"))
        features = enrich([1, 2, 3], {}, client)  # empty tmdb_map
        for mid in (1, 2, 3):
            assert features[mid] == _empty_features()

    def test_tmdb_fetch_failure_gets_empty_features(self, tmp_path):
        """FAIL-5: TMDB fetch failure → empty features."""
        # Cache-only mode with no cache → all returns None
        client = TMDBClient(api_key=None, cache_dir=tmp_path / "empty_cache")
        features = enrich([1], {1: 99999}, client)
        assert features[1] == _empty_features()

    def test_enrichment_with_pre_seeded_cache(self, fixture_data, content_features):
        """Fixture cache provides content for all 10 movies."""
        n_with_content = sum(
            1 for v in content_features.values()
            if v.get("genres") or v.get("overview")
        )
        assert n_with_content == 10  # all fixture movies have cache entries

    def test_features_contain_expected_fields(self, content_features):
        """Features have the 5 expected fields with correct types."""
        for mid, feats in content_features.items():
            assert isinstance(feats["genres"], list)
            assert isinstance(feats["keywords"], list)
            assert isinstance(feats["cast"], list)
            assert isinstance(feats["crew"], list)
            assert isinstance(feats["overview"], str)
