"""W4 — Recommend + Similar APIs (#17): breakdowns wired to the engine.

Invariants: MATH-1..4, WEB-FAIL-2 (bad input -> 4xx never 500),
WEB-FAIL-5 (cold-start responses labeled), WEB-PERF-2 (< 1 s warm).

Verification-command: .venv/bin/pytest tests/test_web_recommend.py -q
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recsys.config import (
    TASTE_FEATURE_ERA_W,
    TASTE_FEATURE_GENRE_W,
    TASTE_FEATURE_POP_W,
    TASTE_SCORE_FEATURE_W,
    TASTE_SCORE_NEIGHBOR_W,
    TASTE_SCORE_SCALE,
)
from recsys.web.app import create_app
from recsys.web.profile_store import ProfileStore

# Ratings chosen from fixture movie ids (1..N) with a clear genre signal.
SEED_RATINGS = {1: 5.0, 2: 4.5, 3: 1.0, 5: 0.5}

APPROX = pytest.approx


def _make_client(tmp_path: Path, ratings: dict[int, float]) -> TestClient:
    profile = tmp_path / "taste_profile.json"
    ProfileStore(profile).save(ratings={k: v for k, v in ratings.items()}, pins={})
    app = create_app(mode="fixture", profile_path=profile)
    return TestClient(app)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return _make_client(tmp_path_factory.mktemp("w4"), SEED_RATINGS)


@pytest.fixture(scope="module")
def cold_client(tmp_path_factory):
    """No ratings at all — cold start."""
    profile = tmp_path_factory.mktemp("w4cold") / "taste_profile.json"
    app = create_app(mode="fixture", profile_path=profile)
    return TestClient(app)


# -- /api/recommend: shape and exclusions --------------------------------


def test_recommend_returns_n_results(client):
    r = client.get("/api/recommend", params={"n": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["cold_start"] is False
    assert len(body["results"]) == 5


def test_recommend_excludes_rated_movies(client):
    r = client.get("/api/recommend", params={"n": 20})
    ids = {item["movie_id"] for item in r.json()["results"]}
    assert ids.isdisjoint(SEED_RATINGS.keys())


def test_recommend_sorted_by_score_then_id(client):
    res = client.get("/api/recommend", params={"n": 20}).json()["results"]
    keys = [(-item["score"], item["movie_id"]) for item in res]
    assert keys == sorted(keys)


# -- MATH-2: every breakdown term present and arithmetically consistent --


def test_breakdown_terms_present_and_consistent(client):
    res = client.get("/api/recommend", params={"n": 10}).json()["results"]
    assert res
    for item in res:
        fm = (
            TASTE_FEATURE_GENRE_W * item["genre_affinity"]
            + TASTE_FEATURE_ERA_W * item["era_match"]
            + TASTE_FEATURE_POP_W * item["pop_match"]
        )
        assert item["feature_match"] == APPROX(fm)
        assert item["feature_term"] == APPROX(
            TASTE_SCORE_SCALE * TASTE_SCORE_FEATURE_W * item["feature_match"]
        )
        assert item["neighbor_term"] == APPROX(
            TASTE_SCORE_SCALE * TASTE_SCORE_NEIGHBOR_W * item["neighbor_evidence"]
        )
        assert item["score"] == APPROX(item["feature_term"] + item["neighbor_term"])


def test_neighbor_citations_are_the_rated_movies(client):
    res = client.get("/api/recommend", params={"n": 5}).json()["results"]
    for item in res:
        cited = {t["movie_id"] for t in item["neighbor_terms"]}
        assert cited <= set(SEED_RATINGS.keys())
        assert cited  # evidence must actually cite the operator's ratings
        for t in item["neighbor_terms"]:
            assert t["rating"] == SEED_RATINGS[t["movie_id"]]
            assert set(t) >= {
                "movie_id", "title", "rating", "signal",
                "similarity", "contribution",
            }
        # neighbor_evidence is exactly the sum of the cited contributions
        assert item["neighbor_evidence"] == APPROX(
            sum(t["contribution"] for t in item["neighbor_terms"])
        )


# -- WEB-FAIL-5: cold start labeled, never a silent empty ----------------


def test_cold_start_is_labeled(cold_client):
    r = cold_client.get("/api/recommend", params={"n": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["cold_start"] is True
    for item in body["results"]:
        assert item["cold_start"] is True
        assert item["neighbor_terms"] == []


# -- WEB-FAIL-2: bad query input answers 4xx, never 500 ------------------


@pytest.mark.parametrize("params", [
    {"n": 0},
    {"n": -3},
    {"n": "ten"},
    {"n": 10_000},
])
def test_recommend_bad_n_is_4xx(client, params):
    r = client.get("/api/recommend", params=params)
    assert 400 <= r.status_code < 500


@pytest.mark.parametrize("params", [
    {},                              # movie_id required
    {"movie_id": "abc"},
    {"movie_id": 999999},            # unknown id -> 404
    {"movie_id": 1, "n": 0},
    {"movie_id": 1, "n": "x"},
])
def test_similar_bad_input_is_4xx(client, params):
    r = client.get("/api/similar", params=params)
    assert 400 <= r.status_code < 500


# -- /api/similar: similarity terms per result ---------------------------


def test_similar_returns_terms_and_excludes_anchor(client):
    r = client.get("/api/similar", params={"movie_id": 1, "n": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["movie_id"] == 1
    assert body["results"]
    assert all(item["movie_id"] != 1 for item in body["results"])
    for item in body["results"]:
        assert set(item) >= {
            "movie_id", "title", "genre_jaccard",
            "era_proximity", "pop_proximity", "similarity",
        }
    keys = [(-item["similarity"], item["movie_id"]) for item in body["results"]]
    assert keys == sorted(keys)


def test_similar_is_deterministic(client):
    a = client.get("/api/similar", params={"movie_id": 1, "n": 10}).json()
    b = client.get("/api/similar", params={"movie_id": 1, "n": 10}).json()
    assert a == b


# -- MATH-3: serving path stays model-free -------------------------------


def test_recommend_module_imports_no_recommenders():
    import recsys.web.app as web_app

    src = Path(web_app.__file__).read_text()
    assert "recsys.recommenders" not in src.replace(
        "never recsys.recommenders", ""
    )


# -- WEB-PERF-2: recommend < 1 s warm on fixture -------------------------


def test_recommend_under_1s_warm(client):
    client.get("/api/recommend", params={"n": 10})  # warm
    t0 = time.perf_counter()
    r = client.get("/api/recommend", params={"n": 10})
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    assert elapsed < 1.0, f"recommend took {elapsed:.3f}s warm"
