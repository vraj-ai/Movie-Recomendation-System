"""W5 — Ratings + profile APIs (#18): mutations recompute, pin semantics.

Invariants: MATH-4 (pinned wins outright, no blend), WEB-FAIL-2 (bad input
-> 4xx never 500), WEB-FAIL-3 (corrupt store quarantined, not fatal),
WEB-PERF-3 (mutation -> fresh profile in the response, no stale reads).

Verification-command: .venv/bin/pytest tests/test_web_profile.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recsys.web.app import create_app


@pytest.fixture()
def profile_path(tmp_path) -> Path:
    return tmp_path / "taste_profile.json"


@pytest.fixture()
def client(profile_path):
    app = create_app(mode="fixture", profile_path=profile_path)
    return TestClient(app)


def _put_rating(client, movie_id, rating):
    return client.put(
        "/api/ratings", json={"movie_id": movie_id, "rating": rating}
    )


# -- ratings CRUD ---------------------------------------------------------


def test_put_rating_persists_and_lists(client):
    assert _put_rating(client, 1, 4.5).status_code == 200
    r = client.get("/api/ratings")
    assert r.status_code == 200
    ratings = r.json()["ratings"]
    assert ratings == [
        {"movie_id": 1, "title": ratings[0]["title"], "rating": 4.5}
    ]
    assert ratings[0]["title"]  # human-readable, not just an id


def test_put_rating_updates_existing(client):
    _put_rating(client, 1, 2.0)
    _put_rating(client, 1, 5.0)
    ratings = client.get("/api/ratings").json()["ratings"]
    assert len(ratings) == 1
    assert ratings[0]["rating"] == 5.0


def test_delete_rating_removes(client):
    _put_rating(client, 1, 4.0)
    _put_rating(client, 2, 3.0)
    r = client.delete("/api/ratings", params={"movie_id": 1})
    assert r.status_code == 200
    ids = [x["movie_id"] for x in client.get("/api/ratings").json()["ratings"]]
    assert ids == [2]


def test_delete_unrated_is_404(client):
    assert client.delete("/api/ratings", params={"movie_id": 1}).status_code == 404


def test_ratings_survive_restart(client, profile_path):
    _put_rating(client, 1, 4.5)
    fresh = TestClient(create_app(mode="fixture", profile_path=profile_path))
    assert fresh.get("/api/ratings").json()["ratings"][0]["rating"] == 4.5


# -- WEB-FAIL-2: validation, 4xx never 500 --------------------------------


@pytest.mark.parametrize("payload", [
    {"movie_id": 1, "rating": 0.0},      # below scale
    {"movie_id": 1, "rating": 5.5},      # above scale
    {"movie_id": 1, "rating": 3.25},     # not a half step
    {"movie_id": 1, "rating": "great"},  # wrong type
    {"movie_id": 1},                     # missing rating
    {"rating": 3.0},                     # missing movie_id
])
def test_put_rating_bad_input_is_4xx(client, payload):
    r = client.put("/api/ratings", json=payload)
    assert 400 <= r.status_code < 500


def test_put_rating_unknown_movie_is_404(client):
    assert _put_rating(client, 999999, 3.0).status_code == 404


@pytest.mark.parametrize("payload", [
    {"slider": "bogus", "value": 0.5},        # unknown slider id
    {"slider": "genre:Action", "value": 2.0},  # out of [-1, 1]
    {"slider": "pop", "value": 1.5},           # out of [0, 1]
    {"slider": "era", "value": 123.0},         # not a plausible decade
    {"value": 0.5},                            # missing slider
])
def test_put_pin_bad_input_is_4xx(client, payload):
    r = client.put("/api/profile/pins", json=payload)
    assert 400 <= r.status_code < 500


# -- WEB-PERF-3: mutation responses carry the recomputed profile ----------


def test_mutation_response_contains_fresh_profile(client):
    r = _put_rating(client, 1, 5.0)
    body = r.json()
    assert body["profile"]["n_ratings"] == 1
    assert body["profile"]["cold_start"] is False
    assert body["profile"]["sliders"]  # derived weights already present

    r2 = client.delete("/api/ratings", params={"movie_id": 1})
    assert r2.json()["profile"]["n_ratings"] == 0
    assert r2.json()["profile"]["cold_start"] is True


# -- GET /api/profile: derived + effective + pin state --------------------


def test_profile_shows_derived_sliders(client):
    _put_rating(client, 1, 5.0)
    sliders = client.get("/api/profile").json()["sliders"]
    assert "era" in sliders and "pop" in sliders
    genre_sliders = [s for s in sliders if s.startswith("genre:")]
    assert genre_sliders
    for s in sliders.values():
        assert set(s) >= {"value", "derived", "pinned"}
        assert s["pinned"] is False
        assert s["value"] == s["derived"]  # nothing pinned yet


# -- MATH-4: pin wins outright, reset returns to derived ------------------


def test_pin_wins_outright_and_derived_is_kept(client):
    _put_rating(client, 1, 5.0)
    sliders = client.get("/api/profile").json()["sliders"]
    sid = next(s for s in sliders if s.startswith("genre:"))
    derived = sliders[sid]["derived"]
    pinned_value = -1.0 if derived != -1.0 else 1.0

    r = client.put("/api/profile/pins", json={"slider": sid, "value": pinned_value})
    assert r.status_code == 200
    s = r.json()["profile"]["sliders"][sid]
    assert s["pinned"] is True
    assert s["value"] == pinned_value          # pinned wins outright, no blend
    assert s["derived"] == pytest.approx(derived)  # derived still visible


def test_unpin_resets_to_derived(client):
    _put_rating(client, 1, 5.0)
    sid = next(
        s for s in client.get("/api/profile").json()["sliders"]
        if s.startswith("genre:")
    )
    client.put("/api/profile/pins", json={"slider": sid, "value": 1.0})
    r = client.put("/api/profile/pins", json={"slider": sid, "value": None})
    s = r.json()["profile"]["sliders"][sid]
    assert s["pinned"] is False
    assert s["value"] == s["derived"]


def test_pins_change_recommendations(client):
    """Pinning a genre re-ranks: the effective profile actually feeds W4."""
    _put_rating(client, 1, 5.0)
    before = client.get("/api/recommend", params={"n": 5}).json()["results"]
    sid = next(
        s for s in client.get("/api/profile").json()["sliders"]
        if s.startswith("genre:")
    )
    client.put("/api/profile/pins", json={"slider": sid, "value": -1.0})
    after = client.get("/api/recommend", params={"n": 5}).json()["results"]
    assert [b["score"] for b in before] != [a["score"] for a in after]


# -- WEB-FAIL-3: corrupt store quarantined, API keeps answering -----------


def test_corrupt_profile_is_quarantined_not_fatal(client, profile_path):
    _put_rating(client, 1, 4.0)
    profile_path.write_text("{not json!!")
    r = client.get("/api/profile")
    assert r.status_code == 200
    assert r.json()["n_ratings"] == 0  # quarantined -> empty, not a crash
    quarantined = list(profile_path.parent.glob("*.corrupt-*"))
    assert quarantined  # the corrupt file was preserved, not destroyed
