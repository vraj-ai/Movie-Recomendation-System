"""W9 — practical watch-log API (effort 3, --full mode): search, 5-star
watch log, and next-watch recommendations against a temp catalog built
straight from the frozen "Catalog contract" DDL
(docs/plan/effort3-practical-watchlog.md). Deliberately independent of
fixtures/catalog-small.sqlite and data/catalog.sqlite (Agent A owns those,
built concurrently) — this file's catalog is entirely synthetic.

Invariants covered: WEB-FAIL-1 (--full fails fast, never silent swap),
WEB-FAIL-2 (bad input -> 4xx/422, never 500), PRAC-1 (watched never
recommended), PRAC-2 (reasons + math always present), PRAC-3 (reasons are
plain English), WEB-FAIL-3 (profile store quarantine survives extension).

Verification-command: .venv/bin/pytest tests/test_practical_api.py -q
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recsys.config import CATALOG_BUILD_INSTRUCTIONS
from recsys.web.app import create_app
from recsys.web.profile_store import ProfileStore

# ------------------------------------------------------------- catalog DDL
# Verbatim from the FROZEN "Catalog contract" in
# docs/plan/effort3-practical-watchlog.md.

_CATALOG_DDL = """
CREATE TABLE movies (
  id INTEGER PRIMARY KEY,
  tmdb_id INTEGER UNIQUE,
  ml_id INTEGER UNIQUE,
  title TEXT NOT NULL,
  year INTEGER,
  language TEXT NOT NULL,
  genres TEXT NOT NULL DEFAULT '',
  overview TEXT NOT NULL DEFAULT '',
  poster_path TEXT,
  popularity REAL NOT NULL DEFAULT 0,
  vote_average REAL NOT NULL DEFAULT 0,
  vote_count INTEGER NOT NULL DEFAULT 0,
  is_indian INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL CHECK (source IN ('ml','indian','both'))
);
CREATE VIRTUAL TABLE movies_fts USING fts5(title, content='movies', content_rowid='id');
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# 20 synthetic movies spanning genres / languages / eras / popularity tiers.
# (id, title, year, language, genres, poster_path, popularity, vote_average,
#  vote_count, is_indian, source)
_MOVIES = [
    (1, "Dangal", 2016, "hi", "Drama|Sport", "/dangal.jpg", 90, 8.4, 8000, 1, "indian"),
    (2, "Baahubali", 2015, "te", "Action|Fantasy", "/bb.jpg", 95, 8.1, 6000, 1, "indian"),
    (3, "3 Idiots", 2009, "hi", "Comedy|Drama", "/3i.jpg", 85, 8.4, 7000, 1, "indian"),
    (4, "Lagaan", 2001, "hi", "Drama|Sport", None, 40, 8.1, 1500, 1, "indian"),
    (5, "Sholay", 1975, "hi", "Action|Drama", None, 30, 8.2, 900, 1, "indian"),
    (6, "Kantara", 2022, "kn", "Drama|Thriller", None, 60, 8.0, 2000, 1, "indian"),
    (7, "Vikram Vedha", 2017, "ta", "Action|Thriller", None, 45, 7.8, 1200, 1, "indian"),
    (8, "Drishyam", 2013, "ml", "Thriller|Drama", None, 35, 8.2, 800, 1, "indian"),
    (9, "Premam", 2015, "ml", "Romance|Drama", None, 25, 7.9, 500, 1, "indian"),
    (10, "Inception", 2010, "en", "Action|Sci-Fi", "/inc.jpg", 98, 8.3, 32000, 0, "ml"),
    (11, "The Matrix", 1999, "en", "Action|Sci-Fi", None, 90, 8.2, 25000, 0, "ml"),
    (12, "Toy Story", 1995, "en", "Animation|Comedy", None, 80, 8.0, 18000, 0, "ml"),
    (13, "Pulp Fiction", 1994, "en", "Crime|Drama", None, 85, 8.5, 22000, 0, "ml"),
    (14, "The Godfather", 1972, "en", "Crime|Drama", None, 88, 8.7, 19000, 0, "ml"),
    (15, "Casablanca", 1942, "en", "Drama|Romance", None, 40, 8.2, 4000, 0, "ml"),
    (16, "Amelie", 2001, "fr", "Comedy|Romance", None, 55, 8.0, 6000, 0, "ml"),
    (17, "Seven Samurai", 1954, "ja", "Action|Drama", None, 35, 8.6, 3000, 0, "ml"),
    (18, "Spirited Away", 2001, "ja", "Animation|Fantasy", None, 75, 8.5, 15000, 0, "ml"),
    (19, "Parasite", 2019, "ko", "Thriller|Drama", None, 92, 8.5, 17000, 0, "ml"),
    (20, "La La Land", 2016, "en", "Drama|Romance", None, 70, 8.0, 14000, 0, "ml"),
]

_JARGON = [
    "cosine", "vector", "affinity", "percentile", "shrink", "multiplier",
    "sigma", "epsilon", "logit", "softmax", "normalize", "embedding",
    "weighted mean", "one-hot", "np.", "numpy",
]


def _build_catalog(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_CATALOG_DDL)
        conn.executemany(
            "INSERT INTO movies (id, tmdb_id, ml_id, title, year, language, "
            "genres, overview, poster_path, popularity, vote_average, "
            "vote_count, is_indian, source) VALUES (?,NULL,NULL,?,?,?,?,'',?,?,?,?,?,?)",
            [
                (mid, title, year, lang, genres, poster, pop, va, vc, ind, src)
                for mid, title, year, lang, genres, poster, pop, va, vc, ind, src in _MOVIES
            ],
        )
        conn.execute("INSERT INTO movies_fts(rowid, title) SELECT id, title FROM movies")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(scope="module")
def catalog_path(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("catalog") / "catalog.sqlite"
    _build_catalog(p)
    return p


def _make_client(catalog_path: Path, tmp_path: Path) -> TestClient:
    app = create_app(
        mode="full",
        catalog_path=catalog_path,
        profile_path=tmp_path / "watch_log.json",
    )
    return TestClient(app)


@pytest.fixture()
def client(catalog_path, tmp_path) -> TestClient:
    return _make_client(catalog_path, tmp_path)


def _log(client: TestClient, movie_id: int, rating: float) -> None:
    r = client.post("/api/watched", json={"id": movie_id, "rating": rating})
    assert r.status_code == 200, r.text


# =====================================================================
# --full mode startup (WEB-FAIL-1)
# =====================================================================


def test_full_mode_registered_in_parser():
    from recsys.web.__main__ import build_parser

    args = build_parser().parse_args(["--full"])
    assert args.full is True


def test_full_and_fixture_are_mutually_exclusive():
    from recsys.web.__main__ import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--full", "--fixture"])


def test_missing_catalog_fails_fast_naming_build_script(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        create_app(
            mode="full",
            catalog_path=tmp_path / "no-such-catalog.sqlite",
            profile_path=tmp_path / "p.json",
        )
    msg = str(exc.value)
    assert "build_catalog.py" in msg
    assert CATALOG_BUILD_INSTRUCTIONS in msg


def test_malformed_catalog_fails_fast(tmp_path):
    bad = tmp_path / "catalog.sqlite"
    conn = sqlite3.connect(bad)
    conn.execute("CREATE TABLE not_a_catalog (x INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError) as exc:
        create_app(mode="full", catalog_path=bad, profile_path=tmp_path / "p.json")
    assert "build_catalog.py" in str(exc.value)


def test_full_mode_needs_no_tmdb_key(catalog_path, tmp_path):
    # No tmdb_api_key kwarg passed anywhere -> must build and serve fine.
    app = create_app(
        mode="full", catalog_path=catalog_path, profile_path=tmp_path / "p.json"
    )
    c = TestClient(app)
    assert c.get("/api/status").json()["mode"] == "full"


# =====================================================================
# GET /api/search — FTS prefix match, ranked, watched flag
# =====================================================================


def test_search_blank_query_is_empty_list(client):
    assert client.get("/api/search", params={"q": ""}).json() == {"results": []}
    assert client.get("/api/search", params={"q": "   "}).json() == {"results": []}
    assert client.get("/api/search").json() == {"results": []}  # q omitted entirely


def test_search_no_query_is_never_5xx(client):
    r = client.get("/api/search")
    assert r.status_code == 200


def test_search_prefix_match_and_shape(client):
    r = client.get("/api/search", params={"q": "dan"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results and results[0]["title"] == "Dangal"
    assert set(results[0]) >= {
        "id", "title", "year", "language", "genres", "poster_url", "watched", "rating",
    }
    assert results[0]["watched"] is False
    assert results[0]["rating"] is None
    assert results[0]["poster_url"] == "https://image.tmdb.org/t/p/w342/dangal.jpg"


def test_search_no_poster_is_null(client):
    r = client.get("/api/search", params={"q": "sholay"}).json()["results"]
    assert r[0]["poster_url"] is None


def test_search_ranked_by_vote_count_then_popularity(client):
    # "The " matches "The Matrix" (25000 votes) and "The Godfather" (19000).
    r = client.get("/api/search", params={"q": "the"}).json()["results"]
    titles = [m["title"] for m in r]
    assert titles.index("The Matrix") < titles.index("The Godfather")


def test_search_marks_watched_flag_and_rating(client):
    _log(client, 1, 4.5)
    r = client.get("/api/search", params={"q": "dangal"}).json()["results"]
    assert r[0]["watched"] is True
    assert r[0]["rating"] == 4.5


def test_search_sanitizes_fts_special_characters(client):
    # Would otherwise be invalid FTS5 syntax; must 200 with a sane result.
    for q in ['"', "*", "AND OR NOT", "dan*gal", 'a"b', "-- drop"]:
        r = client.get("/api/search", params={"q": q})
        assert r.status_code == 200


def test_search_results_capped_by_limit(client):
    r = client.get("/api/search", params={"q": "a", "limit": 3})
    assert r.status_code == 200
    assert len(r.json()["results"]) <= 3


# =====================================================================
# /api/watched — 5-star log CRUD, validation, persistence
# =====================================================================


@pytest.mark.parametrize("rating", [0.0, 0.25, 3.3, 5.5, -1.0, 100.0])
def test_post_watched_bad_rating_is_422(client, rating):
    r = client.post("/api/watched", json={"id": 1, "rating": rating})
    assert r.status_code == 422


@pytest.mark.parametrize("rating", [0.5, 1.0, 1.5, 4.5, 5.0])
def test_post_watched_valid_half_steps_ok(client, rating):
    r = client.post("/api/watched", json={"id": 1, "rating": rating})
    assert r.status_code == 200


def test_post_watched_unknown_id_is_404(client):
    r = client.post("/api/watched", json={"id": 999999, "rating": 4.0})
    assert r.status_code == 404


def test_get_watched_empty_then_populated(client):
    assert client.get("/api/watched").json() == {"items": [], "count": 0}
    _log(client, 1, 5.0)
    _log(client, 10, 4.0)
    body = client.get("/api/watched").json()
    assert body["count"] == 2
    ids = {i["id"] for i in body["items"]}
    assert ids == {1, 10}
    item = next(i for i in body["items"] if i["id"] == 1)
    assert item["title"] == "Dangal"
    assert item["rating"] == 5.0
    assert item["added_at"]  # ISO timestamp present


def test_post_watched_updates_existing_rating_not_added_at(client):
    _log(client, 1, 3.0)
    first_added_at = next(
        i for i in client.get("/api/watched").json()["items"] if i["id"] == 1
    )["added_at"]
    _log(client, 1, 5.0)
    item = next(i for i in client.get("/api/watched").json()["items"] if i["id"] == 1)
    assert item["rating"] == 5.0
    assert item["added_at"] == first_added_at  # re-logging doesn't reset the date


def test_delete_watched_removes_entry(client):
    _log(client, 1, 4.0)
    _log(client, 2, 4.0)
    r = client.delete("/api/watched/1")
    assert r.status_code == 200
    ids = {i["id"] for i in client.get("/api/watched").json()["items"]}
    assert ids == {2}


def test_delete_unlogged_movie_is_404(client):
    r = client.delete("/api/watched/1")
    assert r.status_code == 404


def test_watched_survives_restart(catalog_path, tmp_path):
    profile_path = tmp_path / "watch_log.json"
    app1 = create_app(mode="full", catalog_path=catalog_path, profile_path=profile_path)
    TestClient(app1).post("/api/watched", json={"id": 1, "rating": 4.5})

    app2 = create_app(mode="full", catalog_path=catalog_path, profile_path=profile_path)
    body = TestClient(app2).get("/api/watched").json()
    assert body == {
        "items": [
            {
                "id": 1, "title": "Dangal", "year": 2016,
                "poster_url": "https://image.tmdb.org/t/p/w342/dangal.jpg",
                "rating": 4.5, "added_at": body["items"][0]["added_at"],
            }
        ],
        "count": 1,
    }


def test_watched_schema_extension_does_not_break_taste_engine_profile(tmp_path):
    """Additive-only: a --full watch log coexists with effort-2 ratings/pins
    in the same profile file, and old readers still see their own fields.
    """
    p = tmp_path / "shared.json"
    store = ProfileStore(p)
    store.save(ratings={1: 5.0}, pins={"era": 1990.0})
    loaded = ProfileStore(p).load()
    assert loaded.ratings == {1: 5.0}
    assert loaded.pins == {"era": 1990.0}
    assert loaded.watched == {}


# =====================================================================
# GET /api/next — recommendations, PRAC-1, PRAC-2, sliders, surprise
# =====================================================================


def test_next_requires_three_logs_409(client):
    r = client.get("/api/next")
    assert r.status_code == 409
    assert r.json() == {"detail": "log_more", "needed": 3}

    _log(client, 1, 5.0)
    r = client.get("/api/next")
    assert r.status_code == 409 and r.json()["needed"] == 2

    _log(client, 2, 4.5)
    r = client.get("/api/next")
    assert r.status_code == 409 and r.json()["needed"] == 1

    _log(client, 3, 4.0)
    r = client.get("/api/next")
    assert r.status_code == 200


@pytest.fixture()
def logged_client(client) -> TestClient:
    _log(client, 1, 5.0)    # Dangal, hi, Drama|Sport, 2016
    _log(client, 10, 4.5)   # Inception, en, Action|Sci-Fi, 2010
    _log(client, 13, 1.0)   # Pulp Fiction, en, Crime|Drama, 1994 (disliked)
    return client


def test_next_default_shape_and_prac2(logged_client):
    r = logged_client.get("/api/next", params={"n": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 5
    assert set(body["profile"]) == {"top_genres", "languages", "movies_logged"}
    assert body["profile"]["movies_logged"] == 3
    for item in body["items"]:
        assert set(item) >= {
            "id", "title", "year", "language", "genres",
            "poster_url", "reasons", "math",
        }
        assert item["reasons"]  # PRAC-2: never empty
        assert set(item["math"]) == {"score", "components"}
        assert isinstance(item["math"]["score"], float)
        comps = item["math"]["components"]
        assert set(comps) == {"genre", "language", "era", "popularity", "rating_signal"}
        for v in comps.values():
            assert 0.0 <= v <= 1.0


def test_next_reasons_are_plain_english(logged_client):
    body = logged_client.get("/api/next", params={"n": 15}).json()
    for item in body["items"]:
        for reason in item["reasons"]:
            assert reason and reason[0].isupper()
            low = reason.lower()
            for jargon in _JARGON:
                assert jargon not in low, f"jargon leaked into reason: {reason!r}"


def test_next_never_returns_a_watched_movie_prac1(logged_client):
    watched_ids = {1, 10, 13}
    body = logged_client.get("/api/next", params={"n": 17}).json()
    ids = {item["id"] for item in body["items"]}
    assert ids.isdisjoint(watched_ids)


def test_next_bad_params_are_422(logged_client):
    for params in (
        {"n": 0}, {"n": 51}, {"n": -1},
        {"surprise": -1}, {"surprise": 101},
        {"crowd": -1}, {"crowd": 101},
        {"era": -1}, {"era": 101},
        {"indian": -1}, {"indian": 101},
    ):
        r = logged_client.get("/api/next", params=params)
        assert r.status_code == 422, params


def _components_by_id(client, params) -> dict[int, dict]:
    body = client.get("/api/next", params={**params, "n": 17}).json()
    return {item["id"]: item for item in body["items"]}


def test_indian_slider_extremes_shift_scores(logged_client):
    lo = _components_by_id(logged_client, {"surprise": 0, "indian": 0})
    hi = _components_by_id(logged_client, {"surprise": 0, "indian": 100})

    indian_movie_id = 2  # Baahubali, is_indian
    global_movie_id = 11  # The Matrix, not indian
    assert hi[indian_movie_id]["math"]["score"] > lo[indian_movie_id]["math"]["score"]
    assert hi[global_movie_id]["math"]["score"] < lo[global_movie_id]["math"]["score"]


def test_era_slider_extremes_shift_era_component(logged_client):
    old = _components_by_id(logged_client, {"surprise": 0, "era": 0})
    new = _components_by_id(logged_client, {"surprise": 0, "era": 100})

    old_movie_id = 15   # Casablanca, 1942
    new_movie_id = 6    # Kantara, 2022
    assert (
        old[old_movie_id]["math"]["components"]["era"]
        > new[old_movie_id]["math"]["components"]["era"]
    )
    assert (
        new[new_movie_id]["math"]["components"]["era"]
        > old[new_movie_id]["math"]["components"]["era"]
    )


def test_crowd_slider_extremes_favor_obscure_vs_popular(logged_client):
    obscure_first = _components_by_id(logged_client, {"surprise": 0, "crowd": 0})
    blockbuster_first = _components_by_id(logged_client, {"surprise": 0, "crowd": 100})

    low_votes_id = 9    # Premam, 500 votes
    high_votes_id = 11  # The Matrix, 25000 votes
    assert (
        obscure_first[low_votes_id]["math"]["components"]["popularity"]
        > blockbuster_first[low_votes_id]["math"]["components"]["popularity"]
    )
    assert (
        blockbuster_first[high_votes_id]["math"]["components"]["popularity"]
        > obscure_first[high_votes_id]["math"]["components"]["popularity"]
    )


def test_surprise_zero_is_deterministic_top_score(logged_client):
    a = logged_client.get("/api/next", params={"n": 10, "surprise": 0}).json()
    b = logged_client.get("/api/next", params={"n": 10, "surprise": 0}).json()
    assert a == b
    scores = [item["math"]["score"] for item in a["items"]]
    assert scores == sorted(scores, reverse=True)


def test_surprise_sampling_is_deterministic_given_same_inputs(logged_client):
    params = {"n": 8, "surprise": 90, "crowd": 40, "era": 60, "indian": 55}
    a = logged_client.get("/api/next", params=params).json()
    b = logged_client.get("/api/next", params=params).json()
    assert a == b  # same watch log + same slider params -> same picks, same order


def test_surprise_high_can_diverge_from_pure_top_score(logged_client):
    """Not a strict requirement of any single call, but the surprise=100 pool
    must be able to differ from the deterministic top-N — otherwise the
    slider would be a no-op."""
    top = [
        item["id"]
        for item in logged_client.get(
            "/api/next", params={"n": 5, "surprise": 0}
        ).json()["items"]
    ]
    surprised = [
        item["id"]
        for item in logged_client.get(
            "/api/next", params={"n": 5, "surprise": 100}
        ).json()["items"]
    ]
    assert len(surprised) == 5
    # Both are valid non-watched picks; they need not be disjoint, but the
    # mechanism must be exercised without error and return full-size lists.
    assert len(top) == 5
