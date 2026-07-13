"""W13 — per-card explanations + match10 + external ratings in API payloads
(effort 4). Determinism: same inputs -> identical explanation. Variety:
distinct top picks -> no duplicate sentences. match10 is monotonic in score
and clamped to 1..10. /api/next payloads carry imdb_rating / rt_rating /
age_rating (None when the catalog lacks the columns — frozen effort-3 DDL).

Verification-command: .venv/bin/pytest tests/test_explanations.py -q
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recsys.recommenders import practical as prac
from recsys.web.app import create_app
from recsys.web.profile_store import ProfileStore


def _mk_movie(mid, title, genres, lang="en", year=2000, **kw):
    return prac.CatalogMovie(
        id=mid, title=title, year=year, language=lang,
        genres=frozenset(genres), poster_path=None, popularity=10.0,
        vote_average=7.0, vote_count=100, is_indian=False, **kw,
    )


def _profile_and_sliders():
    watched = [
        prac.WatchedMovie(id=1, rating=5.0, movie=_mk_movie(1, "Loved Film", {"Action"})),
        prac.WatchedMovie(id=2, rating=4.5, movie=_mk_movie(2, "Also Great", {"Drama"}, lang="hi", year=2010)),
        prac.WatchedMovie(id=3, rating=1.0, movie=_mk_movie(3, "Hated Film", {"Horror"})),
    ]
    profile = prac.derive_taste_profile(watched)
    sliders = prac.SliderParams(surprise=50.0, crowd=50.0, era=50.0, indian=50.0)
    return watched, profile, sliders


# ------------------------------------------------------------- match10


def test_match10_monotonic_and_clamped():
    scores = [-0.5, 0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0, 1.5]
    out = [prac.match10(s) for s in scores]
    assert out == sorted(out)  # monotonic non-decreasing
    assert all(1 <= v <= 10 for v in out)
    assert prac.match10(0.0) == 1  # floor: never a 0/10
    assert prac.match10(1.0) == 10


def test_match10_is_int():
    assert isinstance(prac.match10(0.7), int)


# ------------------------------------------------------ build_explanation


def _explain(m):
    watched, profile, sliders = _profile_and_sliders()
    _score, components = prac.score_movie(m, profile, 0.5, sliders, 2000.0)
    return prac.build_explanation(m, profile, components, sliders, watched)


def test_explanation_is_deterministic():
    m = _mk_movie(10, "Candidate", {"Action"})
    assert _explain(m) == _explain(m)


def test_explanation_is_one_nonempty_sentence():
    text = _explain(_mk_movie(10, "Candidate", {"Action"}))
    assert isinstance(text, str) and text.strip()
    assert text.endswith(".")
    assert "\n" not in text


def test_explanation_variety_across_top_picks():
    """No two of 10 distinct candidates read identically (plan: variety)."""
    movies = [
        _mk_movie(100 + i, f"Pick {i}", {g}, lang=lang, year=1970 + 5 * i)
        for i, (g, lang) in enumerate([
            ("Action", "en"), ("Drama", "hi"), ("Comedy", "en"),
            ("Action", "hi"), ("Drama", "en"), ("Thriller", "en"),
            ("Comedy", "hi"), ("Romance", "en"), ("Action", "ta"),
            ("Horror", "en"),
        ])
    ]
    texts = [_explain(m) for m in movies]
    assert len(set(texts)) == len(texts)


# ------------------------------------------------------------- API payload
# Synthetic catalog WITH the W12 ratings columns.

_DDL = """
CREATE TABLE movies (
  id INTEGER PRIMARY KEY, tmdb_id INTEGER UNIQUE, ml_id INTEGER UNIQUE,
  title TEXT NOT NULL, year INTEGER, language TEXT NOT NULL,
  genres TEXT NOT NULL DEFAULT '', overview TEXT NOT NULL DEFAULT '',
  poster_path TEXT, popularity REAL NOT NULL DEFAULT 0,
  vote_average REAL NOT NULL DEFAULT 0, vote_count INTEGER NOT NULL DEFAULT 0,
  is_indian INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL CHECK (source IN ('ml','indian','both')),
  imdb_rating REAL, rt_rating INTEGER, age_rating TEXT
);
CREATE VIRTUAL TABLE movies_fts USING fts5(title, content='movies', content_rowid='id');
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _build_catalog(path: Path, with_ratings_cols: bool = True) -> None:
    conn = sqlite3.connect(path)
    ddl = _DDL if with_ratings_cols else _DDL.replace(
        ",\n  imdb_rating REAL, rt_rating INTEGER, age_rating TEXT", ""
    )
    conn.executescript(ddl)
    rows = [
        (i, 1000 + i, None, f"Movie {i}", 1990 + i, "en" if i % 2 else "hi",
         "Action|Drama" if i % 3 else "Comedy", "", None, 50 + i, 6.5 + (i % 3), 500 + i,
         i % 2 == 0, "ml")
        for i in range(1, 13)
    ]
    if with_ratings_cols:
        conn.executemany(
            "INSERT INTO movies (id, tmdb_id, ml_id, title, year, language, genres,"
            " overview, poster_path, popularity, vote_average, vote_count, is_indian,"
            " source, imdb_rating, rt_rating, age_rating)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [r + ((7.5, 88, "PG-13") if r[0] == 1 else (None, None, None)) for r in rows],
        )
    else:
        conn.executemany(
            "INSERT INTO movies (id, tmdb_id, ml_id, title, year, language, genres,"
            " overview, poster_path, popularity, vote_average, vote_count, is_indian,"
            " source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    conn.execute("INSERT INTO movies_fts (rowid, title) SELECT id, title FROM movies")
    conn.commit()
    conn.close()


def _client(tmp_path: Path, with_ratings_cols: bool = True) -> TestClient:
    catalog = tmp_path / "catalog.sqlite"
    _build_catalog(catalog, with_ratings_cols)
    profile = tmp_path / "profile.json"
    ProfileStore(profile).save(ratings={}, pins={})
    app = create_app(mode="full", profile_path=profile, catalog_path=catalog)
    client = TestClient(app)
    for mid, rating in [(1, 5.0), (2, 4.5), (3, 4.0), (4, 2.0), (5, 1.0)]:
        assert client.post("/api/watched", json={"id": mid, "rating": rating}).status_code == 200
    return client


def test_next_payload_has_explanation_match10_and_ratings(tmp_path):
    client = _client(tmp_path)
    items = client.get("/api/next", params={"n": 5}).json()["items"]
    assert items
    for it in items:
        assert isinstance(it["explanation"], str) and it["explanation"].strip()
        assert isinstance(it["match10"], int) and 1 <= it["match10"] <= 10
        assert "imdb_rating" in it and "rt_rating" in it and "age_rating" in it
    texts = [it["explanation"] for it in items]
    assert len(set(texts)) == len(texts)  # no two cards read the same


def test_next_tolerates_catalog_without_ratings_columns(tmp_path):
    """Frozen effort-3 catalogs (no W12 columns) must still serve — ratings
    fields come back as None, never a 500 (WEB-FAIL-2)."""
    client = _client(tmp_path, with_ratings_cols=False)
    r = client.get("/api/next", params={"n": 3})
    assert r.status_code == 200
    for it in r.json()["items"]:
        assert it["imdb_rating"] is None
        assert it["rt_rating"] is None
        assert it["age_rating"] is None
