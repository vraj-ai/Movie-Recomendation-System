"""Tests for W12 — external ratings enrichment (scripts/enrich_ratings.py).

Contracts under test (docs/plan/effort4-ratings-explanations.md):
- RAT-1: missing data stays NULL — never invented.
- RAT-FAIL-1: no key / no cache -> skip with NULLs, never a failed build.
- RAT-FAIL-2: malformed OMDb payloads ("N/A", missing fields) coerce to NULL.
- Migration: adds the three nullable columns idempotently to an existing DB.
- --offline: reads only the cache, never the network.
- Fixture: committed fixtures/catalog-small.sqlite carries pinned ratings for
  some rows and deliberate NULLs for others (both render paths testable).
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CATALOG = REPO_ROOT / "fixtures" / "catalog-small.sqlite"

spec = importlib.util.spec_from_file_location(
    "enrich_ratings", REPO_ROOT / "scripts" / "enrich_ratings.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

_LEGACY_DDL = """
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
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _make_legacy_db(path: Path, titles=(("Heat", 1995),)) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(_LEGACY_DDL)
    for i, (title, year) in enumerate(titles, start=1):
        conn.execute(
            "INSERT INTO movies (id, title, year, language, source) "
            "VALUES (?, ?, ?, 'en', 'ml')",
            (i, title, year),
        )
    conn.commit()
    conn.close()


def _cols(conn) -> set[str]:
    return {r[1] for r in conn.execute("PRAGMA table_info(movies)")}


# ---------------------------------------------------------------- parsing


class TestParseOmdb:
    def test_full_payload(self):
        raw = {
            "Response": "True",
            "imdbRating": "8.3",
            "Rated": "R",
            "Ratings": [
                {"Source": "Internet Movie Database", "Value": "8.3/10"},
                {"Source": "Rotten Tomatoes", "Value": "87%"},
            ],
        }
        assert mod.parse_omdb(raw) == (8.3, 87, "R")

    def test_na_fields_coerce_to_null(self):
        # RAT-FAIL-2
        raw = {"Response": "True", "imdbRating": "N/A", "Rated": "N/A", "Ratings": []}
        assert mod.parse_omdb(raw) == (None, None, None)

    def test_missing_fields_coerce_to_null(self):
        assert mod.parse_omdb({"Response": "True"}) == (None, None, None)

    def test_no_match_response(self):
        assert mod.parse_omdb({"Response": "False", "Error": "Movie not found!"}) == (
            None,
            None,
            None,
        )

    def test_malformed_rt_value(self):
        raw = {
            "Response": "True",
            "Ratings": [{"Source": "Rotten Tomatoes", "Value": "N/A"}],
        }
        assert mod.parse_omdb(raw) == (None, None, None)

    def test_unrated_normalized(self):
        raw = {"Response": "True", "Rated": "Not Rated"}
        assert mod.parse_omdb(raw) == (None, None, "Not Rated")


# ---------------------------------------------------------------- migration


class TestMigration:
    def test_adds_columns_to_legacy_db(self, tmp_path):
        db = tmp_path / "cat.sqlite"
        _make_legacy_db(db)
        conn = sqlite3.connect(db)
        added = mod.ensure_columns(conn)
        assert added is True
        assert {"imdb_rating", "rt_rating", "age_rating"} <= _cols(conn)
        # nullable: existing rows read back as NULL
        row = conn.execute(
            "SELECT imdb_rating, rt_rating, age_rating FROM movies WHERE id=1"
        ).fetchone()
        assert row == (None, None, None)

    def test_idempotent(self, tmp_path):
        db = tmp_path / "cat.sqlite"
        _make_legacy_db(db)
        conn = sqlite3.connect(db)
        assert mod.ensure_columns(conn) is True
        assert mod.ensure_columns(conn) is False  # second run: nothing to do


# ---------------------------------------------------------------- enrichment


class TestOfflineEnrichment:
    def test_cache_hit_writes_ratings(self, tmp_path):
        db = tmp_path / "cat.sqlite"
        _make_legacy_db(db, titles=(("Heat", 1995),))
        cache = tmp_path / "omdb"
        cache.mkdir()
        (cache / "1.json").write_text(
            json.dumps(
                {
                    "Response": "True",
                    "imdbRating": "8.3",
                    "Rated": "R",
                    "Ratings": [{"Source": "Rotten Tomatoes", "Value": "87%"}],
                }
            )
        )
        stats = mod.enrich(db, cache_dir=cache, offline=True, api_key=None)
        conn = sqlite3.connect(db)
        assert conn.execute(
            "SELECT imdb_rating, rt_rating, age_rating FROM movies WHERE id=1"
        ).fetchone() == (8.3, 87, "R")
        assert stats["enriched"] == 1

    def test_cache_miss_leaves_nulls(self, tmp_path):
        # RAT-1 / RAT-FAIL-1: offline + no cache entry -> NULLs, exit 0 path.
        db = tmp_path / "cat.sqlite"
        _make_legacy_db(db)
        cache = tmp_path / "omdb"
        cache.mkdir()
        stats = mod.enrich(db, cache_dir=cache, offline=True, api_key=None)
        conn = sqlite3.connect(db)
        assert conn.execute(
            "SELECT imdb_rating, rt_rating, age_rating FROM movies WHERE id=1"
        ).fetchone() == (None, None, None)
        assert stats["enriched"] == 0
        assert stats["missing"] == 1

    def test_offline_never_touches_network(self, tmp_path, monkeypatch):
        db = tmp_path / "cat.sqlite"
        _make_legacy_db(db)
        cache = tmp_path / "omdb"
        cache.mkdir()

        def boom(*a, **k):  # pragma: no cover - failure path
            raise AssertionError("network touched in --offline mode")

        monkeypatch.setattr(mod, "fetch_omdb", boom)
        mod.enrich(db, cache_dir=cache, offline=True, api_key="k")

    def test_corrupt_cache_entry_is_skipped(self, tmp_path):
        # RAT-FAIL-2: unparseable cache file -> NULLs, no crash.
        db = tmp_path / "cat.sqlite"
        _make_legacy_db(db)
        cache = tmp_path / "omdb"
        cache.mkdir()
        (cache / "1.json").write_text("{not json")
        stats = mod.enrich(db, cache_dir=cache, offline=True, api_key=None)
        assert stats["enriched"] == 0
        conn = sqlite3.connect(db)
        assert conn.execute(
            "SELECT imdb_rating FROM movies WHERE id=1"
        ).fetchone() == (None,)


# ---------------------------------------------------------------- fixture DB


class TestFixtureRatings:
    """The committed fixture must exercise both render paths (W12)."""

    @pytest.fixture()
    def conn(self):
        assert FIXTURE_CATALOG.exists(), "committed fixture missing"
        c = sqlite3.connect(f"file:{FIXTURE_CATALOG}?mode=ro", uri=True)
        yield c
        c.close()

    def test_fixture_has_rating_columns(self, conn):
        assert {"imdb_rating", "rt_rating", "age_rating"} <= _cols(conn)

    def test_fixture_has_rated_and_null_rows(self, conn):
        rated = conn.execute(
            "SELECT COUNT(*) FROM movies WHERE imdb_rating IS NOT NULL"
        ).fetchone()[0]
        nulls = conn.execute(
            "SELECT COUNT(*) FROM movies WHERE imdb_rating IS NULL"
        ).fetchone()[0]
        assert rated >= 3
        assert nulls >= 3

    def test_fixture_rating_ranges(self, conn):
        for imdb, rt in conn.execute(
            "SELECT imdb_rating, rt_rating FROM movies "
            "WHERE imdb_rating IS NOT NULL OR rt_rating IS NOT NULL"
        ):
            if imdb is not None:
                assert 0.0 <= imdb <= 10.0
            if rt is not None:
                assert 0 <= rt <= 100
