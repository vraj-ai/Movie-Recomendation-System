"""Tests for the combined catalog (W8): schema, dedupe, Indian-language
mapping, fixture integrity.

Always runs against fixtures/catalog-small.sqlite (committed, deterministic,
offline). Also runs the same checks against data/catalog.sqlite when present,
skipped otherwise since that file is gitignored and only produced by running
``scripts/build_catalog.py`` locally.

Contract (FROZEN): docs/plan/effort3-practical-watchlog.md, "Catalog contract".
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CATALOG = REPO_ROOT / "fixtures" / "catalog-small.sqlite"
FULL_CATALOG = REPO_ROOT / "data" / "catalog.sqlite"

INDIAN_LANGS = {"hi", "ta", "te", "ml", "kn", "bn", "mr", "pa", "gu", "or", "as"}

EXPECTED_MOVIE_COLUMNS = [
    ("id", "INTEGER"),
    ("tmdb_id", "INTEGER"),
    ("ml_id", "INTEGER"),
    ("title", "TEXT"),
    ("year", "INTEGER"),
    ("language", "TEXT"),
    ("genres", "TEXT"),
    ("overview", "TEXT"),
    ("poster_path", "TEXT"),
    ("popularity", "REAL"),
    ("vote_average", "REAL"),
    ("vote_count", "INTEGER"),
    ("is_indian", "INTEGER"),
    ("source", "TEXT"),
]


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture(params=["fixture", "full"])
def catalog_conn(request):
    """Parametrized connection: always runs against the fixture; also runs
    against the full data/catalog.sqlite when it exists locally."""
    if request.param == "fixture":
        path = FIXTURE_CATALOG
    else:
        if not FULL_CATALOG.exists():
            pytest.skip("data/catalog.sqlite not built locally (gitignored)")
        path = FULL_CATALOG
    conn = _connect(path)
    yield conn
    conn.close()


class TestSchema:
    def test_movies_table_columns_exact(self, catalog_conn):
        rows = catalog_conn.execute("PRAGMA table_info(movies)").fetchall()
        actual = [(r["name"], r["type"]) for r in rows]
        assert actual == EXPECTED_MOVIE_COLUMNS

    def test_movies_fts_exists_and_is_fts5_external_content(self, catalog_conn):
        row = catalog_conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'movies_fts'"
        ).fetchone()
        assert row is not None
        sql = row["sql"]
        assert "fts5" in sql.lower()
        assert "content='movies'" in sql.replace('"', "'")
        assert "content_rowid='id'" in sql.replace('"', "'")

    def test_meta_table_exists(self, catalog_conn):
        row = catalog_conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'meta'"
        ).fetchone()
        assert row is not None

    def test_source_check_values_only(self, catalog_conn):
        rows = catalog_conn.execute(
            "SELECT DISTINCT source FROM movies"
        ).fetchall()
        sources = {r["source"] for r in rows}
        assert sources <= {"ml", "indian", "both"}

    def test_inserting_invalid_source_is_rejected(self, catalog_conn):
        with pytest.raises(sqlite3.IntegrityError):
            catalog_conn.execute(
                "INSERT INTO movies (id, title, language, source) "
                "VALUES (999999, 'x', 'en', 'bogus')"
            )
            catalog_conn.commit()
        catalog_conn.rollback()


class TestCatalogContract:
    def test_every_row_has_nonempty_title(self, catalog_conn):
        row = catalog_conn.execute(
            "SELECT COUNT(*) AS n FROM movies WHERE title IS NULL OR title = ''"
        ).fetchone()
        assert row["n"] == 0

    def test_every_row_has_nonempty_language(self, catalog_conn):
        row = catalog_conn.execute(
            "SELECT COUNT(*) AS n FROM movies WHERE language IS NULL OR language = ''"
        ).fetchone()
        assert row["n"] == 0

    def test_is_indian_iff_language_in_indian_set(self, catalog_conn):
        """CAT-2: is_indian <=> language in the Indian-language set."""
        placeholders = ",".join("?" for _ in INDIAN_LANGS)

        mismatched_should_be_1 = catalog_conn.execute(
            f"SELECT COUNT(*) AS n FROM movies "
            f"WHERE language IN ({placeholders}) AND is_indian != 1",
            tuple(INDIAN_LANGS),
        ).fetchone()["n"]
        assert mismatched_should_be_1 == 0

        mismatched_should_be_0 = catalog_conn.execute(
            f"SELECT COUNT(*) AS n FROM movies "
            f"WHERE language NOT IN ({placeholders}) AND is_indian != 0",
            tuple(INDIAN_LANGS),
        ).fetchone()["n"]
        assert mismatched_should_be_0 == 0

    def test_tmdb_id_unique_among_non_null(self, catalog_conn):
        row = catalog_conn.execute(
            """
            SELECT COUNT(*) AS n FROM (
                SELECT tmdb_id FROM movies
                WHERE tmdb_id IS NOT NULL
                GROUP BY tmdb_id HAVING COUNT(*) > 1
            )
            """
        ).fetchone()
        assert row["n"] == 0

    def test_ml_id_unique_among_non_null(self, catalog_conn):
        row = catalog_conn.execute(
            """
            SELECT COUNT(*) AS n FROM (
                SELECT ml_id FROM movies
                WHERE ml_id IS NOT NULL
                GROUP BY ml_id HAVING COUNT(*) > 1
            )
            """
        ).fetchone()
        assert row["n"] == 0

    def test_both_source_rows_have_both_ids(self, catalog_conn):
        row = catalog_conn.execute(
            "SELECT COUNT(*) AS n FROM movies WHERE source = 'both' "
            "AND (tmdb_id IS NULL OR ml_id IS NULL)"
        ).fetchone()
        assert row["n"] == 0

    def test_ml_source_rows_have_ml_id(self, catalog_conn):
        row = catalog_conn.execute(
            "SELECT COUNT(*) AS n FROM movies WHERE source = 'ml' AND ml_id IS NULL"
        ).fetchone()
        assert row["n"] == 0

    def test_indian_source_rows_have_no_ml_id(self, catalog_conn):
        row = catalog_conn.execute(
            "SELECT COUNT(*) AS n FROM movies WHERE source = 'indian' AND ml_id IS NOT NULL"
        ).fetchone()
        assert row["n"] == 0

    def test_indian_source_rows_have_tmdb_id(self, catalog_conn):
        row = catalog_conn.execute(
            "SELECT COUNT(*) AS n FROM movies WHERE source = 'indian' AND tmdb_id IS NULL"
        ).fetchone()
        assert row["n"] == 0

    def test_numeric_defaults_not_null(self, catalog_conn):
        """popularity/vote_average/vote_count/genres/overview default to 0/''
        rather than NULL, per the NOT NULL DEFAULT contract."""
        row = catalog_conn.execute(
            "SELECT COUNT(*) AS n FROM movies WHERE popularity IS NULL "
            "OR vote_average IS NULL OR vote_count IS NULL "
            "OR genres IS NULL OR overview IS NULL"
        ).fetchone()
        assert row["n"] == 0


class TestFTSSearch:
    def test_fts_finds_known_fixture_title(self):
        conn = _connect(FIXTURE_CATALOG)
        try:
            rows = conn.execute(
                "SELECT m.title FROM movies_fts f "
                "JOIN movies m ON m.id = f.rowid "
                "WHERE movies_fts MATCH ?",
                ("Idiot*",),
            ).fetchall()
        finally:
            conn.close()
        titles = {r["title"] for r in rows}
        assert "3 Idiots" in titles

    def test_fts_prefix_search_toy_story(self):
        conn = _connect(FIXTURE_CATALOG)
        try:
            rows = conn.execute(
                "SELECT m.title FROM movies_fts f "
                "JOIN movies m ON m.id = f.rowid "
                "WHERE movies_fts MATCH ?",
                ("Toy*",),
            ).fetchall()
        finally:
            conn.close()
        titles = {r["title"] for r in rows}
        assert "Toy Story" in titles

    def test_fts_no_match_returns_empty(self):
        conn = _connect(FIXTURE_CATALOG)
        try:
            rows = conn.execute(
                "SELECT m.title FROM movies_fts f "
                "JOIN movies m ON m.id = f.rowid "
                "WHERE movies_fts MATCH ?",
                ("Zzzznonexistentmovie*",),
            ).fetchall()
        finally:
            conn.close()
        assert rows == []


class TestMetaCounts:
    def test_meta_counts_match_select_counts(self, catalog_conn):
        meta = {
            r["key"]: r["value"]
            for r in catalog_conn.execute("SELECT key, value FROM meta")
        }
        for key in ("total_rows", "ml_rows", "indian_rows", "both_rows"):
            assert key in meta

        actual_total = catalog_conn.execute(
            "SELECT COUNT(*) AS n FROM movies"
        ).fetchone()["n"]
        assert int(meta["total_rows"]) == actual_total

        for source in ("ml", "indian", "both"):
            actual = catalog_conn.execute(
                "SELECT COUNT(*) AS n FROM movies WHERE source = ?", (source,)
            ).fetchone()["n"]
            assert int(meta[f"{source}_rows"]) == actual

    def test_meta_has_build_timestamp(self, catalog_conn):
        row = catalog_conn.execute(
            "SELECT value FROM meta WHERE key = 'build_timestamp'"
        ).fetchone()
        assert row is not None
        assert row["value"]

    def test_meta_has_source_urls(self, catalog_conn):
        for key in ("ml_movies_url", "ml_links_url", "tmdb_dump_url"):
            row = catalog_conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
            assert row is not None
            assert row["value"].startswith("https://")
            # CAT-1: never api.themoviedb.org
            assert "api.themoviedb.org" not in row["value"]


class TestFixtureIntegrity:
    """fixtures/catalog-small.sqlite: 25 rows, 15 with ml_id (14 'ml' + 1
    'both'), 10 pure 'indian', Indian rows span >=3 languages, some ml rows
    have a NULL poster_path."""

    @pytest.fixture
    def conn(self):
        conn = _connect(FIXTURE_CATALOG)
        yield conn
        conn.close()

    def test_row_count(self, conn):
        n = conn.execute("SELECT COUNT(*) AS n FROM movies").fetchone()["n"]
        assert n == 25

    def test_source_breakdown(self, conn):
        counts = {
            r["source"]: r["n"]
            for r in conn.execute(
                "SELECT source, COUNT(*) AS n FROM movies GROUP BY source"
            )
        }
        assert counts.get("ml", 0) == 14
        assert counts.get("both", 0) == 1
        assert counts.get("indian", 0) == 10
        assert counts.get("ml", 0) + counts.get("both", 0) == 15

    def test_indian_rows_span_at_least_three_languages(self, conn):
        langs = {
            r["language"]
            for r in conn.execute(
                "SELECT DISTINCT language FROM movies WHERE is_indian = 1"
            )
        }
        assert len(langs) >= 3
        assert langs <= INDIAN_LANGS

    def test_some_ml_rows_have_null_poster(self, conn):
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM movies WHERE source = 'ml' AND poster_path IS NULL"
        ).fetchone()["n"]
        assert n >= 1

    def test_at_least_one_both_row(self, conn):
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM movies WHERE source = 'both'"
        ).fetchone()["n"]
        assert n >= 1

    def test_deterministic_rebuild_matches_committed_fixture(self, tmp_path):
        """Re-running the build script offline (cached downloads only)
        reproduces byte-identical row data — CAT-1 offline reproducibility."""
        import subprocess
        import sys

        # Skip if the source caches aren't present (fixture is committed and
        # this test is a bonus determinism check, not a hard requirement for
        # CI environments without the multi-hundred-MB TMDB dump cached).
        downloads = REPO_ROOT / "cache" / "downloads"
        required = ["movies.csv", "links.csv", "movies_dataset.csv"]
        if not all((downloads / f).exists() for f in required):
            pytest.skip("cache/downloads/ not populated locally")

        committed_bytes = FIXTURE_CATALOG.read_bytes()

        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "build_catalog.py"), "--fixture"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr

        # E3-6: byte-identical, not merely row-identical — the committed
        # fixture must be exactly what ``--fixture`` regenerates, or every
        # contributor rebuild produces binary churn in git.
        assert FIXTURE_CATALOG.read_bytes() == committed_bytes

        rebuilt = _connect(FIXTURE_CATALOG)
        try:
            rebuilt_rows = rebuilt.execute(
                "SELECT id, tmdb_id, ml_id, title, year, language, genres, "
                "overview, poster_path, is_indian, source FROM movies ORDER BY id"
            ).fetchall()
        finally:
            rebuilt.close()

        assert len(rebuilt_rows) == 25
        titles = [r["title"] for r in rebuilt_rows]
        assert "Toy Story" in titles
        assert "3 Idiots" in titles


class TestPosterCoverage:
    def test_full_catalog_poster_coverage_reported(self):
        report_path = REPO_ROOT / "data" / "catalog-report.json"
        if not report_path.exists():
            pytest.skip("data/catalog-report.json not built locally (gitignored)")
        import json

        report = json.loads(report_path.read_text())
        assert 0.0 <= report["rows_with_poster_pct"] <= 100.0
        assert report["ml_sourced_total"] == 9742
        assert report["indian_rows"] + report["both_rows"] >= 8000


# =====================================================================
# /part3 review (W11) — finding E3-6: byte-level reproducibility
# =====================================================================


def _load_build_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_catalog", REPO_ROOT / "scripts" / "build_catalog.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_df(mod):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "id": 1, "tmdb_id": 11, "ml_id": 101, "title": "Alpha",
                "year": 1999, "language": "en", "genres": "Drama",
                "overview": "a", "poster_path": None, "popularity": 1.0,
                "vote_average": 7.0, "vote_count": 10, "is_indian": 0,
                "source": "ml",
            },
            {
                "id": 2, "tmdb_id": 22, "ml_id": None, "title": "Beta",
                "year": 2001, "language": "hi", "genres": "Action",
                "overview": "b", "poster_path": "/b.jpg", "popularity": 2.0,
                "vote_average": 8.0, "vote_count": 20, "is_indian": 1,
                "source": "indian",
            },
        ]
    )


class TestByteReproducibility:
    def test_write_sqlite_honors_source_date_epoch(self, tmp_path, monkeypatch):
        """E3-6: with SOURCE_DATE_EPOCH pinned, two builds of the same frame
        must be byte-identical — no wall-clock leaks into the artifact."""
        mod = _load_build_module()
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
        df = _tiny_df(mod)
        a, b = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
        mod.write_sqlite(df, a, {"schema_version": "1"})
        mod.write_sqlite(df, b, {"schema_version": "1"})
        assert a.read_bytes() == b.read_bytes()

    def test_fixture_timestamp_is_pinned_constant(self, tmp_path):
        """E3-6: --fixture must not embed wall clock even without
        SOURCE_DATE_EPOCH; it pins FIXTURE_BUILD_TIMESTAMP so the committed
        fixture never churns."""
        mod = _load_build_module()
        df = _tiny_df(mod)
        out = tmp_path / "f.sqlite"
        mod.write_sqlite(
            df, out, {"fixture": "true"}, timestamp=mod.FIXTURE_BUILD_TIMESTAMP
        )
        conn = sqlite3.connect(out)
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'build_timestamp'"
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == mod.FIXTURE_BUILD_TIMESTAMP
