"""Regression tests for the effort-4 /part3 review fixes (E4-1..E4-6)."""

from __future__ import annotations

from pathlib import Path

import recsys.recommenders.practical as prac

REPO_ROOT = Path(__file__).resolve().parents[1]


def _movie(mid: int, year: int = 1995) -> prac.CatalogMovie:
    return prac.CatalogMovie(
        id=mid,
        title=f"Twin {mid}",
        year=year,
        genres=frozenset({"Action"}),
        language="en",
        poster_path=None,
        is_indian=False,
        popularity=50.0,
        vote_average=7.0,
        vote_count=1000,
    )


def _picks(movies):
    watched = []
    profile = prac.derive_taste_profile(
        [prac.WatchedMovie(id=1, rating=5.0, movie=_movie(1))]
    )
    sliders = prac.SliderParams()
    picks = []
    for m in movies:
        _, comps = prac.score_movie(m, profile, 0.5, sliders, 1990.0)
        picks.append(type("P", (), {"movie": m, "components": comps})())
    return picks, profile, sliders, watched


def test_e4_1_attribute_coincident_twins_get_distinct_explanations():
    # Same genre/decade/language, ids congruent mod 3 — the old code collided.
    movies = [_movie(100), _movie(103), _movie(106), _movie(109)]
    picks, profile, sliders, watched = _picks(movies)
    texts = prac.build_unique_explanations(picks, profile, sliders, watched)
    assert len(texts) == len(set(texts)), texts
    # Deterministic: same inputs -> same outputs.
    assert texts == prac.build_unique_explanations(picks, profile, sliders, watched)


def test_e4_5_match10_rounds_half_up():
    assert prac.match10(0.85) == 9
    assert prac.match10(0.75) == 8
    assert prac.match10(0.0) == 1  # never 0/10
    assert prac.match10(1.0) == 10


def test_e4_2_omdb_url_is_https():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "enrich_ratings", REPO_ROOT / "scripts" / "enrich_ratings.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.OMDB_URL.startswith("https://")


def test_e4_3_corrupt_cache_entry_is_quarantined(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "enrich_ratings", REPO_ROOT / "scripts" / "enrich_ratings.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import sqlite3

    db = tmp_path / "catalog.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE movies (id INTEGER PRIMARY KEY, title TEXT, year INTEGER)")
    conn.execute("INSERT INTO movies VALUES (7, 'Broken', 1999)")
    conn.commit()
    conn.close()

    cache_dir = tmp_path / "omdb"
    cache_dir.mkdir()
    corrupt = cache_dir / "7.json"
    corrupt.write_text("{not json", encoding="utf-8")

    stats = mod.enrich(db_path=db, cache_dir=cache_dir, api_key=None, offline=True)
    assert stats["missing"] == 1
    # E4-3: corrupt entry removed so a later online run can re-fetch.
    assert not corrupt.exists()


def test_e4_6_ui_has_nan_guard():
    html = (REPO_ROOT / "recsys" / "web" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "Number.isFinite" in html
