"""Enrich a catalog DB with external ratings from OMDb (W12, effort 4).

Adds three nullable display-metadata columns to ``movies``:

    imdb_rating REAL      -- e.g. 8.3
    rt_rating   INTEGER   -- Rotten Tomatoes %, e.g. 87
    age_rating  TEXT      -- OMDb "Rated", e.g. 'R', 'PG-13', 'Not Rated'

Contracts (docs/plan/effort4-ratings-explanations.md):

- RAT-1: missing data stays NULL — never invented.
- RAT-FAIL-1: no API key / unreachable OMDb -> skip with NULLs, log, exit 0.
  A build never fails because of ratings.
- RAT-FAIL-2: malformed payloads ("N/A", missing fields) coerce to NULL.
- Reproducibility (E3-6): every raw response is cached to
  ``data/cache/omdb/<movie_id>.json``; with ``--offline`` only the cache is
  read, so rebuilds are byte-stable and test runs never touch the network.

Usage:
    .venv/bin/python scripts/enrich_ratings.py                    # data/catalog.sqlite
    .venv/bin/python scripts/enrich_ratings.py --offline          # cache-only replay
    .venv/bin/python scripts/enrich_ratings.py --db path.sqlite
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger("enrich_ratings")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "catalog.sqlite"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache" / "omdb"

OMDB_URL = "https://www.omdbapi.com/"  # E4-2: never leak the API key over plain HTTP
REQUEST_DELAY = 0.2  # be polite on the free tier
TIMEOUT = 10

_COLUMNS = (
    ("imdb_rating", "REAL"),
    ("rt_rating", "INTEGER"),
    ("age_rating", "TEXT"),
)


# ---------------------------------------------------------------- parsing


def parse_omdb(raw: dict) -> tuple[float | None, int | None, str | None]:
    """Extract (imdb_rating, rt_rating, age_rating) from a raw OMDb payload.

    RAT-FAIL-2: anything missing, "N/A", or unparseable coerces to None.
    """
    if not isinstance(raw, dict) or raw.get("Response") != "True":
        return (None, None, None)

    imdb: float | None = None
    v = raw.get("imdbRating")
    if isinstance(v, str) and v not in ("", "N/A"):
        try:
            imdb = float(v)
        except ValueError:
            imdb = None

    rt: int | None = None
    for entry in raw.get("Ratings") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("Source") == "Rotten Tomatoes":
            val = entry.get("Value")
            if isinstance(val, str) and val.endswith("%"):
                try:
                    rt = int(val.rstrip("%"))
                except ValueError:
                    rt = None
            break

    rated: str | None = None
    r = raw.get("Rated")
    if isinstance(r, str) and r not in ("", "N/A"):
        rated = r

    return (imdb, rt, rated)


# ---------------------------------------------------------------- migration


def ensure_columns(conn: sqlite3.Connection) -> bool:
    """Add the three nullable rating columns if absent. Returns True if any
    column was added (idempotent — second call returns False)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(movies)")}
    added = False
    for name, sql_type in _COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE movies ADD COLUMN {name} {sql_type}")
            added = True
    conn.commit()
    return added


# ---------------------------------------------------------------- fetching


def fetch_omdb(title: str, year: int | None, api_key: str) -> dict:
    """One live OMDb lookup by title (+year). Raises on transport errors;
    callers treat any exception as RAT-FAIL-1 (skip, leave NULL)."""
    params = {"apikey": api_key, "t": title, "type": "movie"}
    if year:
        params["y"] = str(year)
    url = f"{OMDB_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------- enrichment


def enrich(
    db_path: Path,
    cache_dir: Path,
    offline: bool,
    api_key: str | None,
) -> dict[str, int]:
    """Migrate + enrich every movie row. Never raises for per-movie failures."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        ensure_columns(conn)
        rows = conn.execute("SELECT id, title, year FROM movies").fetchall()

        stats = {"total": len(rows), "enriched": 0, "missing": 0, "fetched": 0}
        for movie_id, title, year in rows:
            cache_path = cache_dir / f"{movie_id}.json"
            raw: dict | None = None

            if cache_path.exists():
                try:
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    # RAT-FAIL-2 / E4-3: quarantine the corrupt entry so it
                    # no longer permanently blocks a re-fetch, then fall
                    # through to the network path below (when available).
                    logger.warning(
                        "corrupt cache entry for %s — removing so it can be re-fetched",
                        movie_id,
                    )
                    raw = None
                    try:
                        cache_path.unlink()
                    except OSError:
                        pass
            if raw is None and not cache_path.exists() and not offline and api_key:
                try:
                    raw = fetch_omdb(title, year, api_key)
                    cache_path.write_text(
                        json.dumps(raw, sort_keys=True), encoding="utf-8"
                    )
                    stats["fetched"] += 1
                    time.sleep(REQUEST_DELAY)
                except Exception as e:  # RAT-FAIL-1: never fail the build
                    logger.warning("OMDb lookup failed for %r (%s)", title, e)
                    raw = None

            imdb, rt, rated = parse_omdb(raw) if raw else (None, None, None)
            if (imdb, rt, rated) != (None, None, None):
                conn.execute(
                    "UPDATE movies SET imdb_rating=?, rt_rating=?, age_rating=? "
                    "WHERE id=?",
                    (imdb, rt, rated, movie_id),
                )
                stats["enriched"] += 1
            else:
                stats["missing"] += 1

        conn.commit()
        return stats
    finally:
        conn.close()


# ---------------------------------------------------------------- CLI


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Read only the cache; never touch the network (reproducible).",
    )
    args = parser.parse_args()

    if not args.db.exists():
        logger.error("catalog not found: %s", args.db)
        return 1

    api_key = os.environ.get("OMDB_API_KEY")
    if not api_key and not args.offline:
        # RAT-FAIL-1: keyless run degrades to offline, never fails.
        logger.warning("OMDB_API_KEY not set — running cache-only (--offline).")
        args.offline = True

    stats = enrich(args.db, cache_dir=args.cache_dir, offline=args.offline, api_key=api_key)
    logger.info(
        "%(total)d movies: %(enriched)d enriched, %(missing)d left NULL, "
        "%(fetched)d fetched live",
        stats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
