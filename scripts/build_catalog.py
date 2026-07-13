#!/usr/bin/env python3
"""Build the combined catalog (data/catalog.sqlite): full MovieLens
ml-latest-small (9,742 movies) + ALL Indian movies from a TMDB metadata dump.

Contract (FROZEN): docs/plan/effort3-practical-watchlog.md, "Catalog contract".

    CREATE TABLE movies (id, tmdb_id, ml_id, title, year, language, genres,
        overview, poster_path, popularity, vote_average, vote_count,
        is_indian, source CHECK IN ('ml','indian','both'))
    CREATE VIRTUAL TABLE movies_fts USING fts5(title, content='movies',
        content_rowid='id')
    CREATE TABLE meta (key, value)

CAT-1: never calls api.themoviedb.org; reproducible offline from
cache/downloads/ once the two source files are present. Network is only
touched by ``ensure_downloads()`` and only for files that are missing.

CAT-2: every row has non-empty title + language; is_indian <=> language is
in the Indian-language set (independent of dedupe/source bookkeeping).

Usage:
    .venv/bin/python scripts/build_catalog.py            # builds data/catalog.sqlite
    .venv/bin/python scripts/build_catalog.py --fixture   # builds fixtures/catalog-small.sqlite
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger("build_catalog")

REPO_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = REPO_ROOT / "cache" / "downloads"
DATA_DIR = REPO_ROOT / "data"
FIXTURES_DIR = REPO_ROOT / "fixtures"

ML_BASE_URL = "https://raw.githubusercontent.com/smanihwr/ml-latest-small/master"
TMDB_DUMP_URL = (
    "https://huggingface.co/datasets/wykonos/movies/resolve/main/movies_dataset.csv"
)

# filename -> source URL. movies.csv/links.csv are required to build; ratings/
# tags are downloaded for parity with the mirror but unused by this script.
ML_FILES = {
    "movies.csv": f"{ML_BASE_URL}/movies.csv",
    "links.csv": f"{ML_BASE_URL}/links.csv",
    "ratings.csv": f"{ML_BASE_URL}/ratings.csv",
    "tags.csv": f"{ML_BASE_URL}/tags.csv",
}
TMDB_DUMP_FILE = "movies_dataset.csv"

# Indian languages (ISO-639-1) per the frozen contract.
INDIAN_LANGS = {"hi", "ta", "te", "ml", "kn", "bn", "mr", "pa", "gu", "or", "as"}

SCHEMA_SQL = """
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
  source TEXT NOT NULL CHECK (source IN ('ml','indian','both')),
  -- W12 (effort 4): external display-metadata ratings, nullable by design
  -- (RAT-1: missing data stays NULL, never invented).
  imdb_rating REAL,
  rt_rating INTEGER,
  age_rating TEXT
);
CREATE VIRTUAL TABLE movies_fts USING fts5(title, content='movies', content_rowid='id');
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_YEAR_SUFFIX_RE = re.compile(r"\s*\((\d{4})\)\s*$")

# E3-6: the committed fixture must be byte-stable across reruns, so its
# build_timestamp is a pinned constant, never wall clock.
FIXTURE_BUILD_TIMESTAMP = "2026-07-12T00:00:00+00:00"

# W12: hand-pinned external ratings for the fixture so both render paths
# (real values and NULL -> "—") are testable offline. Movies absent from this
# map deliberately keep NULLs (RAT-1). Values as of 2026-07 OMDb, pinned.
# fixture id -> (imdb_rating, rt_rating, age_rating)
FIXTURE_RATINGS: dict[int, tuple[float | None, int | None, str | None]] = {
    1: (8.3, 100, "G"),        # Toy Story
    3: (8.3, 88, "R"),         # Heat
    6: (8.6, 82, "R"),         # Se7en
    10: (8.6, 93, "PG"),       # Star Wars: A New Hope
    11: (8.9, 92, "R"),        # Pulp Fiction
    12: (9.3, 89, "R"),        # The Shawshank Redemption
    15: (8.4, 100, "PG-13"),   # 3 Idiots
    17: (8.0, None, "Not Rated"),  # DDLJ — deliberate partial row
    21: (8.0, 90, "Not Rated"),    # Bāhubali: The Beginning
    24: (7.8, 95, "PG-13"),    # RRR
}


def apply_fixture_ratings(conn: sqlite3.Connection) -> None:
    """Pin the hand-written fixture ratings (W12). Idempotent."""
    for movie_id, (imdb, rt, rated) in FIXTURE_RATINGS.items():
        conn.execute(
            "UPDATE movies SET imdb_rating=?, rt_rating=?, age_rating=? WHERE id=?",
            (imdb, rt, rated, movie_id),
        )


def _build_timestamp() -> str:
    """Wall clock, unless SOURCE_DATE_EPOCH is set (reproducible-builds.org
    convention) — E3-6: lets any build be made byte-reproducible."""
    sde = os.environ.get("SOURCE_DATE_EPOCH")
    if sde is not None:
        return datetime.fromtimestamp(int(sde), timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()

# Deterministic fixture selection (business keys, not timestamps/order-of-
# insertion) so ``--fixture`` produces byte-identical output on every rerun
# against the same cached downloads.
#
# 15 rows carry ml_id: 12 plain 'ml' rows with a TMDB match, 2 'ml' rows with
# NO TMDB match at all (guaranteed poster_path IS NULL: movieId 1107 "Loser",
# movieId 2851 "Saturn 3" have no tmdbId in links.csv), and 1 row
# (movieId 73881 "3 Idiots") whose tmdbId also qualifies as an Indian dump
# row => source='both'. Plus 10 pure Indian-only rows across >=3 languages,
# selected deterministically below.
FIXTURE_ML_MOVIE_IDS = [
    1, 2, 6, 10, 16, 47, 50, 110, 150, 260, 296, 318,  # 12 plain ml, tmdb-matched
    73881,  # 3 Idiots -> source='both' (hi)
    1107, 2851,  # no tmdbId -> poster_path NULL
]
FIXTURE_INDIAN_LANG_QUOTAS = {"hi": 3, "ta": 3, "te": 2, "ml": 2}  # sums to 10


# --------------------------------------------------------------------------- #
# Download (network boundary; CAT-1 — never api.themoviedb.org)
# --------------------------------------------------------------------------- #


def _download(url: str, dest: Path) -> None:
    logger.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    tmp.replace(dest)


def ensure_downloads() -> None:
    """Download source files into cache/downloads/ if not already present.

    Re-runnable offline: once files exist, no network call is made (CAT-1).
    """
    for filename, url in ML_FILES.items():
        dest = DOWNLOADS_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            continue
        _download(url, dest)

    dump_dest = DOWNLOADS_DIR / TMDB_DUMP_FILE
    if not (dump_dest.exists() and dump_dest.stat().st_size > 0):
        _download(TMDB_DUMP_URL, dump_dest)


# --------------------------------------------------------------------------- #
# Load + transform
# --------------------------------------------------------------------------- #


def _parse_title_year(raw_title: str) -> tuple[str, int | None]:
    """Split MovieLens ``"Toy Story (1995)"`` -> ``("Toy Story", 1995)``."""
    m = _YEAR_SUFFIX_RE.search(raw_title)
    if m:
        return _YEAR_SUFFIX_RE.sub("", raw_title).strip(), int(m.group(1))
    return raw_title.strip(), None


def load_ml_movies() -> pd.DataFrame:
    """MovieLens movies.csv + links.csv joined on movieId.

    Returns columns: ml_id, title, year, ml_genres, tmdb_id (nullable Int64).
    """
    movies = pd.read_csv(DOWNLOADS_DIR / "movies.csv")
    links = pd.read_csv(DOWNLOADS_DIR / "links.csv")

    parsed = movies["title"].apply(_parse_title_year)
    movies["title_clean"] = [p[0] for p in parsed]
    movies["year"] = [p[1] for p in parsed]
    movies["ml_genres"] = movies["genres"].apply(
        lambda g: "" if g == "(no genres listed)" or pd.isna(g) else g
    )

    merged = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    merged["tmdbId"] = merged["tmdbId"].astype("Int64")

    # ml-latest-small has a known duplicate: movieId 6003 and 144606 are both
    # "Confessions of a Dangerous Mind (2002)" mapped to the same tmdbId
    # (4912). tmdb_id is UNIQUE in the catalog schema, so keep the mapping on
    # the lower movieId (deterministic — movies.csv is sorted by movieId) and
    # treat the later duplicate as unmatched (tmdb_id NULL, like any other ml
    # row with no TMDB link) rather than dropping a row and breaking the
    # ml rows == 9,742 invariant.
    dup_tmdb = merged["tmdbId"].notna() & merged.duplicated(
        subset=["tmdbId"], keep="first"
    )
    if dup_tmdb.any():
        logger.warning(
            "Nulling duplicate tmdbId mapping for %d ml row(s) (ml-latest-small "
            "quirk, e.g. movieId 6003/144606 both -> tmdbId 4912): %s",
            int(dup_tmdb.sum()),
            merged.loc[dup_tmdb, "movieId"].tolist(),
        )
        merged.loc[dup_tmdb, "tmdbId"] = pd.NA

    out = merged.rename(columns={"movieId": "ml_id", "tmdbId": "tmdb_id"})
    return out[["ml_id", "title_clean", "year", "ml_genres", "tmdb_id"]].rename(
        columns={"title_clean": "title"}
    )


def load_tmdb_dump() -> pd.DataFrame:
    """TMDB metadata dump (HF ``wykonos/movies``), deduped by id.

    Duplicate ids in the raw dump are re-scrapes of the same movie with
    differing popularity/vote fields; keep the richest (highest vote_count)
    row per id, tie-broken by id ascending — deterministic, no wall-clock or
    file-order dependence beyond the (fixed, cached) input file itself.
    """
    cols = [
        "id",
        "title",
        "genres",
        "original_language",
        "overview",
        "popularity",
        "release_date",
        "vote_average",
        "vote_count",
        "poster_path",
    ]
    dtypes = {
        "id": "Int64",
        "title": "string",
        "genres": "string",
        "original_language": "string",
        "overview": "string",
        "popularity": "float64",
        "release_date": "string",
        "vote_average": "float64",
        "vote_count": "Int64",
        "poster_path": "string",
    }
    df = pd.read_csv(
        DOWNLOADS_DIR / TMDB_DUMP_FILE,
        usecols=cols,
        dtype=dtypes,
    )
    df = df.dropna(subset=["id"])
    df["vote_count"] = df["vote_count"].fillna(0)
    df = df.sort_values(["id", "vote_count"], ascending=[True, False])
    df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    # TMDB genre lists in this dump are hyphen-joined ("Action-Crime"); the
    # contract wants pipe-separated to match MovieLens's convention.
    df["genres_pipe"] = df["genres"].apply(
        lambda g: "" if pd.isna(g) else "|".join(x for x in str(g).split("-") if x)
    )
    return df


def build_catalog_df(ml: pd.DataFrame, dump: pd.DataFrame) -> pd.DataFrame:
    """Join, dedupe, and shape the combined catalog per the frozen contract."""
    # ---- ML side: enrich with TMDB dump metadata where matched ----
    ml_joined = ml.merge(
        dump[
            [
                "id",
                "genres_pipe",
                "original_language",
                "overview",
                "popularity",
                "release_date",
                "vote_average",
                "vote_count",
                "poster_path",
            ]
        ],
        left_on="tmdb_id",
        right_on="id",
        how="left",
        suffixes=("", "_dump"),
    )

    def _pick_genres(row):
        if row["ml_genres"]:
            return row["ml_genres"]
        if pd.notna(row.get("genres_pipe")) and row["genres_pipe"]:
            return row["genres_pipe"]
        return ""

    ml_rows = pd.DataFrame(
        {
            "tmdb_id": ml_joined["tmdb_id"],
            "ml_id": ml_joined["ml_id"],
            "title": ml_joined["title"],
            "year": ml_joined["year"],
            "language": ml_joined["original_language"].fillna("xx"),
            "genres": ml_joined.apply(_pick_genres, axis=1),
            "overview": ml_joined["overview"].fillna(""),
            "poster_path": ml_joined["poster_path"],
            "popularity": ml_joined["popularity"].fillna(0.0),
            "vote_average": ml_joined["vote_average"].fillna(0.0),
            "vote_count": ml_joined["vote_count"].fillna(0).astype("Int64"),
        }
    )
    ml_rows["language"] = ml_rows["language"].replace("", "xx")
    ml_rows["source"] = "ml"

    # ---- Indian side: ALL dump rows with Indian language + title + release_date ----
    title_ok = dump["title"].notna() & (dump["title"].str.strip() != "")
    date_ok = dump["release_date"].notna() & (dump["release_date"].str.strip() != "")
    lang_ok = dump["original_language"].isin(INDIAN_LANGS)
    indian_candidates = dump[title_ok & date_ok & lang_ok].copy()

    matched_tmdb_ids = set(ml_rows["tmdb_id"].dropna().astype(int).tolist())
    is_dupe = indian_candidates["id"].astype(int).isin(matched_tmdb_ids)

    # Movies present in both sets: fold into the ml row as source='both'.
    both_ids = set(indian_candidates.loc[is_dupe, "id"].astype(int).tolist())
    ml_rows.loc[
        ml_rows["tmdb_id"].notna() & ml_rows["tmdb_id"].astype("Int64").isin(both_ids),
        "source",
    ] = "both"

    indian_only = indian_candidates.loc[~is_dupe].copy()
    indian_rows = pd.DataFrame(
        {
            "tmdb_id": indian_only["id"].astype("Int64"),
            "ml_id": pd.array([pd.NA] * len(indian_only), dtype="Int64"),
            "title": indian_only["title"],
            "year": indian_only["release_date"].str.slice(0, 4).astype("Int64"),
            "language": indian_only["original_language"],
            "genres": indian_only["genres_pipe"].fillna(""),
            "overview": indian_only["overview"].fillna(""),
            "poster_path": indian_only["poster_path"],
            "popularity": indian_only["popularity"].fillna(0.0),
            "vote_average": indian_only["vote_average"].fillna(0.0),
            "vote_count": indian_only["vote_count"].fillna(0).astype("Int64"),
        }
    )
    indian_rows["source"] = "indian"

    combined = pd.concat([ml_rows, indian_rows], ignore_index=True)

    # CAT-2: is_indian is purely a function of language, independent of source.
    combined["is_indian"] = combined["language"].isin(INDIAN_LANGS).astype(int)

    # Normalize NULL-able columns.
    combined["poster_path"] = combined["poster_path"].where(
        combined["poster_path"].notna() & (combined["poster_path"].astype(str) != ""),
        None,
    )
    combined["title"] = combined["title"].fillna("").astype(str)
    combined["language"] = combined["language"].fillna("xx").astype(str)
    combined.loc[combined["language"] == "", "language"] = "xx"
    combined["overview"] = combined["overview"].fillna("").astype(str)
    combined["genres"] = combined["genres"].fillna("").astype(str)

    # Deterministic internal id: sort by (has_ml_id desc, ml_id, tmdb_id) so
    # rebuilds from the same cached inputs always assign the same ids.
    combined["_ml_sort"] = combined["ml_id"].fillna(10**9).astype("int64")
    combined["_tmdb_sort"] = combined["tmdb_id"].fillna(10**9).astype("int64")
    combined = combined.sort_values(["_ml_sort", "_tmdb_sort"]).reset_index(drop=True)
    combined["id"] = combined.index + 1
    combined = combined.drop(columns=["_ml_sort", "_tmdb_sort"])

    # Drop the small handful of malformed rows (empty title after cleaning).
    combined = combined[combined["title"].str.strip() != ""].reset_index(drop=True)
    combined["id"] = combined.index + 1

    return combined[
        [
            "id",
            "tmdb_id",
            "ml_id",
            "title",
            "year",
            "language",
            "genres",
            "overview",
            "poster_path",
            "popularity",
            "vote_average",
            "vote_count",
            "is_indian",
            "source",
        ]
    ]


# --------------------------------------------------------------------------- #
# Fixture selection (deterministic subset of the full catalog logic)
# --------------------------------------------------------------------------- #


def build_fixture_df(ml: pd.DataFrame, dump: pd.DataFrame) -> pd.DataFrame:
    ml_subset = ml[ml["ml_id"].isin(FIXTURE_ML_MOVIE_IDS)].copy()
    missing = set(FIXTURE_ML_MOVIE_IDS) - set(ml_subset["ml_id"].tolist())
    if missing:
        raise SystemExit(f"Fixture ml movieIds not found in movies.csv: {missing}")

    # Rows the ml subset needs for enrichment (poster/overview/language),
    # including 3 Idiots' tmdb_id which must also pass the Indian dump
    # filter below so it folds into source='both'.
    used_tmdb_ids = set(ml_subset["tmdb_id"].dropna().astype(int).tolist())
    ml_matched_dump_rows = dump[dump["id"].astype(int).isin(used_tmdb_ids)].copy()

    # Deterministic Indian-only picks: richest (vote_count desc, id asc) per
    # language quota, excluding any tmdb_id already used by the ml subset
    # (e.g. 3 Idiots' tmdb_id) so we don't accidentally duplicate a 'both' row.
    title_ok = dump["title"].notna() & (dump["title"].str.strip() != "")
    date_ok = dump["release_date"].notna() & (dump["release_date"].str.strip() != "")
    candidates = dump[title_ok & date_ok].copy()
    candidates = candidates[~candidates["id"].astype(int).isin(used_tmdb_ids)]

    picks = []
    for lang, quota in FIXTURE_INDIAN_LANG_QUOTAS.items():
        lang_rows = candidates[candidates["original_language"] == lang].copy()
        lang_rows = lang_rows.sort_values(
            ["vote_count", "id"], ascending=[False, True]
        )
        picks.append(lang_rows.head(quota))
    indian_dump_subset = pd.concat(picks, ignore_index=True)
    if len(indian_dump_subset) != sum(FIXTURE_INDIAN_LANG_QUOTAS.values()):
        raise SystemExit(
            "Fixture Indian quota not satisfiable from cached dump "
            f"(got {len(indian_dump_subset)}, wanted "
            f"{sum(FIXTURE_INDIAN_LANG_QUOTAS.values())})"
        )

    # Reuse the same shaping logic as the full build, restricted to the
    # tiny slices above (ml matches + the 10 Indian picks) so
    # build_catalog_df's join/dedupe/id-assignment behaves identically to a
    # full run (just on 25 rows). ml_matched_dump_rows supplies the
    # poster/overview/language enrichment for the 12 plain + 1 'both' ml row;
    # indian_dump_subset supplies the 10 Indian-only rows.
    dump_slice = pd.concat(
        [ml_matched_dump_rows, indian_dump_subset]
    ).drop_duplicates(subset="id")
    return build_catalog_df(ml_subset, dump_slice)


# --------------------------------------------------------------------------- #
# SQLite output
# --------------------------------------------------------------------------- #


def write_sqlite(
    df: pd.DataFrame,
    out_path: Path,
    meta_extra: dict[str, str],
    timestamp: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    conn = sqlite3.connect(out_path)
    try:
        conn.executescript(SCHEMA_SQL)

        records = df.to_dict(orient="records")

        def _n(v):
            """NaN/NA -> None for sqlite3 binding."""
            if v is None:
                return None
            try:
                if pd.isna(v):
                    return None
            except (TypeError, ValueError):
                pass
            return v

        rows = [
            (
                int(r["id"]),
                _n(r["tmdb_id"]) and int(r["tmdb_id"]),
                _n(r["ml_id"]) and int(r["ml_id"]),
                str(r["title"]),
                _n(r["year"]) and int(r["year"]),
                str(r["language"]),
                str(r["genres"]),
                str(r["overview"]),
                _n(r["poster_path"]),
                float(r["popularity"]) if _n(r["popularity"]) is not None else 0.0,
                float(r["vote_average"]) if _n(r["vote_average"]) is not None else 0.0,
                int(r["vote_count"]) if _n(r["vote_count"]) is not None else 0,
                int(r["is_indian"]),
                str(r["source"]),
            )
            for r in records
        ]
        conn.executemany(
            """
            INSERT INTO movies (
                id, tmdb_id, ml_id, title, year, language, genres, overview,
                poster_path, popularity, vote_average, vote_count, is_indian,
                source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        # Populate the external-content FTS5 index directly (one-shot build,
        # no live trigger maintenance needed).
        conn.execute(
            "INSERT INTO movies_fts(rowid, title) SELECT id, title FROM movies"
        )

        counts = {
            "total_rows": str(len(df)),
            "ml_rows": str(int((df["source"] == "ml").sum())),
            "indian_rows": str(int((df["source"] == "indian").sum())),
            "both_rows": str(int((df["source"] == "both").sum())),
            # E3-6: injectable/pinnable — wall clock in the artifact breaks
            # byte reproducibility (CAT-1).
            "build_timestamp": timestamp or _build_timestamp(),
        }
        meta = {**counts, **meta_extra}
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)", list(meta.items())
        )

        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def write_report(df: pd.DataFrame, out_path: Path) -> dict:
    indian_by_lang = (
        df[df["is_indian"] == 1]["language"].value_counts().sort_index().to_dict()
    )
    total = len(df)
    with_poster = int(df["poster_path"].notna().sum())
    report = {
        "total_rows": total,
        "ml_rows": int((df["source"] == "ml").sum()),
        "indian_rows": int((df["source"] == "indian").sum()),
        "both_rows": int((df["source"] == "both").sum()),
        "ml_sourced_total": int((df["source"].isin(["ml", "both"])).sum()),
        "indian_by_language": {k: int(v) for k, v in indian_by_lang.items()},
        "rows_with_poster": with_poster,
        "rows_with_poster_pct": round(100.0 * with_poster / total, 2) if total else 0.0,
        "build_timestamp": _build_timestamp(),  # E3-6
        "sources": {
            "ml_movies_url": ML_FILES["movies.csv"],
            "ml_links_url": ML_FILES["links.csv"],
            "tmdb_dump_url": TMDB_DUMP_URL,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Build the small deterministic fixtures/catalog-small.sqlite instead "
        "of the full data/catalog.sqlite.",
    )
    args = parser.parse_args()

    ensure_downloads()

    logger.info("Loading MovieLens movies.csv + links.csv ...")
    ml = load_ml_movies()
    logger.info("Loading TMDB dump (this can take ~30-60s for ~720k rows) ...")
    dump = load_tmdb_dump()

    meta_extra = {
        "ml_movies_url": ML_FILES["movies.csv"],
        "ml_links_url": ML_FILES["links.csv"],
        "tmdb_dump_url": TMDB_DUMP_URL,
        "schema_version": "1",
    }

    if args.fixture:
        logger.info("Building deterministic fixture (25 rows) ...")
        fixture_df = build_fixture_df(ml, dump)
        out_path = FIXTURES_DIR / "catalog-small.sqlite"
        write_sqlite(
            fixture_df,
            out_path,
            {**meta_extra, "fixture": "true"},
            timestamp=FIXTURE_BUILD_TIMESTAMP,  # E3-6: byte-stable fixture
        )
        # W12: pin the hand-written external ratings (+ deliberate NULLs).
        conn = sqlite3.connect(out_path)
        try:
            apply_fixture_ratings(conn)
            conn.commit()
        finally:
            conn.close()
        logger.info(
            "Wrote %s (%d rows: %d ml, %d indian, %d both)",
            out_path,
            len(fixture_df),
            int((fixture_df["source"] == "ml").sum()),
            int((fixture_df["source"] == "indian").sum()),
            int((fixture_df["source"] == "both").sum()),
        )
        return 0

    logger.info("Building full catalog ...")
    catalog_df = build_catalog_df(ml, dump)

    ml_sourced = int(catalog_df["source"].isin(["ml", "both"]).sum())
    if ml_sourced != 9742:
        logger.warning(
            "Expected 9,742 ml-sourced rows (ml+both), got %d — check the "
            "ml-latest-small mirror for drift.",
            ml_sourced,
        )
    indian_total = int((catalog_df["is_indian"] == 1).sum())
    logger.info("ml-sourced rows (ml+both): %d; indian rows: %d", ml_sourced, indian_total)

    out_path = DATA_DIR / "catalog.sqlite"
    write_sqlite(catalog_df, out_path, meta_extra)
    logger.info("Wrote %s (%d rows)", out_path, len(catalog_df))

    report_path = DATA_DIR / "catalog-report.json"
    report = write_report(catalog_df, report_path)
    logger.info("Wrote %s", report_path)
    logger.info(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
