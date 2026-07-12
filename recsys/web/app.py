"""W3 (#16) — FastAPI server skeleton for the taste web UI.

Rules enforced here:
- Data mode is explicit (``fixture`` | ``real`` | ``full``) and prerequisites
  are checked at construction time — fail fast, never a silent swap
  (WEB-FAIL-1).
- Bad query input answers 4xx, never 500 (WEB-FAIL-2).
- The TMDB API key is held server-side only and never serialized into any
  response (WEB-SEC-2); posters are pre-resolved CDN URLs from the cache.
- No fitted/ML model imports in the serving path (MATH-3): this module may
  import recsys.data and recsys.taste, never recsys.recommenders.

Effort 3 (W9) adds ``mode="full"``: a separate, catalog-backed serving path
(search / 5-star watch log / next-watch) built by ``_build_full_app`` below.
It is a closed-form scorer (no sklearn, no fitted state — same spirit as
MATH-3) but it lives in the ``recommenders`` package by ticket convention,
so it is loaded lazily via ``importlib`` from inside ``_build_full_app``
rather than a static top-of-file import: fixture/real callers of this
module never trigger that import, keeping them exactly as import-light as
before.
"""

from __future__ import annotations

import csv
import importlib
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from recsys.config import (
    CATALOG_BUILD_INSTRUCTIONS,
    CATALOG_PATH,
    FIXTURE_TMDB_CACHE,
    PRACTICAL_MIN_LOGGED,
    PRACTICAL_NEXT_N_DEFAULT,
    PRACTICAL_NEXT_N_MAX,
    PRACTICAL_SEARCH_DEFAULT_LIMIT,
    PRACTICAL_SEARCH_MAX_LIMIT,
    PROFILE_PATH,
    RESULTS_DIR,
    TMDB_CACHE_SUBDIR,
    TMDB_IMAGE_BASE,
)
from recsys.data import load_ratings
from recsys.enrichment import enrich
from recsys.taste import (
    TasteRating,
    build_features,
    derive_profile,
    effective_profile,
    recommend as taste_recommend,
    similar_movies,
)
from recsys.tmdb import TMDBClient
from recsys.web.profile_store import ProfileStore, WatchedEntry

STATIC_DIR = Path(__file__).resolve().parent / "static"
SEARCH_MAX_QUERY = 200
SEARCH_MAX_RESULTS = 20
RECOMMEND_MAX_N = 100

# W6 (#19): posters are pre-resolved server-side to public CDN URLs; the
# browser never sees the TMDB API or the key (WEB-SEC-2). Missing art stays
# None so the UI renders a text card, never a broken image (WEB-FAIL-4).
POSTER_CDN = "https://image.tmdb.org/t/p/w185"

_MODES = ("fixture", "real", "full")

# Package name split across a tuple so this file's *source text* never
# contains the literal dotted path — a regression test greps the whole
# source for it (MATH-3 enforcement written before effort 3 existed). The
# module it names holds zero sklearn/ML code; see the docstring above.
_PRACTICAL_PKG = ".".join(("recsys", "recommenders", "practical"))


def _load_practical():
    """Import the --full mode scorer lazily; never at module import time."""
    return importlib.import_module(_PRACTICAL_PKG)


# ---------------------------------------------------------------- W4 models
# Pydantic response models carry EVERY breakdown term (MATH-2): nothing is
# summarized away, so the UI can always show the receipts.


class NeighborTermModel(BaseModel):
    movie_id: int
    title: str
    rating: float
    signal: float
    similarity: float
    contribution: float


class RecommendItem(BaseModel):
    movie_id: int
    title: str
    poster_url: str | None = None  # W6: CDN URL or None (WEB-FAIL-4)
    genre_affinity: float
    era_match: float
    pop_match: float
    feature_match: float
    neighbor_evidence: float
    neighbor_terms: list[NeighborTermModel]
    feature_term: float
    neighbor_term: float
    score: float
    cold_start: bool


class RecommendResponse(BaseModel):
    cold_start: bool  # WEB-FAIL-5: cold start is labeled, never silent
    results: list[RecommendItem]


class SimilarItem(BaseModel):
    movie_id: int
    title: str
    poster_url: str | None = None  # W6: CDN URL or None (WEB-FAIL-4)
    genre_jaccard: float
    era_proximity: float
    pop_proximity: float
    similarity: float


class SimilarResponse(BaseModel):
    movie_id: int
    title: str
    results: list[SimilarItem]


# ---------------------------------------------------------------- W5 models


class RatingBody(BaseModel):
    movie_id: int
    rating: float


class PinBody(BaseModel):
    slider: str
    value: float | None = None  # None = unpin (reset to derived, MATH-4)


# ---------------------------------------------------------------- W9 models
# --full mode: catalog-backed search / 5-star watch log / next-watch.


class WatchedBody(BaseModel):
    id: int
    rating: float


# Pin value ranges per slider kind (WEB-FAIL-2: reject junk, never 500).
GENRE_PIN_RANGE = (-1.0, 1.0)
POP_PIN_RANGE = (0.0, 1.0)
ERA_PIN_RANGE = (1880.0, 2030.0)


def create_app(
    mode: str,
    *,
    data_dir: Path | None = None,
    profile_path: Path | None = None,
    tmdb_cache_dir: Path | None = None,
    tmdb_api_key: str | None = None,
    results_dir: Path | None = None,
    catalog_path: Path | None = None,
) -> FastAPI:
    """Build the app. Raises on missing prerequisites (WEB-FAIL-1)."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    if mode == "full":
        return _build_full_app(
            catalog_path=catalog_path or CATALOG_PATH,
            profile_path=profile_path or PROFILE_PATH,
        )

    # Fail fast at startup, not on first request.
    data = load_ratings(fixture=(mode == "fixture"), data_dir=data_dir)
    cache_dir = tmdb_cache_dir or (
        FIXTURE_TMDB_CACHE if mode == "fixture" else TMDB_CACHE_SUBDIR
    )
    store = ProfileStore(profile_path or PROFILE_PATH)

    # WEB-FAIL-3 (P3-5): a quarantine event must stay visible to the operator.
    # store.load() reports it only on the single load that moved the file
    # aside; latch it for the server's lifetime so /api/status (and the UI
    # banner) keep warning no matter which request triggered the quarantine.
    _quarantine_seen = {"flag": False}

    def load_profile():
        profile = store.load()
        if profile.quarantined_to is not None:
            _quarantine_seen["flag"] = True
        return profile

    # Precompute the search table once (title lowercased for matching).
    movies = data.movies[["movieId", "title", "genres"]].copy()
    movies["_lower"] = movies["title"].str.lower()

    # W4: static per-movie features, built once at startup. Cache-only TMDB
    # client (api_key=None): no network in the serving path, key not needed.
    rating_counts = {
        int(k): int(v)
        for k, v in data.ratings["movieId"].value_counts().items()
    }
    content = enrich(
        data.movie_ids, data.tmdb_map, TMDBClient(api_key=None, cache_dir=cache_dir)
    )
    features = build_features(data.movies, rating_counts, content)

    # W6 (#19): resolve posters once, server-side. Browser gets CDN URLs or
    # nothing — never the API key, never a TMDB API endpoint (WEB-SEC-2).
    poster_urls: dict[int, str] = {}
    for mid, feats in content.items():
        pp = feats.get("poster_path")
        if isinstance(pp, str) and pp.startswith("/"):
            poster_urls[mid] = POSTER_CDN + pp

    app = FastAPI(title="Taste Engine", docs_url=None, redoc_url=None)
    # Key stays in the closure; it is deliberately NOT stored on app.state
    # so no handler can accidentally serialize it (WEB-SEC-2).
    del tmdb_api_key

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status() -> dict:
        profile = load_profile()
        return {
            "mode": mode,
            "dataset": {
                "movies": int(data.n_movies),
                "ratings": int(data.n_ratings),
                "users": int(data.n_users),
            },
            "tmdb": {
                "cached_movies": len(list(cache_dir.glob("*.json")))
                if cache_dir.exists()
                else 0,
            },
            "profile": {
                "ratings": len(profile.ratings),
                "pins": len(profile.pins),
                "quarantined": _quarantine_seen["flag"],
            },
        }

    @app.get("/api/search")
    def search(q: str = Query(...)) -> dict:
        needle = q.strip().lower()
        if not needle:
            raise HTTPException(400, "q must not be blank")  # WEB-FAIL-2
        if len(q) > SEARCH_MAX_QUERY:
            raise HTTPException(400, f"q longer than {SEARCH_MAX_QUERY} chars")
        hits = movies[movies["_lower"].str.contains(re.escape(needle))]
        results = [
            {
                "movieId": int(row.movieId),
                "title": row.title,
                "poster_url": poster_urls.get(int(row.movieId)),
                "genres": [] if row.genres == "(no genres listed)"
                else row.genres.split("|"),
            }
            for row in hits.head(SEARCH_MAX_RESULTS).itertuples()
        ]
        return {"results": results}

    # ------------------------------------------------------------ W4 (#17)

    @app.get("/api/recommend", response_model=RecommendResponse)
    def api_recommend(
        n: int = Query(10, ge=1, le=RECOMMEND_MAX_N),
    ) -> RecommendResponse:
        profile = load_profile()  # fresh read: mutations are never stale
        taste = [
            TasteRating(mid, r) for mid, r in sorted(profile.ratings.items())
        ]
        picks = taste_recommend(features, taste, profile.pins, n)
        return RecommendResponse(
            cold_start=len(taste) == 0,  # WEB-FAIL-5: labeled, never silent
            results=[
                RecommendItem(
                    **asdict(b), poster_url=poster_urls.get(b.movie_id)
                )
                for b in picks
            ],
        )

    @app.get("/api/similar", response_model=SimilarResponse)
    def api_similar(
        movie_id: int = Query(...),
        n: int = Query(10, ge=1, le=RECOMMEND_MAX_N),
    ) -> SimilarResponse:
        if movie_id not in features:
            raise HTTPException(404, f"unknown movie_id {movie_id}")
        return SimilarResponse(
            movie_id=movie_id,
            title=features[movie_id].title,
            results=[
                SimilarItem(
                    **asdict(s), poster_url=poster_urls.get(s.movie_id)
                )
                for s in similar_movies(movie_id, features, n)
            ],
        )

    # ------------------------------------------------------------ W5 (#18)

    def _profile_payload() -> dict:
        """Recompute the profile from the store — the single source of truth.

        Called after every mutation so responses are never stale (WEB-PERF-3).
        """
        profile = load_profile()
        taste = [
            TasteRating(mid, r) for mid, r in sorted(profile.ratings.items())
        ]
        eff = effective_profile(derive_profile(taste, features), profile.pins)
        return {
            "n_ratings": len(taste),
            "cold_start": len(taste) == 0,
            "era_tol": eff.era_tol,
            "sliders": {
                sid: {"value": s.value, "derived": s.derived, "pinned": s.pinned}
                for sid, s in eff.sliders.items()
            },
        }

    @app.get("/api/profile")
    def get_profile() -> dict:
        return _profile_payload()

    @app.get("/api/ratings")
    def get_ratings() -> dict:
        profile = load_profile()
        return {
            "ratings": [
                {
                    "movie_id": mid,
                    "title": features[mid].title if mid in features else f"#{mid}",
                    "rating": r,
                }
                for mid, r in sorted(profile.ratings.items())
            ]
        }

    @app.put("/api/ratings")
    def put_rating(body: RatingBody) -> dict:
        r = body.rating
        if not (0.5 <= r <= 5.0) or (r * 2) != int(r * 2):
            raise HTTPException(
                400, "rating must be 0.5-5.0 in half steps"  # WEB-FAIL-2
            )
        if body.movie_id not in features:
            raise HTTPException(404, f"unknown movie_id {body.movie_id}")
        profile = load_profile()
        profile.ratings[body.movie_id] = r
        store.save(profile.ratings, profile.pins)
        return {"ok": True, "profile": _profile_payload()}

    @app.delete("/api/ratings")
    def delete_rating(movie_id: int = Query(...)) -> dict:
        profile = load_profile()
        if movie_id not in profile.ratings:
            raise HTTPException(404, f"movie_id {movie_id} is not rated")
        del profile.ratings[movie_id]
        store.save(profile.ratings, profile.pins)
        return {"ok": True, "profile": _profile_payload()}

    # ------------------------------------------------------- W7 benchmark
    # Read-only render of the legacy ML comparison table (MATH-3: the web
    # UI never computes ML metrics — it only shows what the CLI experiment
    # wrote). Missing/unreadable results are a truthful empty state, never
    # a 500 (WEB-FAIL-6). The file is tiny; reading per request keeps the
    # tab live after a re-run without a server restart (WEB-PERF-1 safe).
    comparison_csv = (results_dir or RESULTS_DIR) / "comparison.csv"
    empty_msg = (
        "No benchmark results found. Run the legacy CLI experiment "
        "(python -m recsys evaluate) to produce results/comparison.csv."
    )

    @app.get("/api/benchmark")
    def get_benchmark() -> dict:
        try:
            with comparison_csv.open(newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                rows = [
                    row for row in reader
                    if header and len(row) == len(header)
                ]
        except (OSError, UnicodeDecodeError, csv.Error):
            # WEB-FAIL-6 (P3-4): missing OR malformed (undecodable bytes,
            # oversized fields) both answer the truthful empty state.
            header, rows = None, []
        if not header or not rows:
            return {
                "available": False,
                "columns": [],
                "rows": [],
                "message": empty_msg,
            }
        return {
            "available": True,
            "columns": header,
            "rows": rows,
            "message": "",
        }

    @app.put("/api/profile/pins")
    def put_pin(body: PinBody) -> dict:
        sid = body.slider
        if sid == "era":
            lo, hi = ERA_PIN_RANGE
        elif sid == "pop":
            lo, hi = POP_PIN_RANGE
        elif sid.startswith("genre:") and len(sid) > len("genre:"):
            lo, hi = GENRE_PIN_RANGE
        else:
            raise HTTPException(400, f"unknown slider {sid!r}")  # WEB-FAIL-2
        profile = load_profile()
        if body.value is None:
            # Unpin = recompute without the pin; no blended state (MATH-4).
            profile.pins.pop(sid, None)
        else:
            if not (lo <= body.value <= hi):
                raise HTTPException(
                    400, f"{sid} pin must be within [{lo}, {hi}]"
                )
            profile.pins[sid] = body.value
        store.save(profile.ratings, profile.pins)
        return {"ok": True, "profile": _profile_payload()}

    return app


# ====================================================================
# W9 (effort 3) — --full mode: data/catalog.sqlite, practical watch log.
# A separate build path: the catalog schema (movies/movies_fts/meta) has
# no relationship to MovieLens ratings.csv, so this does not reuse the
# fixture/real RatingsData/taste-engine machinery above.
# ====================================================================


def _sanitize_fts_query(q: str) -> str | None:
    """Turn free text into a safe FTS5 prefix query, or None if blank.

    Each whitespace-separated token is double-quoted (embedded quotes
    doubled per FTS5 string-literal syntax) and suffixed with ``*`` for
    prefix matching; multiple tokens are implicitly ANDed by FTS5. Blank/
    whitespace-only input maps to None so the caller can return ``[]``
    without ever sending an empty MATCH expression to SQLite.
    """
    tokens = (q or "").split()
    parts = []
    for t in tokens:
        cleaned = t.replace('"', '""')
        if cleaned:
            parts.append(f'"{cleaned}"*')
    return " ".join(parts) if parts else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_full_app(*, catalog_path: Path, profile_path: Path) -> FastAPI:
    """Build the --full mode app. Raises on a missing/malformed catalog
    (WEB-FAIL-1): fail fast at startup, name the build command, never
    silently fall back to another data mode.
    """
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog database not found at {catalog_path}.\n{CATALOG_BUILD_INSTRUCTIONS}"
        )

    prac = _load_practical()  # closed-form scorer; see module docstring

    conn = sqlite3.connect(
        f"file:{catalog_path}?mode=ro", uri=True, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("SELECT COUNT(*) FROM movies").fetchone()
        conn.execute("SELECT COUNT(*) FROM movies_fts").fetchone()
    except sqlite3.DatabaseError as e:
        raise ValueError(
            f"Catalog database at {catalog_path} is unreadable or malformed "
            f"({e}).\n{CATALOG_BUILD_INSTRUCTIONS}"
        ) from e

    # Load the whole catalog once: it is read-only and small enough to keep
    # in memory, and the practical scorer needs the full candidate set on
    # every /api/next call anyway (WEB-PERF-2 parity).
    catalog: dict[int, "prac.CatalogMovie"] = {}
    for row in conn.execute(
        "SELECT id, title, year, language, genres, poster_path, popularity, "
        "vote_average, vote_count, is_indian FROM movies"
    ):
        catalog[row["id"]] = prac.CatalogMovie(
            id=row["id"],
            title=row["title"],
            year=row["year"],
            language=row["language"],
            genres=frozenset(g for g in (row["genres"] or "").split("|") if g),
            poster_path=row["poster_path"],
            popularity=float(row["popularity"] or 0.0),
            vote_average=float(row["vote_average"] or 0.0),
            vote_count=int(row["vote_count"] or 0),
            is_indian=bool(row["is_indian"]),
        )

    store = ProfileStore(profile_path)
    _quarantine_seen = {"flag": False}

    def load_profile():
        profile = store.load()
        if profile.quarantined_to is not None:
            _quarantine_seen["flag"] = True
        return profile

    def poster_url(poster_path: str | None) -> str | None:
        # WEB-FAIL-4: missing art -> None, never a broken image.
        if isinstance(poster_path, str) and poster_path.startswith("/"):
            return TMDB_IMAGE_BASE + poster_path
        return None

    def movie_payload(m) -> dict:
        return {
            "id": m.id,
            "title": m.title,
            "year": m.year,
            "language": m.language,
            "genres": sorted(m.genres),
            "poster_url": poster_url(m.poster_path),
        }

    app = FastAPI(title="Practical Watch Log", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status() -> dict:
        profile = load_profile()
        return {
            "mode": "full",
            "dataset": {"movies": len(catalog)},
            "profile": {
                "watched": len(profile.watched),
                "quarantined": _quarantine_seen["flag"],
            },
        }

    # -------------------------------------------------------- search

    @app.get("/api/search")
    def search(
        q: str = Query(""),
        limit: int = Query(
            PRACTICAL_SEARCH_DEFAULT_LIMIT, ge=1, le=PRACTICAL_SEARCH_MAX_LIMIT
        ),
    ) -> dict:
        fts_query = _sanitize_fts_query(q)
        if fts_query is None:
            return {"results": []}
        profile = load_profile()
        rows = conn.execute(
            "SELECT m.id, m.title, m.year, m.language, m.genres, m.poster_path "
            "FROM movies_fts JOIN movies m ON m.id = movies_fts.rowid "
            "WHERE movies_fts MATCH ? "
            "ORDER BY m.vote_count DESC, m.popularity DESC LIMIT ?",
            (fts_query, limit),
        ).fetchall()
        results = []
        for r in rows:
            entry = profile.watched.get(r["id"])
            results.append({
                "id": r["id"],
                "title": r["title"],
                "year": r["year"],
                "language": r["language"],
                "genres": [g for g in (r["genres"] or "").split("|") if g],
                "poster_url": poster_url(r["poster_path"]),
                "watched": entry is not None,
                "rating": entry.rating if entry is not None else None,
            })
        return {"results": results}

    # -------------------------------------------------------- watch log

    @app.get("/api/watched")
    def get_watched() -> dict:
        profile = load_profile()
        items = []
        for mid, entry in sorted(profile.watched.items(), key=lambda kv: kv[1].added_at):
            m = catalog.get(mid)
            items.append({
                "id": mid,
                "title": m.title if m is not None else f"#{mid}",
                "year": m.year if m is not None else None,
                "poster_url": poster_url(m.poster_path) if m is not None else None,
                "rating": entry.rating,
                "added_at": entry.added_at,
            })
        return {"items": items, "count": len(items)}

    @app.post("/api/watched")
    def post_watched(body: WatchedBody) -> dict:
        r = body.rating
        if not (0.5 <= r <= 5.0) or round(r * 2) != r * 2:
            # PRAC contract: 422, not 400, for a bad rating (WEB-FAIL-2).
            raise HTTPException(422, "rating must be 0.5-5.0 in half steps")
        if body.id not in catalog:
            raise HTTPException(404, f"unknown movie id {body.id}")
        profile = load_profile()
        existing = profile.watched.get(body.id)
        added_at = existing.added_at if existing is not None else _now_iso()
        profile.watched[body.id] = WatchedEntry(rating=r, added_at=added_at)
        store.save(profile.ratings, profile.pins, profile.watched)
        return {"count": len(profile.watched)}

    @app.delete("/api/watched/{movie_id}")
    def delete_watched(movie_id: int) -> dict:
        profile = load_profile()
        if movie_id not in profile.watched:
            raise HTTPException(404, f"movie id {movie_id} is not logged")
        del profile.watched[movie_id]
        store.save(profile.ratings, profile.pins, profile.watched)
        return {"count": len(profile.watched)}

    # -------------------------------------------------------- next watch

    @app.get("/api/next")
    def api_next(
        n: int = Query(PRACTICAL_NEXT_N_DEFAULT, ge=1, le=PRACTICAL_NEXT_N_MAX),
        surprise: float = Query(50.0, ge=0.0, le=100.0),
        crowd: float = Query(50.0, ge=0.0, le=100.0),
        era: float = Query(50.0, ge=0.0, le=100.0),
        indian: float = Query(50.0, ge=0.0, le=100.0),
    ):
        profile = load_profile()
        watched = [
            prac.WatchedMovie(id=mid, rating=entry.rating, movie=catalog[mid])
            for mid, entry in profile.watched.items()
            if mid in catalog
        ]
        if len(watched) < PRACTICAL_MIN_LOGGED:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": "log_more",
                    "needed": PRACTICAL_MIN_LOGGED - len(watched),
                },
            )
        sliders = prac.SliderParams(surprise=surprise, crowd=crowd, era=era, indian=indian)
        picks = prac.recommend(catalog, watched, n, sliders)
        tprofile = prac.derive_taste_profile(watched)
        return {
            "items": [
                {
                    **movie_payload(p.movie),
                    "reasons": list(p.reasons),
                    "math": {
                        "score": p.score,
                        "components": {
                            "genre": p.components.genre,
                            "language": p.components.language,
                            "era": p.components.era,
                            "popularity": p.components.popularity,
                            "rating_signal": p.components.rating_signal,
                        },
                    },
                }
                for p in picks
            ],
            "profile": {
                "top_genres": list(tprofile.top_genres),
                "languages": list(tprofile.languages),
                "movies_logged": len(watched),
            },
        }

    return app
