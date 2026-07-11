"""W3 (#16) — FastAPI server skeleton for the taste web UI.

Rules enforced here:
- Data mode is explicit (``fixture`` | ``real``) and prerequisites are
  checked at construction time — fail fast, never a silent swap (WEB-FAIL-1).
- Bad query input answers 4xx, never 500 (WEB-FAIL-2).
- The TMDB API key is held server-side only and never serialized into any
  response (WEB-SEC-2); posters are pre-resolved CDN URLs from the cache.
- No fitted/ML model imports in the serving path (MATH-3): this module may
  import recsys.data and recsys.taste, never recsys.recommenders.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from recsys.config import (
    FIXTURE_TMDB_CACHE,
    PROFILE_PATH,
    TMDB_CACHE_SUBDIR,
)
from recsys.data import load_ratings
from recsys.web.profile_store import ProfileStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
SEARCH_MAX_QUERY = 200
SEARCH_MAX_RESULTS = 20

_MODES = ("fixture", "real")


def create_app(
    mode: str,
    *,
    data_dir: Path | None = None,
    profile_path: Path | None = None,
    tmdb_cache_dir: Path | None = None,
    tmdb_api_key: str | None = None,
) -> FastAPI:
    """Build the app. Raises on missing prerequisites (WEB-FAIL-1)."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    # Fail fast at startup, not on first request.
    data = load_ratings(fixture=(mode == "fixture"), data_dir=data_dir)
    cache_dir = tmdb_cache_dir or (
        FIXTURE_TMDB_CACHE if mode == "fixture" else TMDB_CACHE_SUBDIR
    )
    store = ProfileStore(profile_path or PROFILE_PATH)

    # Precompute the search table once (title lowercased for matching).
    movies = data.movies[["movieId", "title", "genres"]].copy()
    movies["_lower"] = movies["title"].str.lower()

    app = FastAPI(title="Taste Engine", docs_url=None, redoc_url=None)
    # Key stays in the closure; it is deliberately NOT stored on app.state
    # so no handler can accidentally serialize it (WEB-SEC-2).
    del tmdb_api_key

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status() -> dict:
        profile = store.load()
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
                "quarantined": profile.quarantined_to is not None,
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
                "genres": [] if row.genres == "(no genres listed)"
                else row.genres.split("|"),
            }
            for row in hits.head(SEARCH_MAX_RESULTS).itertuples()
        ]
        return {"results": results}

    return app
