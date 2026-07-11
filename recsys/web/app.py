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
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from recsys.config import (
    FIXTURE_TMDB_CACHE,
    PROFILE_PATH,
    TMDB_CACHE_SUBDIR,
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
from recsys.web.profile_store import ProfileStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
SEARCH_MAX_QUERY = 200
SEARCH_MAX_RESULTS = 20
RECOMMEND_MAX_N = 100

_MODES = ("fixture", "real")


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

    # ------------------------------------------------------------ W4 (#17)

    @app.get("/api/recommend", response_model=RecommendResponse)
    def api_recommend(
        n: int = Query(10, ge=1, le=RECOMMEND_MAX_N),
    ) -> RecommendResponse:
        profile = store.load()  # fresh read: mutations are never stale
        taste = [
            TasteRating(mid, r) for mid, r in sorted(profile.ratings.items())
        ]
        picks = taste_recommend(features, taste, profile.pins, n)
        return RecommendResponse(
            cold_start=len(taste) == 0,  # WEB-FAIL-5: labeled, never silent
            results=[RecommendItem(**asdict(b)) for b in picks],
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
                SimilarItem(**asdict(s))
                for s in similar_movies(movie_id, features, n)
            ],
        )

    # ------------------------------------------------------------ W5 (#18)

    def _profile_payload() -> dict:
        """Recompute the profile from the store — the single source of truth.

        Called after every mutation so responses are never stale (WEB-PERF-3).
        """
        profile = store.load()
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
        profile = store.load()
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
        profile = store.load()
        profile.ratings[body.movie_id] = r
        store.save(profile.ratings, profile.pins)
        return {"ok": True, "profile": _profile_payload()}

    @app.delete("/api/ratings")
    def delete_rating(movie_id: int = Query(...)) -> dict:
        profile = store.load()
        if movie_id not in profile.ratings:
            raise HTTPException(404, f"movie_id {movie_id} is not rated")
        del profile.ratings[movie_id]
        store.save(profile.ratings, profile.pins)
        return {"ok": True, "profile": _profile_payload()}

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
        profile = store.load()
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
