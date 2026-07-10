"""TMDB REST API client — the sole network boundary.

Responsibilities:
    - Read API key from env var **only** (SEC-1). Never logged.
    - Fail fast at startup if key is missing/invalid (FAIL-1).
    - Retry with exponential backoff on timeout/5xx, then skip (FAIL-2).
    - Honor HTTP 429 ``Retry-After`` (FAIL-3).
    - Coerce malformed/missing fields to empty content features (FAIL-4).
    - Cache responses to disk; cache hit → no network call (SEC-2).
    - Detect corrupt cache entries and re-fetch only affected (FAIL-7).
    - Support cache-only mode (``api_key=None``) for fixture/offline use.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import requests

from recsys.config import (
    TMDB_BASE_URL,
    TMDB_BACKOFF_FACTOR,
    TMDB_MAX_RETRIES,
    TMDB_RATE_LIMIT_DELAY,
    TMDB_TIMEOUT,
    TMDB_CACHE_SUBDIR,
    FIXTURE_TMDB_CACHE,
)

logger = logging.getLogger(__name__)


class TMDBError(Exception):
    """Raised when the TMDB client cannot proceed (e.g., missing key)."""


def _empty_features() -> dict:
    """The canonical empty content-features dict (FAIL-4)."""
    return {
        "genres": [],
        "keywords": [],
        "cast": [],
        "crew": [],
        "overview": "",
    }


class TMDBClient:
    """Sole boundary to the TMDB REST API. Fully mockable for tests."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        max_retries: int = TMDB_MAX_RETRIES,
        backoff_factor: float = TMDB_BACKOFF_FACTOR,
        timeout: int = TMDB_TIMEOUT,
        rate_limit_delay: float = TMDB_RATE_LIMIT_DELAY,
    ):
        self._api_key = api_key  # None → cache-only mode
        self._cache_dir = Path(cache_dir) if cache_dir else TMDB_CACHE_SUBDIR
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._timeout = timeout
        self._rate_limit_delay = rate_limit_delay
        self._session = requests.Session()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_movie(self, tmdb_id: int) -> dict | None:
        """Fetch content features for a TMDB movie ID.

        Returns a dict with keys ``genres, keywords, cast, crew, overview``,
        or ``None`` if the movie could not be fetched (skipped after retries).

        Cache hits do **no** network call (SEC-2). Corrupt cache entries are
        detected and re-fetched (FAIL-7).
        """
        # 1. Check cache
        cached = self._load_cache(tmdb_id)
        if cached is not None:
            return cached

        # 2. Cache miss — need network
        if self._api_key is None:
            logger.warning(
                "Cache miss for tmdbId=%s and no API key configured; skipping",
                tmdb_id,
            )
            return None

        # 3. Fetch with retry/backoff
        raw = self._fetch_with_retry(tmdb_id)
        if raw is None:
            return None  # skipped after retries (FAIL-2)

        # 4. Extract + coerce features (FAIL-4)
        features = self._extract_features(raw)

        # 5. Write to cache
        self._write_cache(tmdb_id, features)

        # 6. Rate-limit pause
        time.sleep(self._rate_limit_delay)

        return features

    # ------------------------------------------------------------------ #
    # Cache layer (FAIL-7, SEC-2)
    # ------------------------------------------------------------------ #

    def _cache_path(self, tmdb_id: int) -> Path:
        return self._cache_dir / f"{tmdb_id}.json"

    def _load_cache(self, tmdb_id: int) -> dict | None:
        """Load a cache entry; return ``None`` if missing or corrupt (FAIL-7)."""
        path = self._cache_path(tmdb_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("cache entry is not a dict")
            # Validate required keys exist (coerce if missing)
            for key in ("genres", "keywords", "cast", "crew", "overview"):
                if key not in data:
                    raise ValueError(f"cache entry missing key '{key}'")
            return data
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Corrupt cache entry for tmdbId=%s (%s); will re-fetch (FAIL-7)",
                tmdb_id, exc,
            )
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def _write_cache(self, tmdb_id: int, features: dict) -> None:
        path = self._cache_path(tmdb_id)
        path.write_text(
            json.dumps(features, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Network layer (FAIL-1, FAIL-2, FAIL-3)
    # ------------------------------------------------------------------ #

    def _fetch_with_retry(self, tmdb_id: int) -> dict | None:
        """Fetch raw TMDB response with exponential backoff.

        On 429: honor ``Retry-After`` header (FAIL-3).
        On 5xx/timeout: exponential backoff, then skip (FAIL-2).
        On 404: skip (movie doesn't exist) (FAIL-5).
        """
        url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
        params = {
            "api_key": self._api_key,
            "append_to_response": "keywords,credits",
        }

        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(
                    url, params=params, timeout=self._timeout
                )
                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 404:
                    logger.warning(
                        "TMDB returned 404 for tmdbId=%s; skipping (FAIL-5)",
                        tmdb_id,
                    )
                    return None

                if resp.status_code == 401:
                    raise TMDBError(
                        "TMDB API key was rejected (HTTP 401). "
                        "Check TMDB_API_KEY (FAIL-1)."
                    )

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After", "1")
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = 1.0
                    logger.warning(
                        "TMDB rate-limited (429) for tmdbId=%s; "
                        "waiting %ss (FAIL-3)",
                        tmdb_id, wait,
                    )
                    time.sleep(wait)
                    continue

                if 500 <= resp.status_code < 600:
                    wait = self._backoff_factor ** attempt
                    logger.warning(
                        "TMDB %d for tmdbId=%s; retry in %.1fs (FAIL-2, attempt %d/%d)",
                        resp.status_code, tmdb_id, wait,
                        attempt + 1, self._max_retries,
                    )
                    time.sleep(wait)
                    continue

                # Other 4xx — non-retryable
                logger.warning(
                    "TMDB returned %d for tmdbId=%s; skipping",
                    resp.status_code, tmdb_id,
                )
                return None

            except requests.exceptions.Timeout:
                wait = self._backoff_factor ** attempt
                logger.warning(
                    "TMDB timeout for tmdbId=%s; retry in %.1fs (FAIL-2, attempt %d/%d)",
                    tmdb_id, wait, attempt + 1, self._max_retries,
                )
                time.sleep(wait)
            except requests.exceptions.RequestException as exc:
                wait = self._backoff_factor ** attempt
                # SEC-1: log the exception *class* only. ``requests`` embeds the
                # full request URL (including the ``api_key`` query param) in the
                # exception message, so ``str(exc)`` must never reach the log.
                logger.warning(
                    "TMDB request error for tmdbId=%s: %s; retry in %.1fs (FAIL-2, attempt %d/%d)",
                    tmdb_id, type(exc).__name__, wait,
                    attempt + 1, self._max_retries,
                )
                time.sleep(wait)

        logger.warning(
            "TMDB fetch exhausted %d retries for tmdbId=%s; skipping (FAIL-2)",
            self._max_retries, tmdb_id,
        )
        return None

    # ------------------------------------------------------------------ #
    # Feature extraction (FAIL-4)
    # ------------------------------------------------------------------ #

    def _extract_features(self, raw: dict) -> dict:
        """Extract and coerce content features from a raw TMDB response.

        Missing/malformed fields are coerced to empty (FAIL-4). The API key
        is never included in the extracted data (SEC-1).
        """
        features = _empty_features()

        # Genres
        try:
            genres = raw.get("genres", [])
            if isinstance(genres, list):
                features["genres"] = [
                    g.get("name", "") for g in genres if isinstance(g, dict)
                ]
        except (TypeError, AttributeError):
            pass

        # Keywords
        try:
            kw_obj = raw.get("keywords", {})
            if isinstance(kw_obj, dict):
                kw_list = kw_obj.get("keywords", [])
            elif isinstance(kw_obj, list):
                kw_list = kw_obj
            else:
                kw_list = []
            features["keywords"] = [
                k.get("name", "") for k in kw_list if isinstance(k, dict)
            ]
        except (TypeError, AttributeError):
            pass

        # Cast
        try:
            credits = raw.get("credits", {})
            if isinstance(credits, dict):
                cast_list = credits.get("cast", [])
            elif isinstance(credits, list):
                cast_list = credits
            else:
                cast_list = []
            features["cast"] = [
                c.get("name", "") for c in cast_list[:20] if isinstance(c, dict)
            ]
        except (TypeError, AttributeError):
            pass

        # Crew (directors + writers only)
        try:
            credits = raw.get("credits", {})
            if isinstance(credits, dict):
                crew_list = credits.get("crew", [])
            elif isinstance(credits, list):
                crew_list = credits
            else:
                crew_list = []
            features["crew"] = [
                c.get("name", "")
                for c in crew_list
                if isinstance(c, dict) and c.get("job") in ("Director", "Writer")
            ]
        except (TypeError, AttributeError):
            pass

        # Overview
        try:
            overview = raw.get("overview", "")
            if isinstance(overview, str):
                features["overview"] = overview
        except (TypeError, AttributeError):
            pass

        return features
