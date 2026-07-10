"""Enrichment: map each MovieLens movie to TMDB content features via tmdbId.

Movies with no ``tmdbId`` or no TMDB match carry empty content features (FAIL-5).
"""

from __future__ import annotations

import logging

from recsys.tmdb import TMDBClient, _empty_features

logger = logging.getLogger(__name__)


def enrich(
    movie_ids: list[int],
    tmdb_map: dict[int, int],
    client: TMDBClient,
) -> dict[int, dict]:
    """Fetch TMDB content features for every movie.

    Args:
        movie_ids: List of MovieLens ``movieId`` values.
        tmdb_map: ``{movieId: tmdbId}`` mapping (from ``RatingsData.tmdb_map``).
        client: A ``TMDBClient`` (may be cache-only for fixture mode).

    Returns:
        ``{movieId: {genres, keywords, cast, crew, overview}}``.
        Movies with no ``tmdbId`` or fetch failure get empty features (FAIL-5).
    """
    features: dict[int, dict] = {}
    n_missing = 0
    n_fetched = 0

    for movie_id in movie_ids:
        tmdb_id = tmdb_map.get(movie_id)
        if tmdb_id is None:
            features[movie_id] = _empty_features()
            n_missing += 1
            continue

        result = client.get_movie(tmdb_id)
        if result is None:
            features[movie_id] = _empty_features()
            n_missing += 1
        else:
            features[movie_id] = result
            n_fetched += 1

    logger.info(
        "Enrichment complete: %d fetched, %d missing/empty (out of %d)",
        n_fetched, n_missing, len(movie_ids),
    )
    return features
