# Effort 4 — External ratings + unique per-movie explanations (W12–W14)

## Goals
1. Replace the repetitive templated `reasons` chips with **one distinct, movie-specific
   explanation sentence per card** — no two adjacent cards read the same.
2. Show on every movie card: **IMDb rating**, **Rotten Tomatoes rating**, **age rating**
   (certification), and an **overall match out of 10**.

## Non-goals
- No change to the frozen 5-component score contract (genre/language/era/popularity/
  rating_signal + post-hoc indian multiplier). External ratings are **display metadata
  only** — never score inputs.
- No live network calls at serve time or test time.

## Data-sourcing decision
- **Source: OMDb API** (`http://www.omdbapi.com/`) — the only single source carrying
  IMDb score, Rotten Tomatoes score, and `Rated` (age certification) together.
- Enrichment happens **offline at build time** via `scripts/enrich_ratings.py`:
  looks up by title+year, requires `OMDB_API_KEY`, caches every raw response to
  `data/cache/omdb/<movie_id>.json` (same pattern as the TMDB cache), and writes three
  new **nullable** columns onto `movies`: `imdb_rating REAL`, `rt_rating INTEGER`,
  `age_rating TEXT`.
- **Honesty invariant (RAT-1): missing data renders as “—”, never invented.** No key,
  no cache, no OMDb match → NULL → “—”.
- Reproducibility: enrichment reads only from the cache when `--offline` is set;
  rebuilds stay byte-identical under `SOURCE_DATE_EPOCH` (preserves E3-6/CAT-1).
- Fixture DB gets hand-written ratings for a handful of movies + deliberate NULLs so
  both render paths are tested offline.

## Match out of 10
- `match10 = round(min(max(score, 0), 1) * 10, 1)` computed server-side per pick.
- Monotone in the real score (RAT-2), stable within a session, display-only.

## Unique explanations
- New `build_explanation(...)` in `practical.py` returns **one sentence** replacing the
  chip list. Deterministic composition:
  - pick the **dominant weighted component** for that movie;
  - fill a movie-specific template citing real facts (anchor watch title + stars,
    shared genre, decade, language, `vote_average`);
  - phrasing pool per component (≥4 variants each), variant chosen by
    `movie.id % len(pool)` → adjacent cards differ, output is deterministic (RAT-3).
- API: `reasons` list replaced by `explanation: str` in `/api/next` items (UI updated
  in the same ticket; no external consumers).

## Failure modes
- **RAT-FAIL-1**: OMDb unreachable/keyless during enrichment → skip, leave NULLs, log;
  build never fails because of ratings.
- **RAT-FAIL-2**: malformed OMDb payload (`"N/A"`, missing fields) → coerce to NULL.
- **RAT-FAIL-3**: UI must not break when all three ratings are NULL (renders “—”).

## Tickets
- **W12** — schema + enrichment script + fixture ratings (`build_catalog.py`,
  `enrich_ratings.py`, migration of existing catalog, tests).
- **W13** — API: `explanation`, `match10`, `imdb_rating`, `rt_rating`, `age_rating`
  in payloads; `build_explanation` with determinism/variety tests.
- **W14** — UI: ratings row (IMDb / 🍅 / age badge / match ×/10) + explanation line on
  cards; tests.

## Test strategy
- Offline fixtures only; determinism test: same inputs → identical explanations;
  variety test: top-10 picks contain no duplicate explanation sentences;
  NULL-rendering test for “—”; monotonicity test for match10.
