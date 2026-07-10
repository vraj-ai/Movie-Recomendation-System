# INVARIANTS — Movie Recommendation System

Non-functional constraints, stated as **named, testable invariants**. Every spec
acceptance constraint and every ticket that touches one of these MUST restate the
relevant budget / failure-mode / boundary in its acceptance criteria.

Baseline assumptions for all budgets: dataset = MovieLens `ml-latest-small`
(~100k ratings, ~9.7k movies, ~610 users); commodity laptop; **warm** TMDB cache
unless stated otherwise.

## Performance budgets

- **PERF-1 (cold enrichment)** — a full cold TMDB enrichment of all ~9.7k movies
  completes in **< 15 min** while respecting the API rate limit. One-time; cached
  thereafter.
- **PERF-2 (warm data load)** — loading ratings + cached TMDB metadata into memory
  completes in **< 20 s**.
- **PERF-3 (per-model training)** — each individual approach trains in **< 60 s**.
- **PERF-4 (full experiment)** — the end-to-end `run` (load → train all 4 →
  evaluate → emit table) completes in **< 5 min** on a warm cache.
- **PERF-5 (recommend latency)** — a `recommend` top-N query returns in **< 1 s**
  after models are loaded.
- **PERF-6 (memory)** — peak resident memory stays **< 2 GB**.

## Failure modes

External dependency under consideration: the **TMDB REST API**. Also covered:
the MovieLens download and the local cache.

- **FAIL-1 (missing/invalid key)** — if the TMDB API key is absent or rejected,
  the program **fails fast at startup** with a clear, actionable message. It never
  begins a long fetch only to die partway.
- **FAIL-2 (transient API error)** — on timeout or 5xx, retry with exponential
  backoff up to a bounded number of attempts; if still failing, **skip that movie**
  with a logged warning and continue. The pipeline never crashes on one bad movie.
- **FAIL-3 (rate limit)** — on HTTP 429, honor `Retry-After` / back off before
  retrying; the client never hammers the API past its limit.
- **FAIL-4 (malformed response)** — missing or malformed fields are coerced to
  empty. The affected movie carries null content features and is still handled by
  the CF and popularity approaches.
- **FAIL-5 (no TMDB match)** — a movie with no `tmdbId` or no TMDB record degrades
  gracefully: content model tolerates empty features; CF/popularity are unaffected.
- **FAIL-6 (MovieLens download)** — if the MovieLens dataset cannot be obtained,
  fail fast with explicit manual-download instructions.
- **FAIL-7 (cache integrity)** — a partial or corrupt cache entry is detected and
  **only the affected entries are re-fetched**; a bad cache never yields silently
  wrong data.

## Security / permission boundaries

- **SEC-1 (key source)** — the TMDB API key is read **only** from an environment
  variable (or a gitignored `.env`). It is **never committed and never logged**
  (not in debug output, not in error messages).
- **SEC-2 (data not committed)** — `data/` (downloaded datasets) and the TMDB
  response `cache/` are **gitignored**. Neither raw datasets nor cached API
  responses enter the repository.
- **SEC-3 (blast radius)** — only public, anonymized MovieLens data and public
  TMDB metadata are handled; no PII. If the key leaks, the exposure is a
  free-tier, read-only TMDB key — revoke/rotate; no user data is at risk.
