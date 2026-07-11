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

---

# Effort 2 — Taste engine + web UI

Baseline assumptions: same dataset as above; the taste engine is closed-form
arithmetic (no training); the server is FastAPI + uvicorn on localhost.
Spec: `docs/SPEC-TASTE-WEBUI.md` · Glossary: `CONTEXT.md` (effort-2 section).

## Determinism / explainability (the point of the effort)

- **MATH-1 (determinism)** — identical inputs (taste ratings, pins, dataset)
  produce identical scores and identical ranking order on every call. No RNG, no
  fitted state, no wall-clock dependence anywhere in the serving path.
- **MATH-2 (auditable score)** — every recommendation and similarity response
  carries a complete score breakdown; the listed terms recombine to the total
  within 1e-9 relative tolerance, and the UI displays every term. No component
  of a score may exist that is absent from the breakdown.
- **MATH-3 (no ML in the serving path)** — the web server never imports, fits,
  or serves the sklearn recommenders. Scoring and similarity are closed-form
  (set overlap, proximity, weighted sums) with constants documented in config.
  The ML four remain CLI-only legacy.
- **MATH-4 (pin semantics)** — effective weight = pinned value if pinned, else
  ratings-derived value. Never a blend; per-dimension reset restores derived.

## Performance budgets

- **WEB-PERF-1 (startup)** — server ready (data loaded, profile computed) in
  **< 5 s** fixture mode; **< 30 s** real mode on a warm TMDB cache.
- **WEB-PERF-2 (request latency)** — every API endpoint responds in **< 1 s**
  after startup (parity with PERF-5).
- **WEB-PERF-3 (recompute)** — a rating or pin change yields an updated profile
  and re-ranked recommendations in **< 500 ms**.

## Failure modes

- **WEB-FAIL-1 (explicit data mode)** — the mode is chosen at startup (fixture
  flag or real data). Missing prerequisites (no dataset, no cache) **fail fast**
  with an actionable message. The server **never silently swaps datasets**.
- **WEB-FAIL-2 (input validation)** — unknown `movieId`, out-of-range rating,
  bad `n`, or malformed JSON returns a **4xx** with a clear error body; user
  input can never produce a 500.
- **WEB-FAIL-3 (profile integrity)** — a corrupt/unreadable profile file is
  quarantined to `<name>.corrupt-<timestamp>`, the server starts with an empty
  profile, and the UI shows a visible warning. Never crashes; never silently
  overwrites the operator's data.
- **WEB-FAIL-4 (posters degrade)** — missing TMDB art yields a text-only card:
  no broken images, and the browser never calls the TMDB API (public image CDN
  URLs only).
- **WEB-FAIL-5 (cold profile)** — zero taste ratings → neutral derived weights;
  the engine still serves (pins/sliders can drive everything) and the UI labels
  the profile as neutral.
- **WEB-FAIL-6 (benchmark absent)** — missing or malformed
  `results/comparison.csv` renders a friendly empty state naming the CLI command
  that produces it; never an error page.

## Security / permission boundaries

- **WEB-SEC-1 (bind)** — the server binds **127.0.0.1** by default; any wider
  bind requires an explicit flag.
- **WEB-SEC-2 (key stays server-side)** — the TMDB key never appears in HTML,
  JS, or any API response. Poster URLs sent to the browser are public CDN paths
  that require no key (SEC-1 still governs the key itself).
- **WEB-SEC-3 (profile privacy)** — the profile/ratings file is gitignored; it
  contains only movie ratings, pins, and derived numbers — no PII.
