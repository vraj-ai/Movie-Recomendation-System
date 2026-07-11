# SPEC — Taste Engine + Web UI (effort 2)

> Glossary: `CONTEXT.md` (effort-2 section) · Non-functional constraints:
> `docs/INVARIANTS.md` (effort-2 section)
> Tracker: GitHub `VrajGupta/Movie-Recomendation-System` · Triage label: `ready-for-agent`
> Predecessor: `docs/SPEC.md` (the four-model algorithm experiment, built and reviewed)

## Problem Statement

The existing experiment scores anonymous MovieLens users with fitted ML models.
It cannot answer the question I actually have: **"what should *I* watch tonight,
and why, exactly?"** I don't want an AI that decides for me — I want a math
engine that looks at *my* taste through numbers, crunches them with logic I can
read, and finds the best movie. Every recommendation must come with its
arithmetic: which of my ratings produced which weights, and which terms produced
the score. And I want to drive it from a real web page, not a CLI.

## Solution

1. **Taste engine** (`recsys/taste.py`) — a pure, deterministic module. The
   operator's 0.5–5★ **taste ratings** produce a **taste profile** (per-genre
   weights, era preference, popularity preference) via closed-form arithmetic.
   Candidates are scored as a documented weighted sum of **feature match** +
   **neighbor evidence**, each response carrying a complete **score breakdown**
   (MATH-1..4). No fitted/ML models anywhere in the path.
2. **Profile store** (`recsys/web/profile_store.py`) — ratings + pins persisted
   in one human-readable, gitignored JSON file (WEB-FAIL-3, WEB-SEC-3).
3. **Local web server** (`recsys/web/app.py`) — FastAPI + uvicorn on
   127.0.0.1, explicit data mode at startup with fail-fast (WEB-FAIL-1,
   WEB-SEC-1), serving a single self-contained static page (no node build).
4. **Web UI** — four tabs: **Recommend** (top-N with breakdowns), **Similar**
   (movie-to-movie with similarity terms), **Profile** (ratings manager,
   derived weights, pin sliders, live re-rank), **Benchmark** (read-only render
   of the legacy CLI `results/comparison.csv`). Posters via public TMDB CDN
   paths with text-only fallback (WEB-FAIL-4); always-visible dataset badge.

The four ML recommenders are **replaced in the serving path** (MATH-3): the web
UI's recommendations come only from the taste engine. The ML four remain
CLI-only, and their comparison table stays visible in the Benchmark tab.

## The math (documented constants, all in `recsys/config.py`)

- **Movie features**: genre set `G(m)` (TMDB/MovieLens genres), decade `d(m)`
  (from title year), popularity percentile `p(m)` (rank of the movie's rating
  count within the loaded dataset).
- **Derived profile** from taste ratings `(m_i, r_i)` with signed signal
  `s_i = (r_i − 3.0) / 2.0 ∈ [−1.25, +1]`:
  - genre weight `w_g` = sum of `s_i` over rated movies containing genre `g`,
    normalized to `[−1, 1]` across genres;
  - era preference = `s`-weighted mean decade + tolerance (weighted spread);
  - popularity preference = `s`-weighted mean of `p(m_i)`.
- **Effective weights** apply MATH-4: pinned slider value if pinned, else the
  derived value.
- **Feature match** `F(m) = 0.5·genre_affinity + 0.25·era_match + 0.25·pop_match`,
  each term in `[−1, 1]`.
- **Similarity** (also powers the Similar tab)
  `sim(a, b) = 0.6·Jaccard(G(a), G(b)) + 0.25·era_proximity + 0.15·pop_proximity`.
- **Neighbor evidence** `N(m)` = normalized `Σ sim(m, m_i)·s_i` over the
  operator's rated movies.
- **Score** `S(m) = 100·(0.65·F(m) + 0.35·N(m))`; the breakdown lists every
  term above with its inputs (MATH-2). Rated movies are excluded from picks.

## User Stories

1. As the operator, I can search any movie in the dataset and rate it 0.5–5★,
   so the engine learns my taste from numbers I typed myself.
2. As the operator, I can read my derived weights as plain numbers next to
   their sliders, so nothing about my profile is hidden.
3. As the operator, I can drag a slider to pin a dimension, see it marked as
   pinned, and reset it per-slider back to the derived value, so I always know
   whether a weight came from my ratings or my hand.
4. As the operator, I get a top-N list where every movie shows its full
   arithmetic (feature-match terms + neighbor evidence terms summing to the
   score), so I can audit any pick down to the numbers.
5. As the operator, each pick cites specific movies I rated ("0.81 similar to
   Inception, rated 5★"), so explanations reference my actual taste.
6. As the operator, I can ask for movies similar to any title and see the
   similarity terms (genre overlap, era, popularity), so Similar is as
   auditable as Recommend.
7. As the operator, I always see which dataset and TMDB mode is live (badge),
   so I never misattribute results (WEB-FAIL-1).
8. As the operator, if I start the server in a mode whose prerequisites are
   missing, it fails immediately with instructions, so I never get a silently
   swapped dataset.
9. As the operator, I see posters when art exists and clean text cards when it
   doesn't, so fixture mode is fully usable offline.
10. As the operator, I can open the Benchmark tab and see the legacy ML
    comparison table, so the old experiment stays one click away.
11. As the operator, my ratings and pins live in one readable JSON file that
    never enters git, so my taste data is inspectable and private.
12. As the operator, with zero ratings the engine still works from neutral or
    pinned weights and says so, so the cold start is honest (WEB-FAIL-5).

## Implementation Decisions

- **Stack**: FastAPI + uvicorn; pydantic request/response models validate at
  the boundary (WEB-FAIL-2). Frontend is one static `index.html` + vanilla
  JS/CSS served by FastAPI — no node toolchain. New deps: `fastapi`, `uvicorn`,
  `httpx` (tests only).
- **Layout**: `recsys/taste.py` (pure engine, no I/O) · `recsys/web/app.py`
  (server + endpoints) · `recsys/web/profile_store.py` (persistence) ·
  `recsys/web/static/index.html` (UI). Engine reuses `load_ratings` and the
  cached-TMDB `enrich` path for genres; cache-only in the serving path.
- **Endpoints**: `GET /` (page) · `GET /api/status` (badge: mode, counts, TMDB
  status, profile stats) · `GET /api/search?q=` · `GET /api/recommend?n=` ·
  `GET /api/similar?movie_id=&n=` · `GET/PUT/DELETE /api/ratings` ·
  `GET /api/profile` + `PUT /api/profile/pins` (pin/unpin/reset) ·
  `GET /api/benchmark`.
- **Startup**: `python -m recsys.web --fixture` or real mode (requires
  `data/` present; TMDB cache for features). No auto-fallback (WEB-FAIL-1).
  Data loads once; profile recomputes on every ratings/pin mutation
  (WEB-PERF-3) — no training, so no model lifecycle.
- **Posters**: TMDB poster paths resolved server-side from the cache; the
  browser receives `image.tmdb.org` CDN URLs only (WEB-SEC-2).

## Testing Decisions

- **Engine**: hand-computed fixtures — a tiny rating set with known derived
  weights and a known ranking; breakdown terms assert-recombine to the score
  (MATH-2); determinism test calls twice and asserts identical output (MATH-1);
  pin-semantics tests (MATH-4). No network, no sklearn imports (MATH-3 enforced
  by a test that inspects loaded modules).
- **Store**: round-trip, corrupt-file quarantine (WEB-FAIL-3), empty-profile
  cold start (WEB-FAIL-5).
- **Server**: `httpx` TestClient — validation 4xx paths (WEB-FAIL-2), status
  badge truthfulness, fail-fast startup (WEB-FAIL-1), no key in any response
  (WEB-SEC-2), benchmark empty state (WEB-FAIL-6).
- **Frontend**: kept logic-thin (render only); contract-tested via the API
  plus a served-page smoke test (page loads, references only existing
  endpoints).

## Out of Scope

- Multi-user support, accounts, authentication; the UI serves one operator.
- Hosting/deployment, HTTPS, reverse proxies — localhost only (WEB-SEC-1).
- Serving, retraining, or extending the ML recommenders from the web path
  (they stay CLI-only; the Benchmark tab is read-only).
- Live TMDB enrichment from the browser or the serving path (cache-only).
- Watch providers, trailers, streaming links, mobile apps.

## Further Notes

- The engine's constants (component weights, similarity weights) are
  deliberately visible in config and in every breakdown; tuning them is the
  operator's prerogative, not a training loop's.
- If the operator later wants the taste engine benchmarked against the ML four,
  it can be wrapped in the `Recommender` seam as a follow-up effort — explicitly
  deferred, not precluded.
