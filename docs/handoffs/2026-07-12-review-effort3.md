# Effort 3 — /part3 Review: practical watch-log recommender (W8–W10)

Reviewer pass on branch `plan/taste-engine-webui`, mirroring the effort-2
review that produced `ae4ca2f`. Baseline at review start: 349 passed.
After review: **359 passed, 0 failed** (1 pre-existing starlette
deprecation warning), all with `.venv/bin/pytest`, no network, no
wall-clock dependence.

## Findings

| ID | Severity | Location | Finding | Verdict | Fix |
|----|----------|----------|---------|---------|-----|
| E3-1 | HIGH | `recsys/web/app.py` (legacy rating/pin mutation endpoints) | Legacy mutations called `store.save(ratings, pins)` without the `--full` watch log — any legacy-tab rating/pin/unpin while running `--full` silently wiped `profile.watched`. Data loss. | CONFIRMED | `8a50d69` |
| E3-2 | MED | `recsys/web/app.py` `/api/search` (`--full`) | Legacy modes cap `q` at `SEARCH_MAX_QUERY` (200 chars); `--full` did not — a megabyte of prefix tokens is a free CPU-burn (WEB-FAIL-2 drift). Now 400s past the cap. | CONFIRMED | `8a50d69` |
| E3-3 | HIGH | `recsys/web/app.py` `/api/watched` | Profile save is read-modify-write of a JSON file; FastAPI sync handlers run in a threadpool, so concurrent POSTs interleave and drop entries. Reproduced with 20 parallel POSTs. Fixed with a per-app mutation lock on all profile writes. | CONFIRMED | `8a50d69` |
| E3-4 | MED | `recsys/recommenders/practical.py` `_anchor_reason` | "Because you gave X 5 stars" could cite *any* watched movie sharing genre/language — including 1-star ones. Reasons now only anchor on positively-weighted (above-neutral) movies. UX honesty (PRAC-2). | CONFIRMED | `8a50d69` |
| E3-5 | LOW | `recsys/web/app.py` `/api/search` | Rows tied on `(vote_count, popularity)` had unspecified order → flaky pagination/ranking. Movie id added as final tie-break; deterministic. | CONFIRMED | `8a50d69` |
| E3-6 | MED | `scripts/build_catalog.py` (CAT-1) | `build_timestamp` was `datetime.now()` inside the sqlite artifact, so the "offline-reproducible" build was never byte-reproducible and rebuilding the committed fixture always dirtied the tree. Fixture now pins `FIXTURE_BUILD_TIMESTAMP`; full builds honor `SOURCE_DATE_EPOCH`. Fixture regenerated once; rebuilds are byte-identical. | CONFIRMED | `db17595` |

### Audited, not exploitable (PLAUSIBLE → cleared)

- **Path traversal in profile store** — profile path comes from CLI/config,
  not request data; no user-controlled path segments reach the filesystem.
- **XSS in UI** — titles/overviews are rendered via `textContent`/DOM
  building, not `innerHTML` string interpolation; script-in-title fixture
  renders inert.
- **SQL injection in FTS search** — user query goes through
  `_sanitize_fts_query` and is bound as a parameter; quote/operator
  payloads return results or empty, never errors.
- **Slider math edge cases** — sliders at 0/100 extremes, empty era,
  and neutral crowd all stay within contract bounds (MATH-3/MATH-4
  behavior confirmed by existing + new tests).
- **Contract drift** — `math.components` is exactly the frozen 5 terms;
  `indian` remains a post-hoc multiplier (E3-4 fix touches only reason
  text, not scoring).
- **Missing `data/catalog.sqlite` in `--full`** — clean startup error
  pointing at `CATALOG_BUILD_INSTRUCTIONS`; not a 500.
- **Min-3-log gate** — `{"detail":"log_more","needed":N}` is intended
  behavior per the plan; not flagged.

## Grade: **B**

- **Correctness: B+** — scorer math is clean and the in-effort era-falloff
  fix (hard clip → `exp` gradient) was the right call, but two HIGH
  data-loss bugs (E3-1, E3-3) sat in the profile write path, and search
  ordering was nondeterministic on ties.
- **Security: A−** — no traversal, no XSS, no SQLi; the only gap was the
  missing query-length cap in `--full` (cheap DoS), now closed.
- **Tests: B+** — 349 green tests with genuinely offline catalog coverage
  is strong, but the suite missed concurrency entirely and never rebuilt
  the fixture it claims is reproducible.
- **UX honesty of reasons: B−** — reason strings are grounded in real
  signals, but "Because you gave X 5 stars" citing a disliked movie is
  exactly the kind of trust-breaking output the plan warns about.
- **Reproducibility of catalog build: C → A after fix** — the
  wall-clock `build_timestamp` contradicted the effort's own headline
  claim; one-line class of bug, but it invalidated CAT-1 until E3-6.

Net: solid engineering with a well-frozen contract and honest gates,
dragged down by write-path races and a reproducibility claim that wasn't
tested against itself.

## Deferred

- Cross-*process* profile locking (the mutation lock is in-process only;
  two servers on one profile can still race). Out of scope — single-server
  is the documented deployment.
- FTS relevance quality for transliterated Indian titles (e.g. Hindi
  romanization variants) — behavior is correct, recall is mediocre;
  belongs in a future effort, not a review fix.
- `SOURCE_DATE_EPOCH` is honored but not threaded through a `--timestamp`
  CLI flag; env var is the reproducible-builds convention and suffices.

## Test count

- Before review: **349 passed**
- After review: **359 passed** (+10: search cap ×2, concurrency, watch-log
  preservation, anchor honesty ×3, deterministic ties, timestamp pinning /
  byte-identical rebuild in `tests/test_catalog_build.py`)
