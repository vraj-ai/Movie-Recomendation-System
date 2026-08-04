# Handoff — Effort 2 implementation: Taste Engine + Web UI (2026-07-11)

## What shipped (branch `plan/taste-engine-webui`)

| Ticket | Commit | Summary |
| --- | --- | --- |
| W1 (#14) | `3a5b7c5` | Deterministic taste engine — profile, pins, scoring, breakdowns |
| W2 (#15) | `891a9f5` | JSON profile store — atomic save, corrupt-file quarantine |
| W3 (#16) | `6689469` | FastAPI server skeleton — explicit mode, status, search |
| W4 (#17) | `0f6c31b` | Recommend + similar APIs with full breakdowns |
| W5 (#18) | `bfe1ab1` | Ratings + profile APIs — mutations recompute fresh, pin semantics |
| W6 (#19) | `939bdd7` | Static page: Recommend / Similar / Profile tabs |
| W7 (#20) | `7550963` | Benchmark tab — read-only legacy comparison table + guard fix |

All eight GitHub tickets (#13 spec anchor + #14–#20) are closed with
commit references. Branch is pushed.

## Test state

- Full suite: **245 passed** (`.venv/bin/pytest -q`), 1 upstream
  starlette deprecation warning (not ours).
- Per-ticket verification commands are pinned in each ticket and in the
  test-file docstrings.

## Invariants held

- **MATH-3** — the web server never computes ML metrics. `/api/benchmark`
  reads `results/comparison.csv` verbatim; the guard test
  `test_no_ml_modules_loaded_by_benchmark` now runs in a **subprocess**
  so suite-order `sys.modules` pollution can't mask (or fake) a violation.
- **WEB-FAIL-6** — missing/malformed/ragged benchmark CSV answers a
  truthful empty state with an operator hint, never a 500.
- **WEB-SEC** — no `innerHTML` sinks in the page; all rendering goes
  through a `textContent`-only `el()` helper, so CSV/TMDB/user strings
  cannot inject HTML.

## Gotchas for the auditor (/debugger)

1. `create_app(...)` takes `pathlib.Path` for `profile_path` /
   `results_dir` — passing `str` raises `TypeError` at
   `recsys/web/app.py:328`. Consider whether coercing via `Path(...)`
   is worth it, or whether the typed contract should be documented.
2. The ML-isolation guard spawns a fresh interpreter (~1s); it is the
   only subprocess test in the suite.
3. `gh` API calls were flaky this session (i/o timeouts to
   api.github.com); retries succeeded. Ticket comments may lag.

## Next

`/debugger` — maker audit → fresh-eyes review → handoff → PR
(task #13, pending).
