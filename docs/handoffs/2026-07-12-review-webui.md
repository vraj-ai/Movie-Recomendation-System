# Handoff — /part3 review of the effort-2 web UI (2026-07-12)

**Scope:** the just-shipped taste engine + web UI (W1–W7): `recsys/taste.py`,
`recsys/profile_store.py`, `recsys/web/app.py`, `recsys/web/__main__.py`,
`recsys/web/static/index.html`, and their tests. Core recsys code was reviewed in the
previous /part3 run; its fixes (P3-1..3) were re-verified via the full suite.

**Agent:** `.claude/agents/part3-movie-recsys.md` — **reused** (not regenerated).
Minimal pin edit only: added the web UI files to the review scope and
`docs/SPEC-TASTE-WEBUI.md` + WEB-FAIL/WEB-SEC/WEB-PERF invariants to the doc list.

## Test state going in

`.venv/bin/pytest -q` → **245 passed, 0 red.** All bugs below were found by the
invariant-reading and test-quality nets, not by failing tests.

## Four-net audit results

- **Failing tests:** none.
- **Static (`py_compile` over `recsys/*.py recsys/recommenders/*.py recsys/web/*.py`):** clean.
- **Invariant violations:** 3 found, 3 fixed (below).
- **Weak/uncovered tests:** WEB-PERF-2/3 have no timing tests (deliberate — CI-flaky;
  recorded as follow-up), and the checker later flagged one of the maker's new UI
  tests as weak (advisory 1 below).

## Bugs fixed (test-first; gates green; recorded in `tickets.md`)

| ID | Invariant | Bug → fix | Verification-command |
|----|-----------|-----------|---------------------|
| P3-4 | WEB-FAIL-6 | `GET /api/benchmark` 500'd on non-UTF-8 or oversized-field `comparison.csv`; except-clause narrowed to `(OSError, UnicodeDecodeError, csv.Error)` → empty state | `.venv/bin/pytest tests/test_web_benchmark.py -q` (10 passed) |
| P3-5 | WEB-FAIL-3 | Quarantine warning never reached the operator: flag was consume-once (single `store.load()`) and `index.html` never rendered it. Fix: app-lifetime latch in `create_app` + `#quarantine` banner wired in `loadStatus()` | `.venv/bin/pytest tests/test_web_profile.py -q` (25 passed) |
| P3-6 | config-as-documented (WEB-SEC-3 anchor) | `python -m recsys.web` hardcoded port 8765/host, leaving `config.WEB_HOST`/`WEB_PORT` dead; parser defaults now come from config | `.venv/bin/pytest tests/test_web_app.py -q` (16 passed) |

**End state:** `.venv/bin/pytest -q` → **250 passed** (245 + 5 new red→green tests);
`py_compile` clean.

GitHub issues for P3-4..6 are **pending** — `gh auth status` timed out against the
keyring, so `tickets.md` is the audit trail; the three issues still need creating
with label `ready-for-agent`.

## Checker's grade (fresh-eyes, separate context): **CLEAN**

All three fixes verified real by reading the code and adversarial probing (fuzzed the
benchmark endpoint with missing/empty/UTF-16/quote-bomb/ragged/unreadable CSVs; checked
the quarantine latch for races, cross-instance leaks, and false positives; confirmed
WEB-SEC-1/2/3 untouched — localhost default preserved and independently pinned).
No BLOCKING findings. Six advisories:

1. `test_ui_wires_quarantine_warning` is tautological — `"quarantined" in html` is
   satisfied by an HTML comment alone; should assert `id="quarantine"` + JS wiring.
2. Python 3.13 `csv` doesn't raise on NUL bytes — binary junk that splits into
   consistent columns renders as a garbage table (`available: true`), not the empty
   state. Not a 500, so WEB-FAIL-6 technically holds.
3. Honoring `WEB_PORT` changes the effective default port 8765 → **8977**; any
   operator bookmark/script pinned to `:8765` breaks.
4. Quarantine banner hardcodes `taste_profile.json.corrupt-<timestamp>` (wrong for a
   custom `profile_path`) and says "server restarted" (it didn't).
5. Banner shows on init/mutations only (no polling) — a read-triggered quarantine
   surfaces on the next mutation or reload, not instantly. Latch guarantees eventual
   visibility.
6. Pre-existing: `ProfileStore._quarantine` can itself raise `OSError` → 500
   (unwritable dir / rename race); unchanged by this diff.

## Honest follow-ups (also in `tickets.md`)

1. WEB-PERF-2 at real scale: `recommend()` is O(n_movies × n_taste_ratings) pure
   Python; fine on the fixture, at ~9.7k movies the <1 s budget is at risk —
   needs profiling/vectorization (refactor territory).
2. No timing tests for WEB-PERF-2/3 (only WEB-PERF-1 startup is tested).
3. Similar-tab weight labels (`×0.6/×0.25/×0.15`) hardcoded in `index.html`,
   duplicating `TASTE_SIM_*` — drift risk; clean fix is a config endpoint.
4. Cold-start era/pop sliders render `0.00` for JSON `null` — cosmetic, JS-only.
5. `put_pin` accepts junk-but-harmless genre ids (range-validated only; inert).
6. Checker advisories 1–5 above (test strengthening, NUL-CSV coercion, port-change
   callout, banner wording, banner promptness).
7. Create the three pending GitHub issues once `gh` auth works.

## Next thing to review

The follow-up cluster above — especially strengthening the quarantine UI test
(advisory 1) and the WEB-PERF-2 real-data profile — or, if effort-3 starts, run
/part3 over whatever `/part2` ships next.
