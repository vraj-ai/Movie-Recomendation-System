# Handoff — Planning, Effort 2: Taste Engine + Web UI (2026-07-11)

## What this effort is

A deterministic **personal taste engine** (closed-form arithmetic over the
operator's own 0.5–5★ ratings, with pinnable weight sliders) served by a
**local FastAPI web UI** with four tabs: Recommend / Similar / Profile /
Benchmark. Every pick ships its full score breakdown. The four ML recommenders
from effort 1 are **replaced in the serving path** — they stay CLI-only, and
their comparison table renders read-only in the Benchmark tab.

## State at handoff

- Branch: `plan/taste-engine-webui` (planning docs only; no product code yet).
- Spec: `docs/SPEC-TASTE-WEBUI.md` (grilled with the operator; key rulings:
  no AI/LLM in the pick path, all math auditable, no auto data-mode fallback,
  fixture mode first-class, single static page / no node build).
- `docs/SPEC.md` Out of Scope amended with a pointer (web UI now in scope
  as a *local* app; hosted serving still out).
- Invariants: effort-2 section in `docs/INVARIANTS.md` — MATH-1..4,
  WEB-PERF-1..3, WEB-FAIL-1..6, WEB-SEC-1..3.
- Context: effort-2 section in `CONTEXT.md`.
- Tickets: GitHub #13 (spec anchor) + #14–#20 (W1–W7), label
  `ready-for-agent`, mirrored in `tickets.md`. Dependency shape:
  frontier {W1 #14, W2 #15, W3 #16} → {W4 #17, W5 #18} → W6 #19 → W7 #20.

## Decisions a builder must not re-litigate

1. **MATH-3**: no fitted/ML model code imports anywhere in the web serving
   path — enforced by a module-inspection test.
2. **MATH-4** pin semantics: pinned slider value wins outright; unpinned uses
   derived; never blended; per-slider reset returns to derived.
3. **WEB-FAIL-1**: data mode is explicit at startup (`--fixture` vs real) and
   fails fast when prerequisites are missing. Never silently swap datasets.
4. **WEB-SEC-1/2**: bind 127.0.0.1; the TMDB key never reaches the browser —
   posters are pre-resolved `image.tmdb.org` CDN URLs from the cache.
5. Profile JSON is human-readable and gitignored (WEB-SEC-3); corrupt files
   quarantine, never crash, never silently overwrite (WEB-FAIL-3).

## Known hazards for /coder

- `gh`/GitHub API was flaky during ticketing (dial timeouts). Use retry loops
  for issue/PR operations; verify with `gh issue list` before re-creating
  anything to avoid duplicates.
- Effort-1 leftovers deliberately unfixed (see `tickets.md` follow-ups):
  FAIL-3 unbounded Retry-After, FAIL-7 cache value-type validation, metric
  averaging over zero-relevant users, duplicate CSR rows. Don't get pulled
  into them unless a W-ticket touches the same lines.
- Tests import `recsys` via `tests/conftest.py` sys.path prepend (P3-1 fix);
  keep new tests compatible with the pinned gate `.venv/bin/pytest -q`.

## First moves for /coder

Start with the frontier in any order (W1 engine is the highest-leverage);
each ticket carries its own `Verification-command`. Work test-first; new deps
(`fastapi`, `uvicorn`, `httpx`) go in `requirements.txt` via W3.
