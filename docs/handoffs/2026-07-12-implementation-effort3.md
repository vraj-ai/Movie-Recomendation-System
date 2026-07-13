# Handoff — Effort 3 implementation: practical watch-log recommender at India scale (2026-07-12)

## What shipped (W8–W10, all committed)

- **W8 — Catalog pipeline** (`28f9123`): `scripts/build_catalog.py` builds
  `data/catalog.sqlite` — **27,255 movies** = 9,742 MovieLens (exact, incl. the
  known "Confessions of a Dangerous Mind" duplicate, kept without breaking the
  invariant) + **17,534 Indian movies** (hi 5,627 / ta 3,168 / ml 3,121 /
  te 1,843 / bn 1,830 / kn 789 / mr 476 / pa 416 / gu 109 / as 106 / or 49).
  Offline-reproducible from cached downloads; FTS title search; poster
  coverage 87.7% with styled fallback. Small fixture:
  `fixtures/catalog-small.sqlite` (20 synthetic movies spanning
  genre/language/era/popularity tiers). 46 tests (`tests/test_catalog_build.py`).
- **W9 — Practical backend** (`87e3f76`): `recsys/recommenders/practical.py` —
  pure closed-form scorer (no fitted/ML model), lazily loaded only in `--full`
  mode so fixture/real serving stays import-light. Endpoints: `/api/status`,
  `/api/search`, `/api/watched` (5-star log, min 3 before recommending →
  `{"detail":"log_more","needed":N}`), `/api/next` with sliders
  `surprise/crowd/era/indian` (0–100, 422 outside). Every item returns
  `math.components` = `{genre, language, era, popularity, rating_signal}`
  (weights .35/.15/.20/.15/.15) + plain-English `reasons`; `indian` is a
  post-hoc multiplier (contract freezes 5 component terms). Surprise uses one
  explicit seeded random step — same profile + params ⇒ same list.
  Tests: `tests/test_practical_api.py`.
- **W10 — Dark cinematic UI** (`1f02c2e`): redesigned
  `recsys/web/static/index.html` — sliders, watch log, next-watch cards.
- **Suite: 349 passed, 0 failed** (`.venv/bin/pytest -q`, 20.5s).

## Bug found & fixed during integration

W9's agent stalled mid-run; the one red test exposed a real scoring bug:
`_era_component` used a hard-clipped linear ramp
`clip01(1 - |decade-target| / (2*tol))`, so anything ≥ 2·tol from the target
scored exactly 0.0 and the era slider had **no gradient** over old classics
(Casablanca 1942 → 0.0 at both extremes). Fixed to an exponential falloff
`exp(-|decade-target| / (2*tol))` — strictly monotonic, never flatlines,
reason threshold (`era > 0.6`) semantics preserved (≈1.0·tol vs 0.8·tol).

## Live smoke (--full, port 8977, verified then stopped)

- `/api/status` → `{"mode":"full","dataset":{"movies":27255},...}`
- 3 ratings logged (Kantara 25451, The Matrix 11, Baahubali 2) → gate opens.
- `era=0` → Saturday Night Fever 1977 (era .893), Moll Flanders 1996 (.713);
  `era=100` → Aadyaa 2020 (.893). Gradient works end-to-end.

## Notes for the /part3 reviewer

- `data/catalog.sqlite` (12 MB) is intentionally untracked; rebuild via
  `scripts/build_catalog.py` (downloads are cached).
- `/api/watched` body is `{"id": <catalog id>, "rating": 0.5–5}` — not
  `movie_id` (my first smoke call got a clean 422, which is the desired shape
  validation working).
- Known deprecation warning: starlette TestClient httpx notice (1 warning,
  pre-existing, not ours).
