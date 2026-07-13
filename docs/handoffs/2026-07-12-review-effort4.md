# Handoff — /part3 review of effort-4 ratings + unique explanations (W12–W14)

## Scope reviewed
- W12 (e88968d): external ratings — catalog schema (imdb_rating, rt_rating, age_rating), OMDb enrichment, pinned fixture ratings.
- W13 (30071ab): /api/next payloads — `explanation` (unique varied sentence), `match10`, external ratings fields.
- W14 (b0ab330): web UI — ratingsRow (IMDb / 🍅 / age / Match ×/10, missing → "—") + explanationLine (chip fallback) in hero + rec cards.

## Invariants exercised
- RAT-1: never invent ratings; missing → "—" / NULL, never a broken card.
- RAT-2: no live network in tests — cache/fixture only.
- RAT-3: unique, deterministic explanations; no two adjacent cards identical.
- WEB-SEC: no innerHTML with API-derived strings.
- Determinism: byte-reproducible catalog builds.

## Test state at review start
- Full suite: 388 passed, 1 warning (StarletteDeprecationWarning).

## Findings (checker audit) — all fixed
- E4-1 (practical.py, RAT-3): attribute-coincident movies (same genre/decade/language, ids congruent mod template count) collided to identical explanation text. Fixed: salt rotation now keyed per-card with a deterministic uniqueness pass; regression test uses ids 100/103/106/109.
- E4-2 (enrich_ratings.py): OMDb URL was plain http — switched to https.
- E4-3 (enrich_ratings.py): a corrupt JSON cache entry poisoned every future run; now quarantined (removed) so a later online run can re-fetch; counted as `missing` offline.
- E4-4 (app.py): external-ratings fields serialized inconsistently for movies missing enrichment — normalized to explicit nulls per RAT-1.
- E4-5 (practical.py): `match10` used banker's rounding — 0.85 → 8. Fixed to half-up; floor 1, cap 10 (never 0/10).
- E4-6 (index.html): ratings row rendered `NaN` for non-numeric payload values — added `Number.isFinite` guard, falls back to "—".

## Verification
- New regression file `tests/test_e4_review_fixes.py` (5 tests) covers E4-1/2/3/5/6.
- Full suite after fixes: 393 passed, 1 warning (StarletteDeprecationWarning).

## Grade
B+ — implementation was solid and invariant-driven; findings were edge-case robustness (rounding, corrupt cache, template collision) rather than design flaws. No security issues beyond the http→https hardening.

## Next steps
- [done] Fix confirmed findings, re-run suite, commit review fixes.
- Push branch + open PR for effort-4.
- Then run /setup-vskills (task #20).
