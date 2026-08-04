# Handoff — effort 4 planning (ratings + unique explanations)

- Plan: `docs/plan/effort4-ratings-explanations.md`
- Tickets: W12 (#24) schema + OMDb enrichment · W13 (#25) API explanation/match10/ratings · W14 (#26) UI ratings row.
- Data source: OMDb (IMDb + RT + Rated in one payload), build-time only, cached, offline-replayable; NULLs render as “—”, never invented (RAT-1).
- match10 = round(clamp(score,0,1)*10,1) — monotone display metadata (RAT-2).
- `reasons` list → single deterministic `explanation` sentence, variant pools keyed by movie id (RAT-3).
- Frozen 5-component contract untouched; byte-reproducibility (E3-6) preserved via cache-only `--offline` enrichment.
- Next: /coder implements W12→W14 test-first on `plan/taste-engine-webui`.
