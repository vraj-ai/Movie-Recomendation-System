# Handoff — Movie Recommendation System (all 6 tickets implemented)

**Date:** 2026-07-10
**Session:** `/part2` implementation chain — all tickets (#7 → #2 → #4 → #6 → #3 → #5)
**Status:** All 6 implementation tickets complete. 145 tests passing. Red-team pass done.
**Previous handoff:** [`2026-07-10-planning.md`](2026-07-10-planning.md)

## What was built

Implemented the entire Movie Recommendation System — all four recommender
approaches, the full data/evaluation/CLI pipeline, and the TMDB network seam —
across all six tickets in dependency order.

### Ticket summary

| Issue | Slice | Status | Tests |
|-------|-------|--------|-------|
| #7 | Walking skeleton + popularity baseline | Done | 53 |
| #2 | TMDB client + enrichment | Done | 24 |
| #4 | SVD recommender + RMSE/MAE | Done | 25 |
| #6 | `recommend` CLI command | Done | 11 |
| #3 | Content-based recommender | Done | 14 |
| #5 | Hybrid recommender (4-way table) | Done | 13 |
| — | Red-team corner cases | Done | 5 |
| | **Total** | | **145** |

All six verification commands exit 0 (see "Verification" section below).

### What each ticket delivered

- **#7 — Walking skeleton + popularity baseline:** `recsys` package scaffolded;
  data module (`data.py`) loads MovieLens ratings + links → sparse user-item
  matrix + `movieId→tmdbId` map, fails fast on missing dataset (FAIL-6);
  per-user hold-out splitter (`split.py`); `Recommender` ABC (`interfaces.py`);
  `PopularityRecommender` (Bayesian-weighted average); `evaluate` with
  Precision@K, Recall@K, NDCG@K; CLI `run` writes `results/comparison.{md,csv}`;
  bundled 10-movie fixture dataset enables offline `run --fixture`.
- **#2 — TMDB client + enrichment:** `TMDBClient` (`tmdb.py`) — sole network
  boundary; reads key from env only (SEC-1); fail fast on missing key (FAIL-1);
  exponential backoff + skip on 5xx/timeout (FAIL-2); honors 429 Retry-After
  (FAIL-3); coerces malformed fields to empty (FAIL-4); 404 → skip (FAIL-5);
  disk cache with no-network-on-hit (SEC-2); corrupt-cache detection + re-fetch
  (FAIL-7). Enrichment (`enrichment.py`) maps movies → features, tolerating
  misses. All tests run offline via mocked HTTP layer.
- **#4 — SVD recommender + RMSE/MAE:** `SVDRecommender` using `TruncatedSVD` on
  mean-centered ratings matrix; predicts unseen ratings; `evaluate` extended
  with `rmse()` and `mae()` functions tested against hand-computed values;
  SVD row added to comparison table with RMSE/MAE.
- **#6 — `recommend` CLI command:** `recommend --user <id> -n N` and
  `recommend --movie <id> -n N`; returns movie titles + IDs; unknown user/movie
  → clear error, exit 1; auto-resolves data source (real data → fixture
  fallback).
- **#3 — Content-based recommender:** `ContentRecommender` — TF-IDF over
  concatenated TMDB features (genres, keywords, cast, crew, overview) + cosine
  similarity; user profile = centroid of liked movies' TF-IDF vectors;
  `similar()` for movie-to-movie similarity; handles empty features gracefully
  (FAIL-5 — degrades to arbitrary-order unseen movies when vocabulary is empty).
- **#5 — Hybrid recommender:** `HybridRecommender` — weighted blend of
  min-max-normalized content + SVD scores; configurable `content_weight`
  (0=pure CF, 1=pure content); `predict()` delegates to SVD; completes the
  4-way comparison table (popularity · content · SVD · hybrid).

### Verification commands (all exit 0)

```
# #7
pytest tests/test_data.py tests/test_split.py tests/test_evaluate.py tests/test_popularity.py -q && python -m recsys.cli run --fixture --out results/

# #2
pytest tests/test_tmdb_client.py tests/test_enrichment.py -q

# #3
pytest tests/test_content.py -q && python -m recsys.cli run --fixture

# #4
pytest tests/test_svd.py tests/test_evaluate_rmse.py -q

# #5
pytest tests/test_hybrid.py -q && python -m recsys.cli run --fixture

# #6
pytest tests/test_cli_recommend.py -q && python -m recsys.cli recommend --user 1 -n 10
```

## Red-team pass findings

A fresh-eyes checker sub-agent (maker ≠ checker) attacked the corners. It
found and fixed **4 real bugs** test-first (red → green):

1. **Stale hybrid cache crash** — `HybridRecommender.fit` didn't clear
   `_content_scores_cache`; refitting on a differently-shaped train set caused
   a shape-mismatch broadcast crash. Fix: `self._content_scores_cache.clear()`
   in `fit()`. (`hybrid.py:54`)
2. **FAIL-1 violation on 401** — a rejected API key (HTTP 401) silently fell
   into the "other 4xx → skip" branch, producing empty features for every
   movie instead of failing fast. Fix: raise `TMDBError` on 401; CLI catches
   it and prints an actionable message. (`tmdb.py`, `cli.py`)
3. **Silent dataset swap** — `_auto_resolve_data_and_cache` returned the
   fixture data directory (10 movies) when real data existed but no API key
   was set, silently discarding the user's real ratings. Fix: return real data
   dir + fixture cache (for content only). (`cli.py:87`)
4. **Empty-split crash** — `holdout_split` on an empty DataFrame produced an
   opaque `pd.concat` error. Fix: explicit `ValueError` with clear message.
   (`split.py:34`)

### Invariant verification (all hold)

| Invariant | Holds | Location |
|-----------|-------|----------|
| FAIL-1 (missing/invalid key) | Yes | `cli.py:67-74` (missing), `tmdb.py` 401→raise (invalid) |
| FAIL-2 (transient error) | Yes | `tmdb.py:200-230` backoff+skip |
| FAIL-3 (rate limit 429) | Yes | `tmdb.py:186-198` Retry-After |
| FAIL-4 (malformed data) | Yes | `tmdb.py:_extract_features` try/except per field |
| FAIL-5 (no TMDB match) | Yes | `enrichment.py:36-44`, `content.py` empty-vocab guard |
| FAIL-6 (dataset download) | Yes | `data.py:79-85,107-113` |
| FAIL-7 (cache integrity) | Yes | `tmdb.py:127-145` |
| SEC-1 (key source) | Yes | env only; never in cache/logs/features |
| SEC-2 (data not committed) | Yes | `.gitignore` lists `data/`, `cache/` |

### Refactor smells flagged (not applied — design changes)

1. **Duplicated RatingsData-rebuild** — `cli.py` and `conftest.py` both rebuild
   sparse matrix from a train DataFrame. Recommend `RatingsData.from_split()`
   factory.
2. **Feature envy in HybridRecommender** — reaches into sub-recommenders'
   private attrs (`_tfidf_matrix`, `_user_profile`, `_reconstructed`). Recommend
   adding public accessors.
3. **Two near-identical data/cache resolvers** in `cli.py` —
   `_resolve_data_and_cache` vs `_auto_resolve_data_and_cache`.
4. **God function** `cmd_run` does load→split→rebuild→enrich→train→eval→write.
5. **Unused `k` param** in `_train_all`.
6. **Repeated try/except scaffold** in `_extract_features` (5x same pattern).
7. **Primitive obsession** — 5-field features dict re-built/validated by string
   keys across 3 modules. Recommend `ContentFeatures` dataclass.

### Unfixed edge cases (out of scope — honest)

- **Duplicate `(userId, movieId)` rows** are summed in the sparse matrix
  (CSR semantics). MovieLens `ml-latest-small` has no duplicates; dedup policy
  is a spec decision. Flag for `/part1`.
- **Cache value-type validation** — FAIL-7 checks keys exist but not that
  values are the right type. Downstream coercion prevents crashes/wrong data;
  strengthening is defense-in-depth.
- **Unbounded `Retry-After`** on 429 — honored as-is per FAIL-3 spec. Bounded
  only by `max_retries`.
- **Zero-relevant-items users** included in per-user metric averages —
  documented evaluation-policy choice, not a bug.

## Environment notes for the next agent

- **Python 3.13 venv** at `.venv/` (Python 3.14 system lacks scikit-learn
  wheels). Use `.venv/bin/python` for all commands.
- **Fixture dataset** (10 movies, 10 users, 66 ratings) under `fixtures/ml-small/`
  with pre-seeded TMDB cache under `fixtures/tmdb_cache/`. All tests run offline.
- **Real dataset:** download `ml-latest-small` from
  https://grouplens.org/datasets/movielens/, unzip to `data/ml-latest-small/`.
  Set `TMDB_API_KEY` in `.env` for real enrichment.
- **No linter configured** — only `py_compile` for syntax. Consider adding
  ruff/flake8.

## Suggested next steps

- **Real-data run** — download MovieLens + set TMDB key, then
  `python -m recsys.cli run` to produce the real comparison table.
- **Refactor pass** — address the 7 flagged smells (especially #1 factory and
  #2 accessors) via `/part1` → `/part2` or a dedicated refactor ticket.
- **Performance verification** — the perf invariants (PERF-1..6) are structural
  but not unit-tested; run on real data to confirm budgets hold.
