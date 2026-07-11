# SPEC — Movie Recommendation System (algorithm experiment)

> Glossary: `CONTEXT.md` · Non-functional constraints: `docs/INVARIANTS.md`
> Tracker: GitHub `VrajGupta/Movie-Recomendation-System` · Triage label: `ready-for-agent`

## Problem Statement

I want to understand which movie-recommendation approach actually works best on
real data, and by how much. Reading about content-based vs. collaborative vs.
hybrid recommenders is not the same as seeing them scored head-to-head on the same
dataset with the same metrics. I need a reproducible experiment that implements
several approaches and produces a single, trustworthy comparison I can point to.

## Solution

A Python **algorithm experiment** that:

1. Loads MovieLens `ml-latest-small` ratings and enriches each movie with TMDB
   content features (genres, keywords, cast, crew, overview), joined via
   `links.csv` (`movieId → tmdbId`), with TMDB responses cached to disk.
2. Implements four recommenders behind one common interface: a **popularity
   baseline**, a **content-based** model, a **matrix-factorization (SVD)** model,
   and a **hybrid** of content + CF.
3. Evaluates all four on a per-user **hold-out split** using top-N **ranking
   metrics** (Precision@K, Recall@K, NDCG@K) as the primary axis, plus RMSE/MAE
   for the rating-predicting models.
4. Emits a **comparison table** (Markdown + CSV) ranking the approaches, runnable
   from a CLI (`run`), with a `recommend` command for ad-hoc top-N lists.

## User Stories

1. As an experimenter, I want to download/prepare the MovieLens `ml-latest-small`
   dataset with one command, so that setup is reproducible.
2. As an experimenter, I want movies joined to their TMDB IDs via `links.csv`, so
   that I can attach content features to ratings.
3. As an experimenter, I want TMDB content features fetched via the API and cached
   to disk, so that the first run enriches and later runs are offline and fast.
4. As an experimenter, I want the program to fail fast with a clear message if the
   TMDB API key is missing or invalid, so that I never wait through a doomed fetch.
5. As an experimenter, I want TMDB fetch failures (timeouts, 5xx, 429, malformed
   data) handled gracefully, so that one bad movie never crashes the run.
6. As an experimenter, I want movies with no TMDB match to still participate via
   CF/popularity, so that coverage gaps don't distort the comparison.
7. As an experimenter, I want a per-user hold-out split of ratings, so that models
   train and are scored on disjoint data.
8. As an experimenter, I want a popularity baseline, so that I have a floor every
   personalized approach must beat.
9. As an experimenter, I want a content-based recommender over TMDB features, so
   that I can measure the value of content signals.
10. As an experimenter, I want a matrix-factorization (SVD) recommender, so that I
    can measure a strong collaborative-filtering approach.
11. As an experimenter, I want a hybrid recommender blending content and CF, so
    that I can measure whether combining signals wins.
12. As an experimenter, I want every recommender to expose the same interface, so
    that evaluation treats them uniformly.
13. As an experimenter, I want Precision@K, Recall@K, and NDCG@K computed per user
    and averaged, so that I can rank all four approaches on one axis.
14. As an experimenter, I want RMSE/MAE reported for rating-predicting approaches,
    so that I also see rating-accuracy where it's meaningful.
15. As an experimenter, I want a single `run` command that executes the full
    pipeline and writes a comparison table (Markdown + CSV), so that results are
    one command away and reproducible.
16. As an experimenter, I want a `recommend` command that returns a top-N list for
    a given user (or movies similar to a given movie), so that I can sanity-check
    outputs qualitatively.
17. As an experimenter, I want the whole warm-cache experiment to finish in a few
    minutes, so that I can iterate quickly.
18. As an experimenter, I want my TMDB API key and downloaded data kept out of the
    repo, so that I never leak secrets or bloat the repository.

## Implementation Decisions

- **Language/stack**: Python with scikit-learn (TF-IDF, cosine similarity,
  `TruncatedSVD`/`NMF`), scipy sparse matrices, pandas, numpy. No heavyweight
  recsys library (avoids `scikit-surprise` install/maintenance friction).
- **Data module**: loads MovieLens ratings and `links.csv`; exposes ratings as a
  sparse user–item matrix and a `movieId → tmdbId` mapping.
- **`TMDBClient` (the network seam)**: the sole boundary to the TMDB REST API.
  Reads the API key from an env var only (SEC-1). Implements retry/backoff
  (FAIL-2), 429 handling (FAIL-3), malformed-response coercion (FAIL-4), and a
  disk **cache** so reruns are offline. Fully mockable for tests.
- **Enrichment**: maps each MovieLens movie to TMDB content features via `tmdbId`;
  movies with no match carry empty content features (FAIL-5).
- **`Recommender` interface (the model seam)**: a common contract —
  `fit(train)`, `recommend(user, n) -> ranked movie list`, and (where meaningful)
  `predict(user, item) -> rating`. Implemented by `PopularityRecommender`,
  `ContentRecommender`, `SVDRecommender`, and `HybridRecommender`.
- **Content model**: TF-IDF over concatenated TMDB text/categorical features +
  cosine similarity; a user profile is aggregated from their liked movies.
- **SVD model**: latent-factor factorization of the ratings matrix
  (`TruncatedSVD`/`NMF`) predicting unseen ratings, ranked for top-N.
- **Hybrid model**: a weighted blend of normalized content and CF scores (weight
  configurable), producing one ranked list.
- **Splitter**: per-user hold-out — for each user, a fraction of their ratings is
  held out for test; a relevance threshold defines a "hit".
- **`evaluate` (the metrics seam)**: computes Precision@K, Recall@K, NDCG@K per
  user and averages them; computes RMSE/MAE for rating-predictors. Takes any
  `Recommender` so all four are scored identically.
- **CLI (the top seam)**: `run` executes load → enrich → train all four →
  evaluate → write comparison table (Markdown + CSV under `results/`);
  `recommend` returns a top-N list for a user or a movie.
- **Config**: dataset variant, K, relevance threshold, hold-out fraction, hybrid
  weight, and paths are configurable (flags/env), with sensible defaults.

### Acceptance constraints (from `docs/INVARIANTS.md`)

- **Performance**: cold enrichment < 15 min (PERF-1); warm load < 20 s (PERF-2);
  each model trains < 60 s (PERF-3); full `run` < 5 min warm (PERF-4);
  `recommend` < 1 s (PERF-5); peak memory < 2 GB (PERF-6).
- **Failure modes**: fail fast on missing/invalid key (FAIL-1); backoff+skip on
  transient errors (FAIL-2); honor 429 (FAIL-3); coerce malformed data (FAIL-4);
  degrade on no-match (FAIL-5); fail fast on dataset-download failure (FAIL-6);
  detect+refetch corrupt cache entries (FAIL-7).
- **Security**: key only from env/gitignored `.env`, never committed/logged
  (SEC-1); `data/` and `cache/` gitignored (SEC-2); public/anonymized data only,
  revocable key (SEC-3).

## Testing Decisions

- **What makes a good test**: exercise external behavior at the seams, not
  implementation details. Tests must not hit the network — the `TMDBClient` is
  mocked/faked.
- **`TMDBClient`**: unit tests with a fake HTTP layer covering the success path
  and each failure mode (missing key → fail fast; 5xx → backoff then skip; 429 →
  honor Retry-After; malformed JSON → empty features; cache hit → no network).
- **Recommenders**: each of the four tested against a tiny fixture ratings/feature
  set with hand-verifiable expected top-N ordering; all satisfy the shared
  `Recommender` contract.
- **`evaluate`**: metric functions checked against hand-computed Precision@K,
  Recall@K, NDCG@K, and RMSE on small inputs with known answers.
- **Splitter**: verifies train/test disjointness per user and threshold behavior.
- **CLI**: end-to-end test runs `run` on a bundled fixture dataset (offline, cache
  pre-seeded) and asserts a well-formed comparison table with all four approaches
  is produced; `recommend` returns N items for a known user.

## Out of Scope

- Any production/serving concern: web UI, hosted API, authentication, databases,
  real-time updates, deployment, monitoring.
  *(Amended — effort 2: a **local** web UI over a deterministic taste engine is
  now specified in `docs/SPEC-TASTE-WEBUI.md`. Hosted/production serving remains
  out of scope.)*
- Real user accounts or non-anonymized data.
- Larger MovieLens variants (`1M`, `25M`) and scalability tuning.
- Item-KNN / user-KNN CF, deep-learning recommenders, and online/bandit methods
  (only the four agreed approaches are in scope).
- Hyperparameter search beyond a small, documented default configuration.

## Further Notes

- The TMDB API key is obtained free from a TMDB account and provided via env var.
- The comparison table is the deliverable artifact; its Markdown form can be
  pasted into a README or writeup.
- If future work wants scalability, the same seams (`Recommender`, `evaluate`,
  `TMDBClient`) extend to `ml-25m` without redesign.
