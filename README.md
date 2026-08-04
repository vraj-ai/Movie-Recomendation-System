# Movie Recommendation System

An **algorithm experiment**: implement four movie-recommendation approaches and
compare them head-to-head on one dataset with one set of metrics. Not a production
service — the deliverable is a reproducible benchmark.

## Approaches compared

| Approach | Signal |
|----------|--------|
| Popularity baseline | Global (non-personalized) reference floor |
| Content-based | TF-IDF + cosine over TMDB features (genres, keywords, cast, crew, overview) |
| Matrix factorization (SVD) | Collaborative filtering on MovieLens ratings |
| Hybrid | Weighted blend of content + CF |

## Data

MovieLens `ml-latest-small` ratings joined to TMDB metadata via `links.csv`
(`movieId → tmdbId`). TMDB metadata is fetched from the TMDB REST API and cached
to disk.

## Evaluation

Top-N ranking metrics (Precision@K, Recall@K, NDCG@K) as the primary axis across
all four approaches, plus RMSE/MAE for the rating-predicting models, on a per-user
hold-out split.

## Stack

Python · scikit-learn · scipy · pandas · numpy. CLI: `run` (full pipeline →
comparison table) and `recommend` (top-N for a user / similar-to-movie).

## Setup

Requires Python 3.11–3.13 (a ready-to-use Python 3.13 virtualenv is checked in at
`.venv/`). Either activate it, or call `.venv/bin/python` directly:

```bash
source .venv/bin/activate            # or prefix commands with .venv/bin/
pip install -r requirements.txt      # only if not using the bundled .venv
```

## Quickstart (offline — no API key or download needed)

A small bundled fixture (10 movies, 10 users) runs the whole pipeline offline:

```bash
python -m recsys.cli run --fixture --out results/
```

This trains all four approaches and writes `results/comparison.{md,csv}`, e.g.:

| Approach | Precision@10 | Recall@10 | NDCG@10 | RMSE | MAE |
|----------|-------------|----------|---------|------|-----|
| popularity | 0.0900 | 0.8000 | 0.5061 | — | — |
| content | 0.0900 | 0.8000 | 0.7631 | — | — |
| svd | 0.0900 | 0.8000 | 0.5131 | 1.0957 | 0.8672 |
| hybrid | 0.0900 | 0.8000 | 0.7262 | 1.0957 | 0.8672 |

> Fixture numbers are illustrative (tiny dataset) — use the full dataset below for
> the real benchmark.

## Real run (full dataset)

1. Download `ml-latest-small` from <https://grouplens.org/datasets/movielens/> and
   unzip to `data/ml-latest-small/`.
2. Get a free TMDB API key: `cp .env.example .env`, then set `TMDB_API_KEY`.
3. `python -m recsys.cli run --out results/`

## Recommend / find similar

```bash
python -m recsys.cli recommend --user 1 -n 10     # top-10 for a user
python -m recsys.cli recommend --movie 1 -n 10    # movies similar to a movie
```

## Tests

```bash
pytest -q      # 147 passing (offline, fixture-based)
```

## Docs

- **Glossary:** [`CONTEXT.md`](CONTEXT.md)
- **Spec:** [`docs/SPEC.md`](docs/SPEC.md) ([issue #1](https://github.com/VrajGupta/Movie-Recomendation-System/issues/1))
- **Invariants:** [`docs/INVARIANTS.md`](docs/INVARIANTS.md)
- **Tickets:** [`tickets.md`](tickets.md) (GitHub issues #2–#7)
- **Handoffs:** [planning](docs/handoffs/2026-07-10-planning.md) · [implementation](docs/handoffs/2026-07-10-implementation.md) · [review](docs/handoffs/2026-07-10-review.md)

> Status: **implementation + code review complete** — all four approaches built,
> **147 tests passing**, reviewed via `/debugger` (SEC-1 key-leak, test-gate, and
> empty-split bugs fixed). Ready for a real-data run.
