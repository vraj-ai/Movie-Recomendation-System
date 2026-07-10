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

## Setup (once code lands)

1. Get a free TMDB API key and set it: `cp .env.example .env` then add `TMDB_API_KEY`.
2. Install deps and run `python -m recsys.cli run` to produce the comparison table.

## Planning docs

- **Glossary:** [`CONTEXT.md`](CONTEXT.md)
- **Spec:** [`docs/SPEC.md`](docs/SPEC.md) ([issue #1](https://github.com/VrajGupta/Movie-Recomendation-System/issues/1))
- **Invariants:** [`docs/INVARIANTS.md`](docs/INVARIANTS.md)
- **Tickets:** [`tickets.md`](tickets.md) (GitHub issues #2–#7)

> Status: planning complete, implementation not started. Build order starts at
> [issue #7](https://github.com/VrajGupta/Movie-Recomendation-System/issues/7).
