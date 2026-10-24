# Movie Recommendation System

**Which recommendation algorithm actually wins, and by how much?**

This is an experiment, not a product. It implements four classic
movie-recommendation approaches, runs all four over the same dataset with the
same train/test split, and prints one comparison table. The deliverable is a
**reproducible benchmark** you can re-run in a single command.

```bash
python -m recsys.cli run --fixture --out results/   # seconds, no API key, no download
```

## The four approaches

| Approach | What signal it uses | What it's good at |
|----------|--------------------|-------------------|
| **Popularity** | None — global top movies | The floor. Any personalized model must beat this to justify itself. |
| **Content-based** | TF-IDF + cosine over TMDB genres, keywords, cast, crew, overview | New/obscure titles; works with zero ratings for a movie |
| **Matrix factorization (SVD)** | Collaborative filtering on MovieLens ratings | Latent taste patterns across users |
| **Hybrid** | Weighted blend of content + CF | Cold-start coverage plus CF accuracy |

## How they're scored

Ranking quality is the primary axis — **Precision@K, Recall@K, NDCG@K** on a
per-user hold-out split, since what matters is the order of the top-N list.
**RMSE / MAE** are reported additionally for the two approaches that predict a
rating value (SVD, hybrid); popularity and content-based don't predict ratings,
so those cells are blank by design.

## Data

MovieLens `ml-latest-small` ratings, joined to TMDB metadata through
`links.csv` (`movieId → tmdbId`). TMDB metadata is pulled from the TMDB REST
API once and cached to disk, so repeat runs are offline.

## Setup

Python 3.11–3.13. A ready-to-use 3.13 virtualenv is checked in at `.venv/`:

```bash
source .venv/bin/activate         # or just prefix commands with .venv/bin/
pip install -r requirements.txt   # only if you're not using the bundled .venv
```

## Run it

### Offline (no API key, no download)

A tiny bundled fixture — 10 movies, 10 users — exercises the whole pipeline:

```bash
python -m recsys.cli run --fixture --out results/
```

It trains all four approaches and writes `results/comparison.md` and
`results/comparison.csv`:

| Approach | Precision@10 | Recall@10 | NDCG@10 | RMSE | MAE |
|----------|-------------|----------|---------|------|-----|
| popularity | 0.0900 | 0.8000 | 0.5061 | — | — |
| content | 0.0900 | 0.8000 | 0.7631 | — | — |
| svd | 0.0900 | 0.8000 | 0.5131 | 1.0957 | 0.8672 |
| hybrid | 0.0900 | 0.8000 | 0.7262 | 1.0957 | 0.8672 |

> These numbers are a smoke test, not a result — with 10 users every approach
> retrieves nearly the same set, so only NDCG separates them. Use the full
> dataset for a real comparison.

### Full dataset

1. Download `ml-latest-small` from
   <https://grouplens.org/datasets/movielens/> and unzip into
   `data/ml-latest-small/`.
2. Get a free TMDB API key, then `cp .env.example .env` and set `TMDB_API_KEY`.
3. Run it:

   ```bash
   python -m recsys.cli run --out results/
   ```

### Get actual recommendations

```bash
python -m recsys.cli recommend --user 1 -n 10     # top 10 for a user
python -m recsys.cli recommend --movie 1 -n 10    # movies similar to a movie
```

## Stack

Python · scikit-learn · scipy · pandas · numpy. No service, no database — the
CLI is the whole interface.

## Tests

```bash
pytest -q     # 393 tests, offline, fixture-based (~3 min)
```

> One known failure: `test_deterministic_rebuild_matches_committed_fixture` —
> the committed catalog fixture has drifted from a fresh deterministic rebuild.
> The other 392 pass.

## Project docs

| Doc | What's in it |
|-----|--------------|
| [`CONTEXT.md`](CONTEXT.md) | Glossary — the vocabulary used across the code |
| [`docs/SPEC.md`](docs/SPEC.md) | What was being built and why ([issue #1](https://github.com/VrajGupta/Movie-Recomendation-System/issues/1)) |
| [`docs/INVARIANTS.md`](docs/INVARIANTS.md) | Properties the benchmark must never violate |
| [`tickets.md`](tickets.md) | Work breakdown (GitHub issues #2–#7) |
| Handoffs | [planning](docs/handoffs/2026-07-10-planning.md) · [implementation](docs/handoffs/2026-07-10-implementation.md) · [review](docs/handoffs/2026-07-10-review.md) |

**Status:** all four approaches implemented and code-reviewed (a key-leak bug, a
test-gate bug, and an empty-split bug were found and fixed). Ready for a
full-dataset run.
