# CONTEXT — Movie Recommendation System (glossary)

This is a **glossary**, not a spec. It defines the canonical language of the
project. No implementation details live here. See `docs/SPEC.md` for the
destination and `docs/INVARIANTS.md` for the non-functional constraints.

## Nature of the effort

This project is an **algorithm experiment**: it implements several movie
recommendation approaches and compares them on the same dataset using the same
metrics. It is *not* a production service. "Done" means a reproducible benchmark,
not an uptime guarantee.

## Core entities

- **User** — an anonymized MovieLens rater, identified by `userId`. No PII.
- **Movie** — a film, identified in MovieLens by `movieId` and linked to external
  catalogs by `imdbId` / `tmdbId` (via MovieLens `links.csv`).
- **Rating** — a single `(userId, movieId, rating, timestamp)` observation from
  MovieLens. The interaction signal used for collaborative filtering.
- **Content features** — attributes of a Movie sourced from TMDB (genres,
  keywords, cast, crew, overview). Used by the content-based and hybrid approaches.

## Approaches (the things being compared)

- **Popularity baseline** — a non-personalized recommender that ranks movies by a
  global signal (e.g. most-rated / highest mean rating). The reference floor every
  personalized approach must beat.
- **Content-based** — recommends movies whose **content features** are similar
  (cosine similarity over a TF-IDF / feature vector) to movies a user liked.
- **Matrix factorization (SVD)** — model-based **collaborative filtering** that
  factorizes the user–movie **Rating** matrix into latent factors to predict
  unseen ratings.
- **Hybrid** — combines content-based and collaborative-filtering signals into a
  single ranked list (weighted blend or feature-augmented).
- **Collaborative filtering (CF)** — any approach that recommends from
  user–movie **Rating** patterns rather than movie content. Here, realized as SVD.

## Evaluation vocabulary

- **Hold-out split** — a per-user partition of Ratings into a training set and a
  held-out test set; models train on the former and are scored on the latter.
- **Top-N** — the ranked list of N movies an approach recommends to a user.
- **Precision@K** — fraction of the top-K recommendations that are relevant.
- **Recall@K** — fraction of a user's relevant items that appear in the top-K.
- **NDCG@K** — Normalized Discounted Cumulative Gain; a rank-aware relevance score
  that rewards putting relevant items higher.
- **RMSE / MAE** — rating-prediction error metrics; only meaningful for approaches
  that predict a numeric rating (SVD, and the hybrid where applicable).
- **Relevant item** — a held-out movie the user rated at or above a relevance
  threshold (the definition of a "hit" for ranking metrics).

## Data-sourcing vocabulary

- **Enrichment** — the one-time process of fetching **Content features** from the
  TMDB API for every Movie and joining them to MovieLens via `tmdbId`.
- **Cache** — the on-disk store of TMDB responses so that reruns are offline and
  reproducible; not committed to the repo.

## Interface vocabulary

- **`run`** — the CLI command that executes the full pipeline (load → enrich →
  train all approaches → evaluate) and emits the **comparison table**.
- **`recommend`** — the CLI command that returns a **Top-N** list for a given user
  (or movies similar to a given movie).
- **Comparison table** — the artifact that reports every approach's metrics
  side-by-side (written as Markdown and CSV).
