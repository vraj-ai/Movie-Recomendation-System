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

## Taste engine + web UI vocabulary (effort 2)

Effort 2 (`docs/SPEC-TASTE-WEBUI.md`) adds a personal, deterministic engine and a
local web UI. New canonical terms:

- **Operator** — the single human whose taste the engine models. Distinct from
  **User** (an anonymized MovieLens rater). The web UI serves the operator only.
- **Taste engine** — the deterministic scorer: closed-form arithmetic from the
  operator's ratings to a ranked movie list. No fitted/ML models in its path.
- **Taste rating** — a 0.5–5 star rating the operator gives a movie in the UI.
  Stored in the **profile file**; excluded movies never reappear as picks.
- **Taste profile** — the numeric summary derived from taste ratings: per-genre
  weights, era preference, and popularity preference.
- **Derived weight** — a profile number computed from taste ratings alone.
- **Pin / pinned weight** — a slider the operator has dragged. A pinned dimension
  holds its value regardless of ratings; unpinned dimensions live-track derived
  values. Effective weight = pinned value if pinned, else derived (never a blend).
- **Feature match** — the score component matching a candidate's features
  (genres, era, popularity) against the taste profile.
- **Neighbor evidence** — the score component from a candidate's similarity to
  specific movies the operator rated ("0.81 similar to Inception, rated 5★").
- **Score breakdown** — the complete per-term arithmetic attached to every
  recommendation; the terms visibly recombine to the total.
- **Data mode** — fixture vs. real, chosen explicitly at server startup; the
  server never silently swaps datasets.
- **Dataset badge** — the always-visible UI element naming the live data mode,
  dataset size, and TMDB status, so results are never misattributed.
