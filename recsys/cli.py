"""CLI entry point: ``run`` (full pipeline → comparison table) and
``recommend`` (top-N for a user / similar-to-movie).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

from recsys.config import (
    DEFAULT_K,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_TEST_FRACTION,
    DEFAULT_HYBRID_WEIGHT,
    DEFAULT_SVD_COMPONENTS,
    RESULTS_DIR,
    FIXTURE_DATA_DIR,
    FIXTURE_TMDB_CACHE,
    DATA_SUBDIR,
    TMDB_CACHE_SUBDIR,
)
from recsys.data import load_ratings, RatingsData
from recsys.split import holdout_split
from recsys.evaluate import evaluate_recommender
from recsys.tmdb import TMDBClient, TMDBError
from recsys.enrichment import enrich
from recsys.recommenders import (
    PopularityRecommender,
    ContentRecommender,
    SVDRecommender,
    HybridRecommender,
)

logger = logging.getLogger("recsys")


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _load_env():
    """Load ``.env`` if present (SEC-1: key from env only)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _resolve_data_and_cache(fixture: bool) -> tuple[Path, Path, str | None]:
    """Resolve data dir, TMDB cache dir, and API key.

    In fixture mode: use bundled fixtures, no API key needed (cache-only).
    In real mode: read API key from env, fail fast if missing (FAIL-1).
    """
    if fixture:
        return FIXTURE_DATA_DIR, FIXTURE_TMDB_CACHE, None

    # Real mode — need API key
    _load_env()
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        print(
            "ERROR: TMDB_API_KEY is not set (FAIL-1).\n"
            "  Set it in .env or as an environment variable.\n"
            "  Get a free key from https://www.themoviedb.org/settings/api",
            file=sys.stderr,
        )
        sys.exit(1)

    return DATA_SUBDIR, TMDB_CACHE_SUBDIR, api_key


def _auto_resolve_data_and_cache() -> tuple[Path, Path, str | None, bool]:
    """Auto-resolve: try real data, fall back to fixture."""
    if DATA_SUBDIR.exists():
        _load_env()
        api_key = os.environ.get("TMDB_API_KEY")
        if api_key:
            return DATA_SUBDIR, TMDB_CACHE_SUBDIR, api_key, False
        # Real data exists but no API key — keep the real ratings data and
        # borrow the fixture TMDB cache for content features if available.
        if FIXTURE_TMDB_CACHE.exists():
            return DATA_SUBDIR, FIXTURE_TMDB_CACHE, None, False
        return DATA_SUBDIR, TMDB_CACHE_SUBDIR, None, False
    # No real data — use fixture
    return FIXTURE_DATA_DIR, FIXTURE_TMDB_CACHE, None, True


def _train_all(
    train: RatingsData,
    content_features: dict | None,
    k: int = DEFAULT_K,
    svd_components: int = DEFAULT_SVD_COMPONENTS,
    hybrid_weight: float = DEFAULT_HYBRID_WEIGHT,
) -> list:
    """Train all four recommenders and return them."""
    recommenders = []

    pop = PopularityRecommender()
    pop.fit(train)
    recommenders.append(pop)

    if content_features is not None:
        content = ContentRecommender()
        content.fit(train, content_features=content_features)
        recommenders.append(content)

    svd = SVDRecommender(n_components=svd_components)
    svd.fit(train)
    recommenders.append(svd)

    if content_features is not None:
        hybrid = HybridRecommender(
            content_weight=hybrid_weight,
            n_components=svd_components,
        )
        hybrid.fit(train, content_features=content_features)
        recommenders.append(hybrid)

    return recommenders


def _format_metric(value: float | None, fmt: str = "{:.4f}") -> str:
    if value is None:
        return "—"
    return fmt.format(value)


def _write_comparison_table(
    results: list[dict],
    out_dir: Path,
    k: int,
) -> tuple[Path, Path]:
    """Write the comparison table as Markdown + CSV. Return ``(md_path, csv_path)``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "comparison.md"
    csv_path = out_dir / "comparison.csv"

    # --- Markdown ---
    lines = [
        "# Movie Recommendation System — Comparison Table",
        "",
        f"K = {k}",
        "",
        f"| Approach | Precision@{k} | Recall@{k} | NDCG@{k} | RMSE | MAE |",
        f"|----------|-------------|----------|---------|------|-----|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} "
            f"| {_format_metric(r.get('precision_at_k'))} "
            f"| {_format_metric(r.get('recall_at_k'))} "
            f"| {_format_metric(r.get('ndcg_at_k'))} "
            f"| {_format_metric(r.get('rmse'))} "
            f"| {_format_metric(r.get('mae'))} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- CSV ---
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Approach", f"Precision@{k}", f"Recall@{k}",
            f"NDCG@{k}", "RMSE", "MAE",
        ])
        for r in results:
            writer.writerow([
                r["name"],
                _format_metric(r.get("precision_at_k")),
                _format_metric(r.get("recall_at_k")),
                _format_metric(r.get("ndcg_at_k")),
                _format_metric(r.get("rmse")),
                _format_metric(r.get("mae")),
            ])

    return md_path, csv_path


# ---------------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------------- #

def cmd_run(args) -> int:
    """Execute the full pipeline and write a comparison table."""
    fixture = args.fixture
    data_dir, cache_dir, api_key = _resolve_data_and_cache(fixture)

    t0 = time.time()

    # 1. Load data
    print("Loading data...")
    data = load_ratings(fixture=fixture)
    print(f"  {data.n_ratings} ratings, {data.n_users} users, {data.n_movies} movies")

    # 2. Split
    print("Splitting train/test...")
    train_df, test_df = holdout_split(
        data.ratings,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"  train={len(train_df)}, test={len(test_df)}")

    # Rebuild RatingsData with train split for the sparse matrix
    from recsys.data import RatingsData
    import scipy.sparse as sp
    import numpy as np

    train_movie_ids = sorted(train_df["movieId"].unique())
    train_user_ids = sorted(train_df["userId"].unique())
    movie_id_to_idx = {mid: i for i, mid in enumerate(train_movie_ids)}
    user_id_to_idx = {uid: i for i, uid in enumerate(train_user_ids)}
    rows = train_df["userId"].map(user_id_to_idx).values
    cols = train_df["movieId"].map(movie_id_to_idx).values
    vals = train_df["rating"].values.astype(np.float64)
    user_item = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(train_user_ids), len(train_movie_ids)),
    )
    train = RatingsData(
        ratings=train_df,
        movies=data.movies,
        links=data.links,
        user_item=user_item,
        movie_ids=train_movie_ids,
        user_ids=train_user_ids,
        movie_id_to_idx=movie_id_to_idx,
        user_id_to_idx=user_id_to_idx,
    )

    # 3. Enrich
    print("Enriching with TMDB features...")
    client = TMDBClient(api_key=api_key, cache_dir=cache_dir)
    try:
        content_features = enrich(train_movie_ids, data.tmdb_map, client)
    except TMDBError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    n_with_content = sum(
        1 for v in content_features.values() if v.get("genres") or v.get("overview")
    )
    print(f"  {n_with_content} movies with content features")

    # 4. Train all
    print("Training recommenders...")
    recommenders = _train_all(
        train, content_features,
        svd_components=args.svd_components,
        hybrid_weight=args.hybrid_weight,
    )
    for rec in recommenders:
        print(f"  trained: {rec.name}")

    # 5. Evaluate
    print("Evaluating...")
    results = []
    for rec in recommenders:
        metrics = evaluate_recommender(
            rec, train, test_df,
            k=args.k,
            relevance_threshold=args.relevance_threshold,
            n=args.k,
        )
        metrics["name"] = rec.name
        results.append(metrics)
        print(
            f"  {rec.name}: "
            f"P@{args.k}={_format_metric(metrics.get('precision_at_k'))} "
            f"R@{args.k}={_format_metric(metrics.get('recall_at_k'))} "
            f"NDCG@{args.k}={_format_metric(metrics.get('ndcg_at_k'))}"
            + (f" RMSE={_format_metric(metrics.get('rmse'))}" if "rmse" in metrics else "")
        )

    # 6. Write comparison table
    out_dir = Path(args.out) if args.out else RESULTS_DIR
    md_path, csv_path = _write_comparison_table(results, out_dir, args.k)
    print(f"\nComparison table written to:")
    print(f"  {md_path}")
    print(f"  {csv_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    return 0


def cmd_recommend(args) -> int:
    """Return a top-N list for a user or movies similar to a given movie."""
    data_dir, cache_dir, api_key, fixture = _auto_resolve_data_and_cache()

    print("Loading data...")
    data = load_ratings(data_dir=data_dir)
    print(f"  {data.n_ratings} ratings, {data.n_users} users, {data.n_movies} movies")

    # Enrich for content features
    client = TMDBClient(api_key=api_key, cache_dir=cache_dir)
    try:
        content_features = enrich(data.movie_ids, data.tmdb_map, client)
    except TMDBError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.movie is not None:
        # Similar-to-movie mode
        movie_id = args.movie
        if movie_id not in data.movie_ids:
            print(f"ERROR: movieId {movie_id} not found in dataset.", file=sys.stderr)
            return 1

        rec = ContentRecommender()
        rec.fit(data, content_features=content_features)

        title_row = data.movies[data.movies["movieId"] == movie_id]
        title = title_row.iloc[0]["title"] if len(title_row) else f"movieId={movie_id}"
        print(f"\nMovies similar to {title} [movieId={movie_id}]:")

        t0 = time.time()
        similar = rec.similar(movie_id, n=args.n)
        elapsed = time.time() - t0

        for i, mid in enumerate(similar, 1):
            row = data.movies[data.movies["movieId"] == mid]
            t = row.iloc[0]["title"] if len(row) else f"movieId={mid}"
            print(f"  {i}. {t} [movieId={mid}]")
        print(f"({elapsed:.3f}s)")
        return 0

    if args.user is not None:
        user_id = args.user
        if user_id not in data.user_ids:
            print(f"ERROR: userId {user_id} not found in dataset.", file=sys.stderr)
            return 1

        # Use hybrid if content available, else SVD
        if content_features:
            rec = HybridRecommender(
                content_weight=args.hybrid_weight,
                n_components=args.svd_components,
            )
        else:
            rec = SVDRecommender(n_components=args.svd_components)
        rec.fit(data, content_features=content_features)

        print(f"\nTop-{args.n} recommendations for user {user_id}:")
        t0 = time.time()
        recommendations = rec.recommend(user_id, n=args.n)
        elapsed = time.time() - t0

        for i, mid in enumerate(recommendations, 1):
            row = data.movies[data.movies["movieId"] == mid]
            t = row.iloc[0]["title"] if len(row) else f"movieId={mid}"
            print(f"  {i}. {t} [movieId={mid}]")
        print(f"({elapsed:.3f}s)")
        return 0

    print("ERROR: must specify --user or --movie", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------- #
# Argument parsing
# ---------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys",
        description="Movie Recommendation System — algorithm experiment",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = sub.add_parser("run", help="Run the full pipeline and emit a comparison table")
    run_parser.add_argument("--fixture", action="store_true", help="Use bundled fixture dataset")
    run_parser.add_argument("--out", default=None, help="Output directory for comparison table")
    run_parser.add_argument("-k", type=int, default=DEFAULT_K, help="K for metrics")
    run_parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    run_parser.add_argument("--relevance-threshold", type=float, default=DEFAULT_RELEVANCE_THRESHOLD)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--svd-components", type=int, default=DEFAULT_SVD_COMPONENTS)
    run_parser.add_argument("--hybrid-weight", type=float, default=DEFAULT_HYBRID_WEIGHT)
    run_parser.set_defaults(func=cmd_run)

    # recommend
    rec_parser = sub.add_parser("recommend", help="Get top-N recommendations")
    rec_parser.add_argument("--user", type=int, default=None, help="User ID to recommend for")
    rec_parser.add_argument("--movie", type=int, default=None, help="Movie ID to find similar movies for")
    rec_parser.add_argument("-n", type=int, default=10, help="Number of results")
    rec_parser.add_argument("--svd-components", type=int, default=DEFAULT_SVD_COMPONENTS)
    rec_parser.add_argument("--hybrid-weight", type=float, default=DEFAULT_HYBRID_WEIGHT)
    rec_parser.set_defaults(func=cmd_recommend)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
