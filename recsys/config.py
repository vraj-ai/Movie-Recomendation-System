"""Configuration constants for the recommendation system."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = PROJECT_ROOT / "fixtures"
FIXTURE_DATA_DIR = FIXTURE_DIR / "ml-small"
FIXTURE_TMDB_CACHE = FIXTURE_DIR / "tmdb_cache"
DATA_DIR = PROJECT_ROOT / "data"
DATA_SUBDIR = DATA_DIR / "ml-latest-small"
CACHE_DIR = PROJECT_ROOT / "cache"
TMDB_CACHE_SUBDIR = CACHE_DIR / "tmdb"
RESULTS_DIR = PROJECT_ROOT / "results"

# MovieLens download URL
MOVIELENS_URL = "https://grouplens.org/datasets/movielens/"
MOVIELENS_DOWNLOAD_INSTRUCTIONS = (
    "Download ml-latest-small.zip from https://grouplens.org/datasets/movielens/, "
    "unzip it, and place the contents under data/ml-latest-small/ in this project."
)

# Evaluation defaults
DEFAULT_K = 10
DEFAULT_RELEVANCE_THRESHOLD = 3.5
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_HYBRID_WEIGHT = 0.5  # 0 = pure CF, 1 = pure content
DEFAULT_SVD_COMPONENTS = 20

# TMDB API
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_MAX_RETRIES = 3
TMDB_BACKOFF_FACTOR = 2.0  # exponential backoff base (seconds)
TMDB_TIMEOUT = 10  # seconds per request
TMDB_RATE_LIMIT_DELAY = 0.25  # seconds between requests (250ms = 4 req/s < 5 req/s limit)
