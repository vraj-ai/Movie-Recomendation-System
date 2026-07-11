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

# --- Effort 2: taste engine + web UI ---------------------------------------
# Every constant below appears verbatim in score breakdowns (MATH-2); tuning
# them is the operator's prerogative (docs/SPEC-TASTE-WEBUI.md, "The math").

# Feature match F(m) component weights (sum to 1)
TASTE_FEATURE_GENRE_W = 0.5
TASTE_FEATURE_ERA_W = 0.25
TASTE_FEATURE_POP_W = 0.25

# Similarity sim(a, b) component weights (sum to 1)
TASTE_SIM_GENRE_W = 0.6
TASTE_SIM_ERA_W = 0.25
TASTE_SIM_POP_W = 0.15

# Final score S(m) = SCALE * (FEATURE_W * F + NEIGHBOR_W * N)
TASTE_SCORE_FEATURE_W = 0.65
TASTE_SCORE_NEIGHBOR_W = 0.35
TASTE_SCORE_SCALE = 100.0

# Rating -> signed signal: s = (r - NEUTRAL) / SCALE, r in [0.5, 5.0]
TASTE_NEUTRAL_RATING = 3.0
TASTE_SIGNAL_SCALE = 2.0

# Era arithmetic
TASTE_ERA_TOL_MIN = 10.0  # decades; floor added to the weighted spread
TASTE_ERA_PROX_SCALE = 30.0  # years; sim era-proximity reaches 0 at this gap

# Web server
WEB_HOST = "127.0.0.1"  # WEB-SEC-1: localhost only
WEB_PORT = 8977
PROFILE_PATH = PROJECT_ROOT / "profile" / "taste_profile.json"  # gitignored (WEB-SEC-3)
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"  # CDN only, no key (WEB-SEC-2)
