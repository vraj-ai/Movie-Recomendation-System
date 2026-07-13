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

# --- Effort 3: practical watch log (--full mode) ----------------------------
# Catalog contract: docs/plan/effort3-practical-watchlog.md ("Catalog contract").
CATALOG_PATH = DATA_DIR / "catalog.sqlite"
CATALOG_BUILD_INSTRUCTIONS = (
    "Build it with: python scripts/build_catalog.py "
    "(see docs/plan/effort3-practical-watchlog.md for the catalog contract)."
)

# /api/next request bounds (all validated 4xx/422, never 500 — WEB-FAIL-2)
PRACTICAL_NEXT_N_DEFAULT = 10
PRACTICAL_NEXT_N_MAX = 50
PRACTICAL_MIN_LOGGED = 3  # < this many watched movies -> 409 log_more (PRAC contract)

# Taste weight from a 5-star rating: w = (r - NEUTRAL) / SCALE, r in [0.5, 5.0].
PRACTICAL_SIGNAL_NEUTRAL = 3.0
PRACTICAL_SIGNAL_SCALE = 2.0

# Era arithmetic (decades)
PRACTICAL_ERA_TOL_MIN = 8.0        # floor added to the weighted decade spread
PRACTICAL_ERA_LOW_DECADE = 1975.0  # era slider = 0 anchor ("older classics")
PRACTICAL_ERA_HIGH_DECADE = 2015.0  # era slider = 100 anchor ("new releases")

# Bayesian shrinkage for the rating_signal component: shrink vote_average/10
# toward 0.5 when vote_count is low (avoids a single 10/10 vote dominating).
PRACTICAL_RATING_SHRINK_K = 20.0

# Multiplier range for the "indian" slider on candidate scores: at the
# slider's extreme, is_indian rows are scaled by (1 +/- this); non-indian
# rows move inversely so the two ends actually swap the top of the list.
PRACTICAL_INDIAN_BOOST = 0.8

# Final score = weighted sum of the 5 math.components (each in [0, 1]),
# weights sum to 1; the indian multiplier is applied after this sum so it
# never appears as a fabricated 6th component (contract freezes exactly 5).
PRACTICAL_W_GENRE = 0.35
PRACTICAL_W_LANGUAGE = 0.15
PRACTICAL_W_ERA = 0.20
PRACTICAL_W_POPULARITY = 0.15
PRACTICAL_W_RATING = 0.15

# "surprise" slider: pool to sample from (top decile) and how hard already-
# dominant logged genres get downweighted in that pool as surprise -> 100.
PRACTICAL_SURPRISE_POOL_FRACTION = 0.10
PRACTICAL_SURPRISE_DOMINANCE_PENALTY = 0.4

# /api/search
PRACTICAL_SEARCH_DEFAULT_LIMIT = 20
PRACTICAL_SEARCH_MAX_LIMIT = 50
