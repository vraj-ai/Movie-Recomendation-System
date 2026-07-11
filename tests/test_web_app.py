"""W3 — server skeleton (#16): explicit data mode, status, search, security.

Verification-command: .venv/bin/pytest tests/test_web_app.py -q
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recsys.config import FIXTURE_TMDB_CACHE
from recsys.web.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent

SENTINEL_KEY = "tmdb-key-should-never-leak-0xDEADBEEF"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    profile = tmp_path_factory.mktemp("profile") / "taste_profile.json"
    app = create_app(
        mode="fixture",
        profile_path=profile,
        tmdb_api_key=SENTINEL_KEY,
    )
    with TestClient(app) as c:
        yield c


# -- WEB-FAIL-1: explicit mode, fail fast, never silent swap ------------


def test_mode_is_required_and_explicit():
    with pytest.raises(ValueError):
        create_app(mode="auto")  # no such thing — explicit only


def test_real_mode_missing_dataset_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_app(
            mode="real",
            data_dir=tmp_path / "nonexistent",
            profile_path=tmp_path / "p.json",
        )


# -- basic serving -------------------------------------------------------


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()


def test_status_reports_mode_counts_and_profile(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "fixture"
    assert body["dataset"]["movies"] > 0
    assert body["dataset"]["ratings"] > 0
    assert body["dataset"]["users"] > 0
    assert body["tmdb"]["cached_movies"] == len(
        list(FIXTURE_TMDB_CACHE.glob("*.json"))
    )
    assert body["profile"] == {"ratings": 0, "pins": 0, "quarantined": False}


# -- search --------------------------------------------------------------


def test_search_finds_title_case_insensitive(client):
    r = client.get("/api/search", params={"q": "toy STORY"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert any(m["title"].startswith("Toy Story") for m in results)
    first = results[0]
    assert set(first) >= {"movieId", "title", "genres"}
    assert isinstance(first["genres"], list)


def test_search_no_match_is_empty_list(client):
    r = client.get("/api/search", params={"q": "zzzzxqwv-no-such-movie"})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_search_results_are_capped(client):
    r = client.get("/api/search", params={"q": "a"})
    assert len(r.json()["results"]) <= 20


# -- WEB-FAIL-2: bad input -> 4xx, never 500 ------------------------------


@pytest.mark.parametrize(
    "params", [{}, {"q": ""}, {"q": "   "}, {"q": "x" * 300}]
)
def test_search_bad_input_is_4xx(client, params):
    r = client.get("/api/search", params=params)
    assert 400 <= r.status_code < 500


# -- WEB-SEC-1: localhost bind by default ---------------------------------


def test_default_bind_is_localhost():
    from recsys.web.__main__ import build_parser

    args = build_parser().parse_args(["--fixture"])
    assert args.host == "127.0.0.1"


# -- WEB-SEC-2: the TMDB key never appears in any response ----------------


def test_key_never_in_responses(client):
    for path in ("/", "/api/status", "/api/search?q=toy"):
        r = client.get(path)
        assert SENTINEL_KEY not in r.text, f"key leaked via {path}"


# -- MATH-3: no fitted-model code in the serving path ---------------------


def test_serving_path_imports_no_recommenders():
    code = (
        "import sys; import recsys.web.app; "
        "bad = [m for m in sys.modules if m.startswith('recsys.recommenders') "
        "or m.startswith('sklearn')]; "
        "assert not bad, f'ML modules in serving path: {bad}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)


# -- WEB-PERF-1: fixture startup < 5 s ------------------------------------


def test_fixture_startup_under_5s(tmp_path):
    t0 = time.perf_counter()
    create_app(mode="fixture", profile_path=tmp_path / "p.json")
    assert time.perf_counter() - t0 < 5.0
