"""W6 — web UI tabs (#19): page contract, posters, text-only fallback.

The frontend is logic-thin, so it is contract-tested: the served page must
reference only endpoints that actually exist, must never use innerHTML
(XSS guard), and the poster pipeline must emit only pre-resolved
``image.tmdb.org`` CDN URLs or ``null`` (WEB-SEC-2, WEB-FAIL-4).

Verification-command: .venv/bin/pytest tests/test_web_ui.py -q
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recsys.config import FIXTURE_TMDB_CACHE
from recsys.tmdb import TMDBClient
from recsys.web.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_PATH = REPO_ROOT / "recsys" / "web" / "static" / "index.html"

SENTINEL_KEY = "tmdb-key-should-never-leak-0xDEADBEEF"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    profile = tmp_path_factory.mktemp("profile") / "taste_profile.json"
    return create_app(
        mode="fixture", profile_path=profile, tmdb_api_key=SENTINEL_KEY
    )


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app) as c:
        yield c


# -- page structure --------------------------------------------------------


def test_page_has_all_w6_tabs(client):
    page = client.get("/").text
    for tab in ("search", "recommend", "similar", "profile"):
        assert f'data-tab="{tab}"' in page, f"missing tab control: {tab}"


def test_page_never_uses_innerhtml():
    # Render via textContent/createElement only — no HTML injection sink.
    assert "innerHTML" not in PAGE_PATH.read_text(encoding="utf-8")


def test_page_references_only_existing_endpoints(app, client):
    """Every fetch() target in the page must be a registered route."""
    page = client.get("/").text
    called = {
        m.split("?")[0].split("${")[0].rstrip("?")
        for m in re.findall(r"""fetch\(\s*[`"']([^`"')]+)""", page)
    }
    assert called, "page makes no API calls — tabs cannot be wired"
    registered = {r.path for r in app.routes}
    unknown = {p for p in called if p and p not in registered}
    assert not unknown, f"page calls unregistered endpoints: {sorted(unknown)}"


# -- posters: WEB-FAIL-4 + WEB-SEC-2 ---------------------------------------


def test_extract_features_picks_up_poster_path():
    c = TMDBClient(api_key=None)
    out = c._extract_features({"poster_path": "/abc123.jpg", "genres": []})
    assert out["poster_path"] == "/abc123.jpg"


def test_extract_features_coerces_junk_poster_to_none():
    c = TMDBClient(api_key=None)
    for junk in (42, ["x"], {"a": 1}, ""):
        assert c._extract_features({"poster_path": junk})["poster_path"] is None


def test_old_cache_entries_without_poster_still_load(tmp_path):
    """Pre-W6 cache entries lack poster_path; they must NOT be re-fetched."""
    entry = {
        "genres": [], "keywords": [], "cast": [], "crew": [], "overview": ""
    }
    (tmp_path / "77.json").write_text(json.dumps(entry), encoding="utf-8")
    c = TMDBClient(api_key=None, cache_dir=tmp_path)
    loaded = c._load_cache(77)
    assert loaded is not None, "legacy cache entry treated as corrupt"


def test_fixture_mode_poster_url_is_null_not_missing(client):
    """Fixture cache has no art: poster_url must be present and null."""
    r = client.get("/api/recommend", params={"n": 5})
    items = r.json()["results"]
    assert items
    for item in items:
        assert "poster_url" in item
        assert item["poster_url"] is None  # text card fallback, no broken img

    hits = client.get("/api/search", params={"q": "toy"}).json()["results"]
    assert hits and all("poster_url" in m for m in hits)


def test_poster_url_is_cdn_only_and_never_the_key(tmp_path):
    """When the cache has art, the browser sees image.tmdb.org URLs only."""
    cache = tmp_path / "tmdb_cache"
    shutil.copytree(FIXTURE_TMDB_CACHE, cache)
    for f in cache.glob("*.json"):
        entry = json.loads(f.read_text(encoding="utf-8"))
        entry["poster_path"] = "/poster-art.jpg"
        f.write_text(json.dumps(entry), encoding="utf-8")

    app = create_app(
        mode="fixture",
        profile_path=tmp_path / "p.json",
        tmdb_cache_dir=cache,
        tmdb_api_key=SENTINEL_KEY,
    )
    with TestClient(app) as c:
        items = c.get("/api/recommend", params={"n": 100}).json()["results"]
        with_art = [i for i in items if i["poster_url"] is not None]
        assert with_art, "no poster made it through the resolution path"
        for item in with_art:
            assert item["poster_url"].startswith("https://image.tmdb.org/")
            assert SENTINEL_KEY not in item["poster_url"]

        sim = c.get(
            "/api/similar", params={"movie_id": items[0]["movie_id"], "n": 5}
        ).json()["results"]
        assert all("poster_url" in s for s in sim)
