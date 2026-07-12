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


# The frozen `--full` practical contract (effort3-practical-watchlog.md). These
# routes are owned + registered by Agent B; the redesigned page codes against
# them even in fixture mode, so the endpoint-existence check tolerates them.
FROZEN_FULL_ENDPOINTS = {"/api/search", "/api/watched", "/api/watched/", "/api/next"}


def test_page_has_all_w6_tabs(client):
    """Legacy tabs are tucked behind the Advanced panel but remain reachable."""
    page = client.get("/").text
    for tab in ("search", "recommend", "similar", "profile"):
        assert f'data-tab="{tab}"' in page, f"missing tab control: {tab}"


def test_page_never_uses_innerhtml():
    # Render via textContent/createElement only — no HTML injection sink.
    assert "innerHTML" not in PAGE_PATH.read_text(encoding="utf-8")


def test_page_references_only_existing_endpoints(app, client):
    """Every fetch() target in the page must be a registered route."""
    page = client.get("/").text
    # API calls go through fetch() directly and through the getJSON() wrapper.
    called = {
        m.split("?")[0].split("${")[0].rstrip("?")
        for m in re.findall(r"""(?:fetch|getJSON)\(\s*[`"']([^`"')]+)""", page)
    }
    assert called, "page makes no API calls — tabs cannot be wired"
    registered = {r.path for r in app.routes}
    # Catches typo'd endpoints while allowing the frozen full-mode contract
    # routes that Agent B registers (not present in fixture-mode routing here).
    unknown = {
        p for p in called
        if p and p not in registered and p not in FROZEN_FULL_ENDPOINTS
    }
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


# == W10 practical redesign =================================================

PAGE = PAGE_PATH.read_text(encoding="utf-8")


def _strip_code(html: str) -> str:
    """Visible markup only: drop <script> and <style> bodies."""
    no_script = re.sub(r"<script\b.*?</script>", "", html, flags=re.S | re.I)
    return re.sub(r"<style\b.*?</style>", "", no_script, flags=re.S | re.I)


# -- Slider contract (FROZEN): exact names, end labels, helper text --------

# (UI name, left label, right label, helper text) — verbatim from
# docs/plan/effort3-practical-watchlog.md "Slider contract".
SLIDER_CONTRACT = [
    ("Comfort zone", "Stick to my taste", "Surprise me",
     "Left: more like the movies you already love. "
     "Right: stretch into new genres and styles."),
    ("Fame", "Hidden gems", "Blockbusters",
     "Left: lesser-known films worth digging up. "
     "Right: crowd-pleasers everyone's heard of."),
    ("Era", "Older classics", "New releases",
     "Which decades your picks lean toward."),
    ("Cinema mix", "Global", "Indian",
     "Balance between international films and Indian cinema "
     "(Hindi, Tamil, Telugu, Malayalam"),
]


@pytest.mark.parametrize("name,left,right,helper", SLIDER_CONTRACT)
def test_slider_contract_strings_present(name, left, right, helper):
    body = _strip_code(PAGE)
    for text in (name, left, right, helper):
        assert text in body, f"slider text missing from visible markup: {text!r}"


def test_all_four_sliders_default_to_midpoint():
    # Frozen contract: four sliders, 0..100, default 50.
    ranges = re.findall(
        r'<input type="range"[^>]*min="0"[^>]*max="100"[^>]*value="50"', PAGE
    )
    assert len(ranges) == 4
    # Each declared slider carries a frozen param name.
    for param in ("surprise", "crowd", "era", "indian"):
        assert f'data-param="{param}"' in PAGE


# -- PRAC-3: math hidden by default ----------------------------------------


def test_show_the_math_button_present_but_panel_hidden_by_default():
    assert "Show the math" in PAGE            # explicit reveal control exists
    assert "math-panel" in PAGE               # the panel template exists
    assert "panel.hidden = true" in PAGE      # ...but starts hidden (PRAC-3)


def test_default_visible_markup_has_no_raw_math():
    """PRAC-3: no scores / percentages / weight numerics in default markup.

    All math lives inside the JS-built math panel (inside <script>), never in
    the statically served, always-visible body.
    """
    body = _strip_code(PAGE)
    assert "score" not in body.lower()
    assert "%" not in body
    # No bare decimal numbers-as-weights on display (e.g. 0.81, 3.14).
    assert not re.search(r"\b\d+\.\d+\b", body), "decimal weight visible by default"


# -- star rating control (0.5–5, half steps) -------------------------------


def test_star_rating_control_markup_present():
    assert "starControl" in PAGE              # the 5-star widget builder
    assert 'el("span", "star"' in PAGE        # individual star elements
    # committed rating is POSTed to the frozen watched endpoint
    assert '"/api/watched"' in PAGE


# -- PRAC-4: styled poster fallback, never a broken image ------------------


def test_poster_fallback_markup_exists():
    assert "fallbackPoster" in PAGE           # deterministic gradient builder
    assert "poster-fallback" in PAGE          # its styled class
    # onerror handler swaps a failed <img> for the fallback (never broken img)
    assert 'addEventListener("error"' in PAGE


# -- data-mode badge invariant + full mode ---------------------------------


def test_badge_supports_all_three_data_modes():
    for mode in ("fixture", "real", "full"):
        assert f"#badge.{mode}" in PAGE       # styled per mode
    assert 'id="badge"' in PAGE               # always-visible element


# -- Advanced / legacy screens still reachable -----------------------------


def test_advanced_panel_reveals_legacy_tabs():
    assert 'id="advanced"' in PAGE
    assert 'id="advanced-toggle"' in PAGE
    assert "/api/benchmark" in PAGE           # legacy JS preserved
    assert "/api/recommend" in PAGE
    assert "/api/similar" in PAGE
