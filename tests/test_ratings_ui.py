"""W14 — UI ratings row + explanation line on cards (effort 4, #26).

The frontend is logic-thin, so like test_web_ui.py this is contract-tested
against the served static page: the ratings row (IMDb / 🍅 / age badge /
match ×/10) must exist, NULL ratings must render as "—" and never be
invented (RAT-1 / RAT-FAIL-3), and next-watch cards must carry the single
explanation sentence instead of the old templated chips (RAT-3).

Verification-command: .venv/bin/pytest tests/test_ratings_ui.py -q
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_PATH = REPO_ROOT / "recsys" / "web" / "static" / "index.html"
PAGE = PAGE_PATH.read_text(encoding="utf-8")


# -- ratings row ------------------------------------------------------------


def test_ratings_row_markup_present():
    assert "ratings-row" in PAGE
    assert "ratingsRow(" in PAGE


def test_ratings_row_covers_all_four_facts():
    """IMDb, Rotten Tomatoes, age certification, and the match badge."""
    assert "imdb_rating" in PAGE
    assert "rt_rating" in PAGE
    assert "age_rating" in PAGE
    assert "match10" in PAGE
    assert "match-badge" in PAGE


def test_null_ratings_render_as_dash_never_invented():
    """RAT-1 / RAT-FAIL-3: missing data renders as an em-dash placeholder."""
    fn = re.search(r"function ratingsRow\([\s\S]*?\n    }", PAGE)
    assert fn, "ratingsRow() missing"
    assert "—" in fn.group(0)


def test_match_badge_is_out_of_ten():
    assert "/10" in PAGE


# -- explanation line -------------------------------------------------------


def test_explanation_line_markup_present():
    assert "explanation" in PAGE
    assert 'el("p", "explanation"' in PAGE


def test_cards_prefer_explanation_over_chips():
    """RAT-3: rec cards render the single sentence when the API provides it;
    the chip list is only a fallback for payloads without one."""
    fn = re.search(r"function explanationLine\([\s\S]*?\n    }", PAGE)
    assert fn, "explanationLine() missing"
    assert "reasonChips" in fn.group(0)  # fallback path retained
    assert "m.explanation" in fn.group(0)
    # both card renderers go through the explanation line, not raw chips
    for renderer in ("function heroCard", "function recCard"):
        body = re.search(renderer + r"\([\s\S]*?\n    }", PAGE)
        assert body, f"{renderer} missing"
        assert "explanationLine(m)" in body.group(0)
        assert "reasonChips(m.reasons)" not in body.group(0)


def test_page_still_never_uses_innerhtml():
    """XSS guard (WEB-SEC-2) must survive the W14 edit."""
    assert "innerHTML" not in PAGE
