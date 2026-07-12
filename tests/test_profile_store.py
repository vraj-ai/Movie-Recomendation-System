"""W2 — profile store (#15): ratings + pins JSON, quarantine, gitignore."""

from __future__ import annotations

import json
from pathlib import Path

from recsys.web.profile_store import ProfileStore

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_round_trip(tmp_path):
    store = ProfileStore(tmp_path / "taste_profile.json")
    store.save(ratings={1: 5.0, 3: 0.5}, pins={"genre:Action": -1.0, "pop": 0.25})
    loaded = ProfileStore(tmp_path / "taste_profile.json").load()
    assert loaded.ratings == {1: 5.0, 3: 0.5}
    assert loaded.pins == {"genre:Action": -1.0, "pop": 0.25}


def test_missing_file_is_empty_profile(tmp_path):
    """WEB-FAIL-5: no file yet -> empty profile, no crash."""
    loaded = ProfileStore(tmp_path / "nope.json").load()
    assert loaded.ratings == {}
    assert loaded.pins == {}
    assert loaded.quarantined_to is None


def test_corrupt_json_quarantined_not_overwritten(tmp_path):
    """WEB-FAIL-3: corrupt file -> quarantine + empty profile, bytes kept."""
    p = tmp_path / "taste_profile.json"
    p.write_text("{not json!!", encoding="utf-8")
    loaded = ProfileStore(p).load()
    assert loaded.ratings == {} and loaded.pins == {}
    q = loaded.quarantined_to
    assert q is not None and q.exists()
    assert q.name.startswith("taste_profile.json.corrupt-")
    assert q.read_text(encoding="utf-8") == "{not json!!"
    assert not p.exists()  # moved, never silently rewritten in place


def test_wrong_structure_quarantined(tmp_path):
    p = tmp_path / "taste_profile.json"
    p.write_text(json.dumps({"ratings": [1, 2], "pins": {}}), encoding="utf-8")
    loaded = ProfileStore(p).load()
    assert loaded.ratings == {}
    assert loaded.quarantined_to is not None


def test_non_numeric_values_quarantined(tmp_path):
    p = tmp_path / "taste_profile.json"
    p.write_text(
        json.dumps({"ratings": {"1": "five"}, "pins": {}}), encoding="utf-8"
    )
    loaded = ProfileStore(p).load()
    assert loaded.ratings == {}
    assert loaded.quarantined_to is not None


def test_save_creates_parent_and_is_human_readable(tmp_path):
    p = tmp_path / "deep" / "taste_profile.json"
    ProfileStore(p).save(ratings={7: 3.5}, pins={})
    text = p.read_text(encoding="utf-8")
    assert "\n" in text and '"ratings"' in text and '"7"' in text  # indented JSON
    assert json.loads(text)["ratings"] == {"7": 3.5}


def test_save_then_corrupt_then_save_recovers(tmp_path):
    p = tmp_path / "taste_profile.json"
    store = ProfileStore(p)
    store.save(ratings={1: 4.0}, pins={})
    p.write_text("garbage", encoding="utf-8")
    assert store.load().ratings == {}
    store.save(ratings={2: 2.5}, pins={"era": 1980.0})
    assert store.load().ratings == {2: 2.5}


def test_profile_dir_gitignored():
    """WEB-SEC-3: the real profile location never enters git."""
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "profile/" in gitignore
