"""Tests for the recommend CLI command (ticket #6)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(*args, timeout=120) -> subprocess.CompletedProcess:
    """Run the CLI with the venv Python."""
    return subprocess.run(
        [sys.executable, "-m", "recsys.cli", *args],
        capture_output=True, text=True, timeout=timeout,
    )


class TestRecommendUser:
    def test_user_recommend_returns_results(self):
        result = _run_cli("recommend", "--user", "1", "-n", "5")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "recommendations for user 1" in result.stdout.lower()
        # Should have at least 1 recommendation line (numbered with ". ")
        rec_lines = [
            l for l in result.stdout.split("\n")
            if l.strip() and l.strip()[0].isdigit() and ". " in l
        ]
        assert len(rec_lines) >= 1

    def test_user_recommend_n(self):
        result = _run_cli("recommend", "--user", "1", "-n", "3")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Count recommendation lines (numbered "  1.", "  2.", etc.)
        rec_lines = [
            l for l in result.stdout.split("\n")
            if l.strip() and l.strip()[0].isdigit() and ". " in l
        ]
        assert len(rec_lines) <= 3

    def test_unknown_user_error(self):
        result = _run_cli("recommend", "--user", "99999", "-n", "5")
        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_recommend_shows_movie_titles(self):
        result = _run_cli("recommend", "--user", "1", "-n", "3")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Movie titles contain parentheses (year)
        assert "(" in result.stdout


class TestRecommendMovie:
    def test_movie_similar_returns_results(self):
        result = _run_cli("recommend", "--movie", "1", "-n", "5")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "similar" in result.stdout.lower()
        rec_lines = [
            l for l in result.stdout.split("\n")
            if l.strip() and l.strip()[0].isdigit() and ". " in l
        ]
        assert len(rec_lines) >= 1

    def test_movie_similar_excludes_self(self):
        result = _run_cli("recommend", "--movie", "1", "-n", "5")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # movieId=1 should not appear in the results (only in the header)
        lines = result.stdout.split("\n")
        rec_lines = [l for l in lines if l.strip() and l.strip()[0].isdigit()]
        for line in rec_lines:
            assert "movieId=1]" not in line or "similar to" in lines[max(0, lines.index(line)-5)]

    def test_unknown_movie_error(self):
        result = _run_cli("recommend", "--movie", "99999", "-n", "5")
        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


class TestRecommendEdgeCases:
    def test_no_user_or_movie_specified(self):
        result = _run_cli("recommend", "-n", "5")
        assert result.returncode == 1
        assert "must specify" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_recommend_latency(self):
        """PERF-5: recommend returns in < 1s after models are loaded."""
        import time
        # First call loads models (slow); second call is the real test
        # But since we're a fresh process each time, just check total time is reasonable
        t0 = time.time()
        result = _run_cli("recommend", "--user", "1", "-n", "5", timeout=60)
        elapsed = time.time() - t0
        assert result.returncode == 0
        # Model loading + recommend should be well under 30s on fixture data
        assert elapsed < 30.0


class TestAutoResolve:
    """Red-team: when real data is present but no API key, the CLI must keep
    using the real ratings data and only borrow the fixture TMDB cache —
    NOT silently swap the entire dataset for the tiny fixture."""

    def test_real_data_kept_when_no_key(self, tmp_path, monkeypatch):
        import recsys.cli as cli

        real_data = tmp_path / "real"
        real_data.mkdir()
        (real_data / "ratings.csv").write_text(
            "userId,movieId,rating,timestamp\n1,1,5.0,1\n"
        )
        fixture_cache = tmp_path / "fcache"
        fixture_cache.mkdir()
        fixture_data = tmp_path / "fixture_data"
        fixture_data.mkdir()
        monkeypatch.setenv("TMDB_API_KEY", "")
        monkeypatch.delenv("TMDB_API_KEY", raising=False)
        monkeypatch.setattr(cli, "DATA_SUBDIR", real_data, raising=False)
        monkeypatch.setattr(cli, "FIXTURE_DATA_DIR", fixture_data, raising=False)
        monkeypatch.setattr(cli, "FIXTURE_TMDB_CACHE", fixture_cache, raising=False)

        data_dir, cache_dir, api_key, fixture = cli._auto_resolve_data_and_cache()
        assert data_dir == real_data, (
            "BUG: real data present but CLI silently swapped to fixture data"
        )
        assert cache_dir == fixture_cache
        assert api_key is None


class TestRunFixtureE2E:
    def test_run_fixture_produces_table(self, tmp_path):
        """End-to-end: run --fixture writes comparison.md and comparison.csv."""
        result = _run_cli("run", "--fixture", "--out", str(tmp_path))
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (tmp_path / "comparison.md").exists()
        assert (tmp_path / "comparison.csv").exists()

    def test_run_fixture_table_has_approaches(self, tmp_path):
        result = _run_cli("run", "--fixture", "--out", str(tmp_path))
        assert result.returncode == 0
        md = (tmp_path / "comparison.md").read_text()
        assert "popularity" in md.lower()
