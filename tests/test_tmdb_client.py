"""Tests for the TMDB client (ticket #2).

All tests run offline with a mocked HTTP layer — no real network.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import requests

from recsys.tmdb import TMDBClient, TMDBError, _empty_features


# ---------------------------------------------------------------------- #
# Mock HTTP layer
# ---------------------------------------------------------------------- #

class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}

    def json(self):
        return self._json


class FakeSession:
    """A fake requests.Session that returns canned responses."""

    def __init__(self):
        self._responses: dict[str, list[FakeResponse]] = {}
        self.calls: list[dict] = []

    def set_response(self, url, response):
        if url not in self._responses:
            self._responses[url] = []
        self._responses[url].append(response)

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if url in self._responses and self._responses[url]:
            return self._responses[url].pop(0)
        return FakeResponse(404)

    def close(self):
        pass


def _tmdb_url(tmdb_id):
    return f"https://api.themoviedb.org/3/movie/{tmdb_id}"


def _make_raw_response():
    """A valid raw TMDB API response."""
    return {
        "id": 862,
        "title": "Toy Story",
        "overview": "A cowboy doll and a space ranger compete for a boy's love.",
        "genres": [{"id": 16, "name": "Animation"}, {"id": 35, "name": "Comedy"}],
        "keywords": {"keywords": [{"id": 1, "name": "toy"}, {"id": 2, "name": "friendship"}]},
        "credits": {
            "cast": [{"id": 31, "name": "Tom Hanks"}, {"id": 32, "name": "Tim Allen"}],
            "crew": [
                {"id": 1, "name": "John Lasseter", "job": "Director"},
                {"id": 2, "name": "Someone Else", "job": "Producer"},
            ],
        },
    }


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #

class TestCacheHit:
    def test_cache_hit_no_network(self, tmp_path):
        """Cache hit → no network call (SEC-2)."""
        tmdb_id = 862
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / f"{tmdb_id}.json"
        cache_file.write_text(json.dumps(_empty_features() | {"genres": ["Test"]}))

        client = TMDBClient(api_key="fake_key", cache_dir=cache_dir)
        client._session = FakeSession()

        result = client.get_movie(tmdb_id)
        assert result is not None
        assert result["genres"] == ["Test"]
        # No network calls should have been made
        assert len(client._session.calls) == 0

    def test_cache_miss_fetches_from_api(self, tmp_path):
        """Cache miss → fetch from API and cache the result."""
        tmdb_id = 862
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        client = TMDBClient(api_key="fake_key", cache_dir=cache_dir, rate_limit_delay=0)
        session = FakeSession()
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(200, _make_raw_response()))
        client._session = session

        result = client.get_movie(tmdb_id)
        assert result is not None
        assert "Animation" in result["genres"]
        assert "Tom Hanks" in result["cast"]
        # Should have been cached
        cache_file = cache_dir / f"{tmdb_id}.json"
        assert cache_file.exists()
        cached = json.loads(cache_file.read_text())
        assert "Animation" in cached["genres"]


class TestCacheOnlyMode:
    def test_no_key_cache_miss_returns_none(self, tmp_path):
        """api_key=None (cache-only) + cache miss → None with warning (FAIL-5)."""
        client = TMDBClient(api_key=None, cache_dir=tmp_path / "cache")
        client._session = FakeSession()
        result = client.get_movie(99999)
        assert result is None
        assert len(client._session.calls) == 0  # no network attempt


class TestFailureModes:
    def test_5xx_retries_then_skips(self, tmp_path):
        """FAIL-2: 5xx → retry with backoff, then skip."""
        tmdb_id = 100
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(
            api_key="fake", cache_dir=cache_dir,
            max_retries=3, backoff_factor=0.01, rate_limit_delay=0,
        )
        session = FakeSession()
        for _ in range(3):
            session.set_response(_tmdb_url(tmdb_id), FakeResponse(500))
        client._session = session

        result = client.get_movie(tmdb_id)
        assert result is None
        assert len(session.calls) == 3  # 3 retries

    def test_timeout_retries_then_skips(self, tmp_path):
        """FAIL-2: timeout → retry, then skip."""
        tmdb_id = 101
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(
            api_key="fake", cache_dir=cache_dir,
            max_retries=2, backoff_factor=0.01, rate_limit_delay=0,
        )
        session = FakeSession()
        session.set_response(_tmdb_url(tmdb_id), None)  # signal timeout
        client._session = session

        # Override get to raise Timeout
        original_get = session.get

        def timeout_get(url, params=None, timeout=None):
            session.calls.append({"url": url, "params": params})
            raise requests.exceptions.Timeout("timed out")

        session.get = timeout_get

        result = client.get_movie(tmdb_id)
        assert result is None
        assert len(session.calls) == 2  # 2 retries

    def test_429_honors_retry_after(self, tmp_path):
        """FAIL-3: 429 → honor Retry-After, then succeed on retry."""
        tmdb_id = 102
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(
            api_key="fake", cache_dir=cache_dir,
            max_retries=3, backoff_factor=0.01, rate_limit_delay=0,
        )
        session = FakeSession()
        # First: 429 with Retry-After: 0 (no real wait)
        session.set_response(
            _tmdb_url(tmdb_id),
            FakeResponse(429, headers={"Retry-After": "0"}),
        )
        # Then: success
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(200, _make_raw_response()))
        client._session = session

        result = client.get_movie(tmdb_id)
        assert result is not None
        assert len(session.calls) == 2  # 429 then success

    def test_404_skips(self, tmp_path):
        """FAIL-5: 404 → skip (movie doesn't exist)."""
        tmdb_id = 103
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(
            api_key="fake", cache_dir=cache_dir,
            max_retries=3, backoff_factor=0.01, rate_limit_delay=0,
        )
        session = FakeSession()
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(404))
        client._session = session

        result = client.get_movie(tmdb_id)
        assert result is None
        assert len(session.calls) == 1  # no retry on 404

    def test_4xx_non_retryable(self, tmp_path):
        """400/403 → skip immediately (no retry)."""
        tmdb_id = 104
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(
            api_key="fake", cache_dir=cache_dir,
            max_retries=3, backoff_factor=0.01, rate_limit_delay=0,
        )
        session = FakeSession()
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(403))
        client._session = session

        result = client.get_movie(tmdb_id)
        assert result is None
        assert len(session.calls) == 1


class TestMalformedCoercion:
    def test_missing_fields_coerced_to_empty(self, tmp_path):
        """FAIL-4: missing/malformed fields → empty content features."""
        tmdb_id = 200
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(
            api_key="fake", cache_dir=cache_dir, rate_limit_delay=0,
        )
        session = FakeSession()
        # Response with missing genres, malformed keywords, no credits
        raw = {"id": tmdb_id, "title": "Bad Movie"}
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(200, raw))
        client._session = session

        result = client.get_movie(tmdb_id)
        assert result is not None
        assert result["genres"] == []
        assert result["keywords"] == []
        assert result["cast"] == []
        assert result["crew"] == []
        assert result["overview"] == ""

    def test_partial_fields_coerced(self, tmp_path):
        """Some fields present, some malformed."""
        tmdb_id = 201
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(
            api_key="fake", cache_dir=cache_dir, rate_limit_delay=0,
        )
        session = FakeSession()
        raw = {
            "id": tmdb_id,
            "genres": [{"name": "Action"}, {"name": "Thriller"}],
            "keywords": "not a dict",  # malformed
            "credits": "broken",  # malformed
            "overview": 12345,  # wrong type
        }
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(200, raw))
        client._session = session

        result = client.get_movie(tmdb_id)
        assert result is not None
        assert result["genres"] == ["Action", "Thriller"]
        assert result["keywords"] == []
        assert result["cast"] == []
        assert result["crew"] == []
        assert result["overview"] == ""

    def test_crew_only_director_and_writer(self, tmp_path):
        """Only Director and Writer crew members are extracted."""
        tmdb_id = 202
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(
            api_key="fake", cache_dir=cache_dir, rate_limit_delay=0,
        )
        session = FakeSession()
        raw = {
            "id": tmdb_id,
            "genres": [],
            "credits": {
                "cast": [],
                "crew": [
                    {"name": "Dir Person", "job": "Director"},
                    {"name": "Prod Person", "job": "Producer"},
                    {"name": "Wri Person", "job": "Writer"},
                ],
            },
        }
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(200, raw))
        client._session = session

        result = client.get_movie(tmdb_id)
        assert "Dir Person" in result["crew"]
        assert "Wri Person" in result["crew"]
        assert "Prod Person" not in result["crew"]


class TestCacheIntegrity:
    def test_corrupt_cache_refetched(self, tmp_path):
        """FAIL-7: corrupt cache entry → detected and re-fetched."""
        tmdb_id = 300
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Write corrupt cache
        cache_file = cache_dir / f"{tmdb_id}.json"
        cache_file.write_text("{ this is not valid json")

        client = TMDBClient(
            api_key="fake", cache_dir=cache_dir, rate_limit_delay=0,
        )
        session = FakeSession()
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(200, _make_raw_response()))
        client._session = session

        result = client.get_movie(tmdb_id)
        assert result is not None
        # Should have made a network call (re-fetch)
        assert len(session.calls) == 1
        # Cache should now be valid
        cached = json.loads(cache_file.read_text())
        assert "genres" in cached

    def test_missing_keys_cache_refetched(self, tmp_path):
        """FAIL-7: cache entry missing required keys → re-fetched."""
        tmdb_id = 301
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / f"{tmdb_id}.json"
        cache_file.write_text(json.dumps({"genres": ["Action"]}))  # missing keys

        client = TMDBClient(
            api_key="fake", cache_dir=cache_dir, rate_limit_delay=0,
        )
        session = FakeSession()
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(200, _make_raw_response()))
        client._session = session

        result = client.get_movie(tmdb_id)
        assert result is not None
        assert len(session.calls) == 1  # re-fetched


class TestInvalidKeyFailFast:
    def test_401_raises_fails_fast(self, tmp_path):
        """FAIL-1: a rejected API key (HTTP 401) must fail fast, not silently
        skip every movie."""
        tmdb_id = 500
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(
            api_key="rejected", cache_dir=cache_dir,
            max_retries=3, backoff_factor=0.01, rate_limit_delay=0,
        )
        session = FakeSession()
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(401))
        client._session = session

        with pytest.raises(TMDBError):
            client.get_movie(tmdb_id)
        # Must not have retried — fail fast on the first 401
        assert len(session.calls) == 1

    def test_401_message_does_not_leak_key(self, tmp_path):
        """SEC-1: the 401 error message must not include the API key."""
        tmdb_id = 501
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(
            api_key="SECRET_VALUE_42", cache_dir=cache_dir,
            max_retries=3, backoff_factor=0.01, rate_limit_delay=0,
        )
        session = FakeSession()
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(401))
        client._session = session

        with pytest.raises(TMDBError) as exc_info:
            client.get_movie(tmdb_id)
        assert "SECRET_VALUE_42" not in str(exc_info.value)


class TestSecurity:
    def test_api_key_not_in_cache(self, tmp_path):
        """SEC-1: API key is never written to cache."""
        tmdb_id = 400
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(api_key="SECRET_KEY_123", cache_dir=cache_dir, rate_limit_delay=0)
        session = FakeSession()
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(200, _make_raw_response()))
        client._session = session

        client.get_movie(tmdb_id)
        cache_file = cache_dir / f"{tmdb_id}.json"
        content = cache_file.read_text()
        assert "SECRET_KEY_123" not in content

    def test_api_key_not_in_features(self, tmp_path):
        """SEC-1: extracted features don't contain the API key."""
        tmdb_id = 401
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(api_key="SECRET_KEY_123", cache_dir=cache_dir, rate_limit_delay=0)
        session = FakeSession()
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(200, _make_raw_response()))
        client._session = session

        result = client.get_movie(tmdb_id)
        assert result is not None
        # Check no value contains the key
        for key, val in result.items():
            if isinstance(val, list):
                for item in val:
                    assert "SECRET_KEY_123" not in str(item)
            else:
                assert "SECRET_KEY_123" not in str(val)

    def test_api_key_not_logged(self, tmp_path, caplog):
        """SEC-1: API key is never logged."""
        tmdb_id = 402
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(api_key="SECRET_KEY_123", cache_dir=cache_dir, rate_limit_delay=0)
        session = FakeSession()
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(500))
        client._session = session

        with caplog.at_level(logging.DEBUG, logger="recsys.tmdb"):
            client.get_movie(tmdb_id)

        for record in caplog.records:
            assert "SECRET_KEY_123" not in record.getMessage()

    def test_api_key_sent_as_param(self, tmp_path):
        """API key is sent as query param (not in URL path)."""
        tmdb_id = 403
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        client = TMDBClient(api_key="SECRET_KEY_123", cache_dir=cache_dir, rate_limit_delay=0)
        session = FakeSession()
        session.set_response(_tmdb_url(tmdb_id), FakeResponse(200, _make_raw_response()))
        client._session = session

        client.get_movie(tmdb_id)
        call = session.calls[0]
        assert call["params"]["api_key"] == "SECRET_KEY_123"
