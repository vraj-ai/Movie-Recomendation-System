"""W2 (#15) — JSON profile store for the web UI.

Single small file holding the user's ratings and slider pins. Rules:
- load() never raises on bad input: a corrupt/mis-shaped file is moved
  aside to ``<name>.corrupt-<UTC stamp>`` (bytes preserved, WEB-FAIL-3)
  and an empty profile is returned.
- Missing file is just an empty profile (WEB-FAIL-5).
- save() is atomic (tmp file + os.replace) and human-readable JSON.
- No ML/scipy imports here (MATH-3): this module is stdlib-only.

Effort 3 (W9) extends the schema additively with a ``watched`` map for the
--full mode 5-star watch log: ``{id: {rating, added_at}}``. Existing
``ratings``/``pins`` callers (the effort-2 taste engine) are unaffected —
``watched`` defaults to ``{}`` when absent, old files load unchanged.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class WatchedEntry:
    """One --full mode watch-log entry: a 5-star rating + when it was logged."""

    rating: float
    added_at: str  # ISO 8601 (UTC)


@dataclass
class StoredProfile:
    ratings: dict[int, float] = field(default_factory=dict)
    pins: dict[str, float] = field(default_factory=dict)
    watched: dict[int, WatchedEntry] = field(default_factory=dict)
    quarantined_to: Path | None = None


class ProfileStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # -- public API ---------------------------------------------------

    def load(self) -> StoredProfile:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return StoredProfile()
        try:
            data = json.loads(raw)
            ratings, pins, watched = self._validate(data)
        except (ValueError, TypeError, KeyError):
            return StoredProfile(quarantined_to=self._quarantine())
        return StoredProfile(ratings=ratings, pins=pins, watched=watched)

    def save(
        self,
        ratings: dict[int, float],
        pins: dict[str, float],
        watched: dict[int, WatchedEntry] | None = None,
    ) -> None:
        payload = {
            "ratings": {str(int(k)): float(v) for k, v in ratings.items()},
            "pins": {str(k): float(v) for k, v in pins.items()},
            "watched": {
                str(int(k)): {"rating": float(v.rating), "added_at": str(v.added_at)}
                for k, v in (watched or {}).items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=self.path.parent, prefix=self.path.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # -- internals ----------------------------------------------------

    @staticmethod
    def _validate(
        data: object,
    ) -> tuple[dict[int, float], dict[str, float], dict[int, WatchedEntry]]:
        if not isinstance(data, dict):
            raise ValueError("profile root must be an object")
        raw_ratings = data.get("ratings", {})
        raw_pins = data.get("pins", {})
        raw_watched = data.get("watched", {})
        if not isinstance(raw_ratings, dict) or not isinstance(raw_pins, dict):
            raise ValueError("ratings/pins must be objects")
        if not isinstance(raw_watched, dict):
            raise ValueError("watched must be an object")
        ratings: dict[int, float] = {}
        for k, v in raw_ratings.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"rating for {k!r} is not numeric")
            ratings[int(k)] = float(v)
        pins: dict[str, float] = {}
        for k, v in raw_pins.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"pin {k!r} is not numeric")
            pins[str(k)] = float(v)
        watched: dict[int, WatchedEntry] = {}
        for k, v in raw_watched.items():
            if not isinstance(v, dict):
                raise ValueError(f"watched entry for {k!r} must be an object")
            rating = v.get("rating")
            added_at = v.get("added_at")
            if isinstance(rating, bool) or not isinstance(rating, (int, float)):
                raise ValueError(f"watched rating for {k!r} is not numeric")
            if not isinstance(added_at, str) or not added_at:
                raise ValueError(f"watched added_at for {k!r} is not a string")
            watched[int(k)] = WatchedEntry(rating=float(rating), added_at=added_at)
        return ratings, pins, watched

    def _quarantine(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        target = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
        os.replace(self.path, target)
        return target
