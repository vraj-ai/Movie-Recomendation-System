"""W2 (#15) — JSON profile store for the web UI.

Single small file holding the user's ratings and slider pins. Rules:
- load() never raises on bad input: a corrupt/mis-shaped file is moved
  aside to ``<name>.corrupt-<UTC stamp>`` (bytes preserved, WEB-FAIL-3)
  and an empty profile is returned.
- Missing file is just an empty profile (WEB-FAIL-5).
- save() is atomic (tmp file + os.replace) and human-readable JSON.
- No ML/scipy imports here (MATH-3): this module is stdlib-only.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class StoredProfile:
    ratings: dict[int, float] = field(default_factory=dict)
    pins: dict[str, float] = field(default_factory=dict)
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
            ratings, pins = self._validate(data)
        except (ValueError, TypeError, KeyError):
            return StoredProfile(quarantined_to=self._quarantine())
        return StoredProfile(ratings=ratings, pins=pins)

    def save(self, ratings: dict[int, float], pins: dict[str, float]) -> None:
        payload = {
            "ratings": {str(int(k)): float(v) for k, v in ratings.items()},
            "pins": {str(k): float(v) for k, v in pins.items()},
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
    def _validate(data: object) -> tuple[dict[int, float], dict[str, float]]:
        if not isinstance(data, dict):
            raise ValueError("profile root must be an object")
        raw_ratings = data.get("ratings", {})
        raw_pins = data.get("pins", {})
        if not isinstance(raw_ratings, dict) or not isinstance(raw_pins, dict):
            raise ValueError("ratings/pins must be objects")
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
        return ratings, pins

    def _quarantine(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        target = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
        os.replace(self.path, target)
        return target
