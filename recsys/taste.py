"""Deterministic personal taste engine (effort 2, W1 #14).

Pure closed-form arithmetic over the operator's own taste ratings — no fitted
models, no randomness, no I/O (MATH-1, MATH-3). Every score decomposes into
documented terms that recombine exactly (MATH-2). Pinned sliders override
derived values outright, never blended (MATH-4).

Spec: docs/SPEC-TASTE-WEBUI.md ("The math"). Constants: recsys.config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from recsys.config import (
    TASTE_ERA_PROX_SCALE,
    TASTE_ERA_TOL_MIN,
    TASTE_FEATURE_ERA_W,
    TASTE_FEATURE_GENRE_W,
    TASTE_FEATURE_POP_W,
    TASTE_NEUTRAL_RATING,
    TASTE_SCORE_FEATURE_W,
    TASTE_SCORE_NEIGHBOR_W,
    TASTE_SCORE_SCALE,
    TASTE_SIGNAL_SCALE,
    TASTE_SIM_ERA_W,
    TASTE_SIM_GENRE_W,
    TASTE_SIM_POP_W,
)

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
_NO_GENRES = "(no genres listed)"

# ---------------------------------------------------------------- datatypes


@dataclass(frozen=True)
class MovieFeatures:
    """Static, dataset-derived features for one movie."""

    movie_id: int
    title: str
    genres: frozenset[str]
    decade: int | None  # e.g. 1990; None when the title has no parseable year
    pop_pct: float  # popularity percentile in [0, 1] within the loaded dataset


@dataclass(frozen=True)
class TasteRating:
    """One operator taste rating (0.5–5.0 in half steps)."""

    movie_id: int
    rating: float

    @property
    def signal(self) -> float:
        """Signed signal s = (r - 3.0) / 2.0 in [-1.25, +1.0]."""
        return (self.rating - TASTE_NEUTRAL_RATING) / TASTE_SIGNAL_SCALE


@dataclass(frozen=True)
class Profile:
    """Derived taste profile (pure function of ratings + features)."""

    genre_weights: dict[str, float]  # normalized to [-1, 1]
    era_pref: float | None  # positive-signal weighted mean decade
    era_tol: float  # weighted decade spread + TASTE_ERA_TOL_MIN
    pop_pref: float | None  # positive-signal weighted mean percentile
    n_ratings: int


@dataclass(frozen=True)
class Slider:
    """One UI slider: effective value + provenance (MATH-4)."""

    value: float | None
    derived: float | None
    pinned: bool


@dataclass(frozen=True)
class EffectiveProfile:
    """Profile after pin resolution: pinned wins outright, else derived."""

    genre_weights: dict[str, float]
    era_pref: float | None
    era_tol: float
    pop_pref: float | None
    n_ratings: int
    sliders: dict[str, Slider] = field(default_factory=dict)


@dataclass(frozen=True)
class NeighborTerm:
    """One rated movie's contribution to N(m) (User Story 5 citations)."""

    movie_id: int
    title: str
    rating: float
    signal: float
    similarity: float
    contribution: float  # similarity * signal / sum(|signals|)


@dataclass(frozen=True)
class ScoreBreakdown:
    """Complete arithmetic for one recommendation (MATH-2)."""

    movie_id: int
    title: str
    genre_affinity: float
    era_match: float
    pop_match: float
    feature_match: float  # 0.5*ga + 0.25*em + 0.25*pm
    neighbor_evidence: float  # sum of neighbor contributions
    neighbor_terms: tuple[NeighborTerm, ...]
    feature_term: float  # SCALE * 0.65 * F
    neighbor_term: float  # SCALE * 0.35 * N
    score: float  # feature_term + neighbor_term
    cold_start: bool


@dataclass(frozen=True)
class SimilarityBreakdown:
    """Similar-tab entry: sim(a, b) with its terms."""

    movie_id: int
    title: str
    genre_jaccard: float
    era_proximity: float
    pop_proximity: float
    similarity: float


# ---------------------------------------------------------------- features


def _parse_decade(title: str) -> int | None:
    m = _YEAR_RE.search(title or "")
    if not m:
        return None
    return (int(m.group(1)) // 10) * 10


def _parse_genres(raw: str) -> frozenset[str]:
    if not raw or raw == _NO_GENRES:
        return frozenset()
    return frozenset(g for g in raw.split("|") if g and g != _NO_GENRES)


def build_features(
    movies_df: pd.DataFrame,
    rating_counts: dict[int, int],
    content_features: dict[int, dict] | None = None,
) -> dict[int, MovieFeatures]:
    """Build per-movie features from the movies table + dataset rating counts.

    ``pop_pct`` is a mid-rank percentile: with ``n`` movies sorted by rating
    count, ``p = (strictly_less + 0.5 * (ties - 1)) / (n - 1)`` (0.5 when
    ``n == 1``). TMDB genres from ``content_features`` are merged in when
    present (cache-only path).
    """
    rows = list(movies_df.itertuples(index=False))
    n = len(rows)
    counts = {int(r.movieId): int(rating_counts.get(int(r.movieId), 0)) for r in rows}
    sorted_counts = sorted(counts.values())

    def pct(c: int) -> float:
        if n <= 1:
            return 0.5
        less = _bisect_left(sorted_counts, c)
        ties = _bisect_right(sorted_counts, c) - less
        return (less + 0.5 * (ties - 1)) / (n - 1)

    out: dict[int, MovieFeatures] = {}
    for r in rows:
        mid = int(r.movieId)
        genres = set(_parse_genres(str(r.genres)))
        if content_features and mid in content_features:
            genres |= {str(g) for g in content_features[mid].get("genres", [])}
        out[mid] = MovieFeatures(
            movie_id=mid,
            title=str(r.title),
            genres=frozenset(genres),
            decade=_parse_decade(str(r.title)),
            pop_pct=pct(counts[mid]),
        )
    return out


def _bisect_left(a: list[int], x: int) -> int:
    from bisect import bisect_left

    return bisect_left(a, x)


def _bisect_right(a: list[int], x: int) -> int:
    from bisect import bisect_right

    return bisect_right(a, x)


# ---------------------------------------------------------------- profile


def derive_profile(
    taste: list[TasteRating], features: dict[int, MovieFeatures]
) -> Profile:
    """Closed-form profile from taste ratings (spec, "The math").

    Genre weights: raw sum of signals per genre, normalized by max |raw|.
    Era/popularity preferences: positive-signal (``max(s, 0)``) weighted
    means — liked movies define where your taste lives; dislikes shape genre
    weights and neighbor evidence instead.
    """
    raw: dict[str, float] = {}
    pos_w = era_num = era_wsum = pop_num = pop_wsum = 0.0
    era_pairs: list[tuple[float, float]] = []  # (weight, decade)

    for tr in taste:
        f = features.get(tr.movie_id)
        if f is None:
            continue
        s = tr.signal
        for g in f.genres:
            raw[g] = raw.get(g, 0.0) + s
        u = max(s, 0.0)
        if u > 0.0:
            if f.decade is not None:
                era_num += u * f.decade
                era_wsum += u
                era_pairs.append((u, float(f.decade)))
            pop_num += u * f.pop_pct
            pop_wsum += u
            pos_w += u

    max_abs = max((abs(v) for v in raw.values()), default=0.0)
    weights = {g: v / max_abs for g, v in raw.items()} if max_abs > 0 else {}

    era_pref = era_num / era_wsum if era_wsum > 0 else None
    if era_pref is not None:
        var = sum(w * (d - era_pref) ** 2 for w, d in era_pairs) / era_wsum
        era_tol = var**0.5 + TASTE_ERA_TOL_MIN
    else:
        era_tol = TASTE_ERA_TOL_MIN
    pop_pref = pop_num / pop_wsum if pop_wsum > 0 else None

    return Profile(
        genre_weights=weights,
        era_pref=era_pref,
        era_tol=era_tol,
        pop_pref=pop_pref,
        n_ratings=len(taste),
    )


def effective_profile(profile: Profile, pins: dict[str, float]) -> EffectiveProfile:
    """Apply pin semantics (MATH-4): pinned value wins outright, else derived.

    Slider ids: ``genre:<Name>``, ``era``, ``pop``. Resetting a slider is
    simply recomputing without its pin — there is no blended state.
    """
    sliders: dict[str, Slider] = {}
    genre_ids = {f"genre:{g}" for g in profile.genre_weights} | {
        k for k in pins if k.startswith("genre:")
    }

    genre_weights: dict[str, float] = {}
    for sid in sorted(genre_ids):
        g = sid.split(":", 1)[1]
        derived = profile.genre_weights.get(g)
        pinned = sid in pins
        value = pins[sid] if pinned else derived
        sliders[sid] = Slider(value=value, derived=derived, pinned=pinned)
        if value is not None:
            genre_weights[g] = value

    era_pinned = "era" in pins
    era_val = pins["era"] if era_pinned else profile.era_pref
    sliders["era"] = Slider(value=era_val, derived=profile.era_pref, pinned=era_pinned)

    pop_pinned = "pop" in pins
    pop_val = pins["pop"] if pop_pinned else profile.pop_pref
    sliders["pop"] = Slider(value=pop_val, derived=profile.pop_pref, pinned=pop_pinned)

    return EffectiveProfile(
        genre_weights=genre_weights,
        era_pref=era_val,
        era_tol=profile.era_tol,
        pop_pref=pop_val,
        n_ratings=profile.n_ratings,
        sliders=sliders,
    )


# ---------------------------------------------------------------- scoring


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _genre_affinity(f: MovieFeatures, weights: dict[str, float]) -> float:
    if not f.genres:
        return 0.0
    return sum(weights.get(g, 0.0) for g in f.genres) / len(f.genres)


def _era_match(f: MovieFeatures, era_pref: float | None, era_tol: float) -> float:
    if era_pref is None or f.decade is None:
        return 0.0
    return _clip(1.0 - abs(f.decade - era_pref) / (2.0 * era_tol))


def _pop_match(f: MovieFeatures, pop_pref: float | None) -> float:
    if pop_pref is None:
        return 0.0
    return _clip(1.0 - 2.0 * abs(f.pop_pct - pop_pref))


def similarity(a: MovieFeatures, b: MovieFeatures) -> tuple[float, dict[str, float]]:
    """sim(a, b) = 0.6*Jaccard + 0.25*era_proximity + 0.15*pop_proximity."""
    union = a.genres | b.genres
    jac = len(a.genres & b.genres) / len(union) if union else 0.0
    if a.decade is None or b.decade is None:
        era = 0.0
    else:
        era = max(0.0, 1.0 - abs(a.decade - b.decade) / TASTE_ERA_PROX_SCALE)
    pop = 1.0 - abs(a.pop_pct - b.pop_pct)
    terms = {"genre_jaccard": jac, "era_proximity": era, "pop_proximity": pop}
    sim = (
        TASTE_SIM_GENRE_W * jac + TASTE_SIM_ERA_W * era + TASTE_SIM_POP_W * pop
    )
    return sim, terms


def score_movie(
    movie_id: int,
    eff: EffectiveProfile,
    taste: list[TasteRating],
    features: dict[int, MovieFeatures],
) -> ScoreBreakdown:
    """Score one movie with its complete arithmetic (MATH-1, MATH-2)."""
    f = features[movie_id]
    ga = _genre_affinity(f, eff.genre_weights)
    em = _era_match(f, eff.era_pref, eff.era_tol)
    pm = _pop_match(f, eff.pop_pref)
    fm = (
        TASTE_FEATURE_GENRE_W * ga
        + TASTE_FEATURE_ERA_W * em
        + TASTE_FEATURE_POP_W * pm
    )

    signal_sum = sum(abs(tr.signal) for tr in taste if tr.movie_id in features)
    terms: list[NeighborTerm] = []
    if signal_sum > 0.0:
        for tr in taste:
            nf = features.get(tr.movie_id)
            if nf is None or tr.movie_id == movie_id:
                continue
            sim, _ = similarity(f, nf)
            terms.append(
                NeighborTerm(
                    movie_id=tr.movie_id,
                    title=nf.title,
                    rating=tr.rating,
                    signal=tr.signal,
                    similarity=sim,
                    contribution=sim * tr.signal / signal_sum,
                )
            )
    ne = sum(t.contribution for t in terms)

    ft = TASTE_SCORE_SCALE * TASTE_SCORE_FEATURE_W * fm
    nt = TASTE_SCORE_SCALE * TASTE_SCORE_NEIGHBOR_W * ne
    return ScoreBreakdown(
        movie_id=movie_id,
        title=f.title,
        genre_affinity=ga,
        era_match=em,
        pop_match=pm,
        feature_match=fm,
        neighbor_evidence=ne,
        neighbor_terms=tuple(
            sorted(terms, key=lambda t: (-abs(t.contribution), t.movie_id))
        ),
        feature_term=ft,
        neighbor_term=nt,
        score=ft + nt,
        cold_start=eff.n_ratings == 0,
    )


def recommend(
    features: dict[int, MovieFeatures],
    taste: list[TasteRating],
    pins: dict[str, float],
    n: int,
) -> list[ScoreBreakdown]:
    """Top-N unrated movies, deterministic order (-score, movie_id)."""
    eff = effective_profile(derive_profile(taste, features), pins)
    rated = {tr.movie_id for tr in taste}
    picks = [
        score_movie(mid, eff, taste, features)
        for mid in sorted(features)
        if mid not in rated
    ]
    picks.sort(key=lambda b: (-b.score, b.movie_id))
    return picks[:n]


def similar_movies(
    movie_id: int, features: dict[int, MovieFeatures], n: int
) -> list[SimilarityBreakdown]:
    """Movies most similar to ``movie_id``, deterministic order."""
    anchor = features[movie_id]
    out: list[SimilarityBreakdown] = []
    for mid in sorted(features):
        if mid == movie_id:
            continue
        f = features[mid]
        sim, terms = similarity(anchor, f)
        out.append(
            SimilarityBreakdown(
                movie_id=mid,
                title=f.title,
                genre_jaccard=terms["genre_jaccard"],
                era_proximity=terms["era_proximity"],
                pop_proximity=terms["pop_proximity"],
                similarity=sim,
            )
        )
    out.sort(key=lambda s: (-s.similarity, s.movie_id))
    return out[:n]
