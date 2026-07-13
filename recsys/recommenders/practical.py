"""Practical watch-log recommender (effort 3, W9).

Pure, unit-testable content-based scorer over the ``data/catalog.sqlite``
catalog. No fitted/ML model anywhere: every score is closed-form arithmetic
over the operator's 5-star watch log, plus one explicit, *seeded* random
sampling step used only when the "surprise" slider is above zero (so the
same profile + params always produce the same list — no wall-clock or
hidden global state).

Every recommendation carries a full breakdown (PRAC-2: ``math.components``)
and plain-English ``reasons`` — see ``Recommendation`` below and the API
contract in docs/plan/effort3-practical-watchlog.md.

This module is intentionally never imported at ``recsys.web.app`` module
scope (see the loader in that file) so the fixture/real serving path stays
exactly as import-light as before; --full mode loads it lazily instead.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from recsys.config import (
    PRACTICAL_ERA_HIGH_DECADE,
    PRACTICAL_ERA_LOW_DECADE,
    PRACTICAL_ERA_TOL_MIN,
    PRACTICAL_INDIAN_BOOST,
    PRACTICAL_RATING_SHRINK_K,
    PRACTICAL_SIGNAL_NEUTRAL,
    PRACTICAL_SIGNAL_SCALE,
    PRACTICAL_SURPRISE_DOMINANCE_PENALTY,
    PRACTICAL_SURPRISE_POOL_FRACTION,
    PRACTICAL_W_ERA,
    PRACTICAL_W_GENRE,
    PRACTICAL_W_LANGUAGE,
    PRACTICAL_W_POPULARITY,
    PRACTICAL_W_RATING,
)

# ISO-639-1 -> a human-readable label used only inside plain-English reasons
# (PRAC-3: no jargon, no codes, in the default UI copy).
_LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "ml": "Malayalam", "kn": "Kannada", "bn": "Bengali", "mr": "Marathi",
    "pa": "Punjabi", "gu": "Gujarati", "or": "Odia", "as": "Assamese",
    "fr": "French", "es": "Spanish", "ja": "Japanese", "ko": "Korean",
    "de": "German", "zh": "Chinese", "it": "Italian", "ru": "Russian",
    "xx": "an unknown language",
}


def language_label(code: str) -> str:
    """Human-readable language name for reasons text; falls back to the code."""
    if not code:
        return "an unknown language"
    return _LANGUAGE_NAMES.get(code, code.upper())


# ---------------------------------------------------------------- datatypes


@dataclass(frozen=True)
class CatalogMovie:
    """One row from the frozen catalog schema (``movies`` table)."""

    id: int
    title: str
    year: int | None
    language: str
    genres: frozenset[str]
    poster_path: str | None
    popularity: float
    vote_average: float
    vote_count: int
    is_indian: bool
    # W12/W13 (effort 4): external ratings — None when the catalog predates
    # the ratings migration (frozen effort-3 DDL) or enrichment found nothing.
    imdb_rating: float | None = None
    rt_rating: int | None = None
    age_rating: str | None = None

    @property
    def decade(self) -> int | None:
        return (self.year // 10) * 10 if self.year else None


@dataclass(frozen=True)
class WatchedMovie:
    """One watch-log entry joined to its catalog row."""

    id: int
    rating: float
    movie: CatalogMovie

    @property
    def weight(self) -> float:
        """Signed taste weight: (r - 3) / 2, in [-1.25, 1.0] for r in [0.5, 5]."""
        return (self.rating - PRACTICAL_SIGNAL_NEUTRAL) / PRACTICAL_SIGNAL_SCALE


@dataclass(frozen=True)
class SliderParams:
    """The four human-named sliders (Slider contract), all 0-100."""

    surprise: float = 50.0
    crowd: float = 50.0
    era: float = 50.0
    indian: float = 50.0


@dataclass(frozen=True)
class TasteProfile:
    """Closed-form summary of the watch log (pure function of the log)."""

    genre_weights: dict[str, float]      # weighted mean signal per genre, [-1, 1]
    language_weights: dict[str, float]   # weighted mean signal per language, [-1, 1]
    era_pref: float | None               # positive-signal weighted-mean decade
    era_tol: float
    top_genres: tuple[str, ...]          # positively-weighted genres, best first
    languages: tuple[str, ...]           # languages present in the log, most-weighted first
    n_logged: int


@dataclass(frozen=True)
class ScoreComponents:
    """The exact 5 terms frozen by the API contract's ``math.components``."""

    genre: float
    language: float
    era: float
    popularity: float
    rating_signal: float


@dataclass(frozen=True)
class Recommendation:
    movie: CatalogMovie
    reasons: tuple[str, ...]
    score: float
    components: ScoreComponents


# ------------------------------------------------------------- small helpers


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _clip01(x: float) -> float:
    return _clip(x, 0.0, 1.0)


def _fmt_stars(r: float) -> str:
    n = f"{r:g}"
    return f"{n} star" if r == 1.0 else f"{n} stars"


# ---------------------------------------------------------------- profile


def derive_taste_profile(watched: list[WatchedMovie]) -> TasteProfile:
    """Closed-form profile from the watch log alone (no candidate needed).

    Genre/language affinity is the *weighted mean* signal of the watched
    movies carrying that genre/language (ticket: "weighted mean over
    one-hot genres"), then scaled by the largest |mean| so weights land in
    [-1, 1]. Era preference is the positive-signal weighted-mean decade —
    liked movies define where taste lives; dislikes still shape genre and
    language weights.
    """
    genre_sum: dict[str, float] = {}
    genre_n: dict[str, int] = {}
    lang_sum: dict[str, float] = {}
    lang_n: dict[str, int] = {}
    lang_evidence: dict[str, float] = {}
    era_num = era_wsum = 0.0
    era_pairs: list[tuple[float, float]] = []

    for w in watched:
        m = w.movie
        wt = w.weight
        for g in m.genres:
            genre_sum[g] = genre_sum.get(g, 0.0) + wt
            genre_n[g] = genre_n.get(g, 0) + 1
        lang_sum[m.language] = lang_sum.get(m.language, 0.0) + wt
        lang_n[m.language] = lang_n.get(m.language, 0) + 1
        lang_evidence[m.language] = lang_evidence.get(m.language, 0.0) + abs(wt) + 0.01

        pos = max(wt, 0.0)
        if pos > 0.0 and m.decade is not None:
            era_num += pos * m.decade
            era_wsum += pos
            era_pairs.append((pos, float(m.decade)))

    genre_mean = {g: genre_sum[g] / genre_n[g] for g in genre_sum}
    lang_mean = {l: lang_sum[l] / lang_n[l] for l in lang_sum}

    genre_max = max((abs(v) for v in genre_mean.values()), default=0.0)
    genre_weights = (
        {g: v / genre_max for g, v in genre_mean.items()} if genre_max > 0 else {}
    )
    lang_max = max((abs(v) for v in lang_mean.values()), default=0.0)
    language_weights = (
        {l: v / lang_max for l, v in lang_mean.items()} if lang_max > 0 else {}
    )

    era_pref = era_num / era_wsum if era_wsum > 0 else None
    if era_pref is not None:
        var = sum(w * (d - era_pref) ** 2 for w, d in era_pairs) / era_wsum
        era_tol = var**0.5 + PRACTICAL_ERA_TOL_MIN
    else:
        era_tol = PRACTICAL_ERA_TOL_MIN

    top_genres = tuple(
        g for g, v in sorted(genre_weights.items(), key=lambda kv: (-kv[1], kv[0]))
        if v > 0
    )
    languages = tuple(
        l for l, _ in sorted(lang_evidence.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    return TasteProfile(
        genre_weights=genre_weights,
        language_weights=language_weights,
        era_pref=era_pref,
        era_tol=era_tol,
        top_genres=top_genres,
        languages=languages,
        n_logged=len(watched),
    )


# ---------------------------------------------------------------- popularity


def _popularity_percentiles(catalog: dict[int, CatalogMovie]) -> dict[int, float]:
    """Log-scaled ``vote_count`` percentile in [0, 1] within the catalog.

    Mid-rank percentile so ties share a value: with ``n`` movies sorted by
    ``log1p(vote_count)``, ``p = (less + 0.5*(ties-1)) / (n-1)`` (0.5 if n<=1).
    """
    items = list(catalog.values())
    logs = sorted(math.log1p(m.vote_count) for m in items)
    n = len(logs)

    def pct(v: float) -> float:
        if n <= 1:
            return 0.5
        lo = bisect_left(logs, v)
        hi = bisect_right(logs, v)
        ties = hi - lo
        return (lo + 0.5 * (ties - 1)) / (n - 1)

    return {m.id: pct(math.log1p(m.vote_count)) for m in items}


# ---------------------------------------------------------------- components


def _genre_component(m: CatalogMovie, weights: dict[str, float]) -> float:
    if not m.genres or not weights:
        return 0.5  # no evidence either way -> neutral, not a penalty
    x = sum(weights.get(g, 0.0) for g in m.genres) / len(m.genres)
    return (x + 1.0) / 2.0


def _language_component(m: CatalogMovie, weights: dict[str, float]) -> float:
    if not weights:
        return 0.5
    x = weights.get(m.language, 0.0)
    return (x + 1.0) / 2.0


def _era_slider_target(era_slider: float) -> float:
    t = era_slider / 100.0
    return PRACTICAL_ERA_LOW_DECADE + t * (PRACTICAL_ERA_HIGH_DECADE - PRACTICAL_ERA_LOW_DECADE)


def _era_component(
    m: CatalogMovie,
    profile: TasteProfile,
    era_slider: float,
    fallback_decade: float,
) -> float:
    """Smooth interpolation vs the movie's decade.

    era_slider=50 is neutral: the target is purely the log-derived
    preference (or the catalog's median decade with an empty log). Moving
    to either extreme blends in a slider-anchored target (<1990 / >=2015).
    """
    if m.decade is None:
        return 0.5
    derived = profile.era_pref if profile.era_pref is not None else fallback_decade
    blend = abs(era_slider - 50.0) / 50.0  # 0 at neutral, 1 at extremes
    target = (1.0 - blend) * derived + blend * _era_slider_target(era_slider)
    tol = max(profile.era_tol, PRACTICAL_ERA_TOL_MIN)
    # Exponential falloff, never exactly 0: a hard-clipped linear ramp made
    # every movie further than 2*tol from the target score identically 0.0,
    # so the era slider produced no gradient over old/new extremes.
    return math.exp(-abs(m.decade - target) / (2.0 * tol))


def _popularity_component(pop_pct: float, crowd_slider: float) -> float:
    """crowd=0 inverts (favor obscure), 100 favors popular, 50 is neutral."""
    s = (crowd_slider - 50.0) / 50.0  # -1 .. +1
    return _clip01(0.5 + s * (pop_pct - 0.5))


def _rating_signal_component(m: CatalogMovie) -> float:
    """vote_average/10 shrunk toward 0.5 when vote_count is low."""
    k = PRACTICAL_RATING_SHRINK_K
    shrink = m.vote_count / (m.vote_count + k)
    return shrink * (m.vote_average / 10.0) + (1.0 - shrink) * 0.5


def _indian_multiplier(m: CatalogMovie, indian_slider: float) -> float:
    """Post-hoc multiplier (not a math.component — contract freezes 5 terms)."""
    t = (indian_slider - 50.0) / 50.0  # -1 .. +1
    if m.is_indian:
        return _clip(1.0 + t * PRACTICAL_INDIAN_BOOST, 0.05, 1.0 + PRACTICAL_INDIAN_BOOST)
    return _clip(1.0 - t * PRACTICAL_INDIAN_BOOST, 0.05, 1.0 + PRACTICAL_INDIAN_BOOST)


def score_movie(
    m: CatalogMovie,
    profile: TasteProfile,
    pop_pct: float,
    sliders: SliderParams,
    fallback_decade: float,
) -> tuple[float, ScoreComponents]:
    genre = _genre_component(m, profile.genre_weights)
    language = _language_component(m, profile.language_weights)
    era = _era_component(m, profile, sliders.era, fallback_decade)
    popularity = _popularity_component(pop_pct, sliders.crowd)
    rating_signal = _rating_signal_component(m)

    raw = (
        PRACTICAL_W_GENRE * genre
        + PRACTICAL_W_LANGUAGE * language
        + PRACTICAL_W_ERA * era
        + PRACTICAL_W_POPULARITY * popularity
        + PRACTICAL_W_RATING * rating_signal
    )
    score = raw * _indian_multiplier(m, sliders.indian)
    components = ScoreComponents(
        genre=genre, language=language, era=era,
        popularity=popularity, rating_signal=rating_signal,
    )
    return score, components


# ---------------------------------------------------------------- reasons


def _anchor_reason(m: CatalogMovie, watched: list[WatchedMovie]) -> str | None:
    """"Because you gave X 5 stars" for the log's best movie sharing genre/language.

    E3-4 (PRAC-2 honesty): only positively-weighted movies (rating above
    neutral) may anchor — "Because you gave X 1 star" is never a reason to
    recommend something.
    """
    shared = [
        w for w in watched
        if w.movie.id != m.id
        and w.weight > 0.0
        and (w.movie.genres & m.genres or w.movie.language == m.language)
    ]
    if not shared:
        return None
    best = max(shared, key=lambda w: (w.rating, w.id))
    return f"Because you gave {best.movie.title} {_fmt_stars(best.rating)}"


def _component_reasons(
    m: CatalogMovie,
    profile: TasteProfile,
    components: ScoreComponents,
    sliders: SliderParams,
) -> list[str]:
    ranked = sorted(
        [
            ("genre", components.genre, PRACTICAL_W_GENRE),
            ("language", components.language, PRACTICAL_W_LANGUAGE),
            ("era", components.era, PRACTICAL_W_ERA),
            ("popularity", components.popularity, PRACTICAL_W_POPULARITY),
            ("rating_signal", components.rating_signal, PRACTICAL_W_RATING),
        ],
        key=lambda t: -(t[1] * t[2]),
    )
    reasons: list[str] = []
    for name, value, _w in ranked:
        if len(reasons) >= 2:
            break
        if name == "genre" and value > 0.5:
            shared_top = sorted(
                (g for g in m.genres if profile.genre_weights.get(g, 0.0) > 0),
                key=lambda g: -profile.genre_weights.get(g, 0.0),
            )
            if shared_top:
                reasons.append(f"Matches your love of {shared_top[0]}")
        elif name == "language" and value > 0.5 and profile.language_weights.get(m.language, 0.0) > 0:
            reasons.append(f"You gravitate toward {language_label(m.language)} films")
        elif name == "era" and value > 0.6 and m.decade is not None:
            reasons.append(f"From the {m.decade}s, an era you tend to enjoy")
        elif name == "popularity":
            if sliders.crowd <= 35:
                reasons.append("A hidden gem you probably haven't seen")
            elif sliders.crowd >= 65 and value > 0.5:
                reasons.append("A crowd-pleasing favorite")
        elif name == "rating_signal" and m.vote_count >= PRACTICAL_RATING_SHRINK_K and m.vote_average >= 7.0:
            reasons.append("Highly rated by other viewers")
    return reasons


def build_reasons(
    m: CatalogMovie,
    profile: TasteProfile,
    components: ScoreComponents,
    sliders: SliderParams,
    watched: list[WatchedMovie],
) -> list[str]:
    """Plain-English reasons (PRAC-2): anchor citation + top-2 components."""
    reasons: list[str] = []
    anchor = _anchor_reason(m, watched)
    if anchor:
        reasons.append(anchor)
    reasons.extend(_component_reasons(m, profile, components, sliders))

    seen: set[str] = set()
    deduped: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    if not deduped:
        deduped.append(f"A solid pick based on your {profile.n_logged} logged movies")
    return deduped[:3]


# ---------------------------------------------------------------- W13: match10


def match10(score: float) -> int:
    """Map a [0, 1] score to a 1..10 match badge — monotonic, clamped, and
    never 0/10 (a movie we recommend is never a "zero match")."""
    clamped = max(0.0, min(1.0, score))
    # E4-5: round half UP consistently (0.85 -> 9/10), not banker's rounding.
    return max(1, min(10, int(clamped * 10 + 0.5)))


# ------------------------------------------------------ W13: explanations

# Sentence templates rotated by movie id so adjacent cards never share
# phrasing even when their strongest signals coincide. Deterministic:
# same movie + same profile -> same sentence.
_EXPLANATION_TEMPLATES: tuple[str, ...] = (
    "{a} — {b}.",
    "{a}, and {b}.",
    "{b}. Also: {a}.",
)


def _movie_descriptor(m: CatalogMovie) -> str:
    """A movie-specific clause used as a variety fallback — mentions concrete
    attributes (genre/decade/language) so no two distinct movies collide."""
    genre = sorted(m.genres)[0].lower() if m.genres else "well-matched"
    if m.decade is not None:
        return f"a {m.decade}s {genre} pick in {language_label(m.language)}"
    return f"a {genre} pick in {language_label(m.language)}"


def build_explanation(
    m: CatalogMovie,
    profile: TasteProfile,
    components: ScoreComponents,
    sliders: SliderParams,
    watched: list[WatchedMovie],
    salt: int = 0,
) -> str:
    """One plain-English sentence per card (W13). Deterministic in its
    inputs; varied across movies via id-rotated templates plus a
    movie-specific descriptor clause.

    Attribute-coincident movies (same genre/decade/language, same strongest
    reason, ids congruent mod len(templates)) CAN collide — callers that
    render a list must dedupe by bumping ``salt`` (rotates the template) and,
    as a last resort, anchor on the movie's year (E4-1 / RAT-3)."""
    reasons = build_reasons(m, profile, components, sliders, watched)
    a = reasons[0].rstrip(".")
    b = _movie_descriptor(m).rstrip(".")
    if salt >= len(_EXPLANATION_TEMPLATES) and m.year is not None:
        # Salted past every template: disambiguate with the concrete year.
        b = f"{b} from {m.year}"
    # Lowercase the second clause's leading letter — it sits mid-sentence.
    b = b[0].lower() + b[1:] if b else b
    template = _EXPLANATION_TEMPLATES[(m.id + salt) % len(_EXPLANATION_TEMPLATES)]
    text = template.format(a=a, b=b)
    return text[0].upper() + text[1:]


def build_unique_explanations(
    picks: list,
    profile: TasteProfile,
    sliders: SliderParams,
    watched: list[WatchedMovie],
) -> list[str]:
    """Explanations for an ordered pick list with a RAT-3 uniqueness
    guarantee: deterministic, and no two cards in the same response read
    identically (E4-1). Salts rotate templates; the final salt appends the
    year, and an explicit id suffix is the provably-unique last resort."""
    seen: set[str] = set()
    out: list[str] = []
    for p in picks:
        text = ""
        for salt in range(len(_EXPLANATION_TEMPLATES) + 1):
            text = build_explanation(
                p.movie, profile, p.components, sliders, watched, salt=salt
            )
            if text not in seen:
                break
        if text in seen:  # pragma: no cover — attribute+year twins
            text = f"{text.rstrip('.')} (pick #{p.movie.id})."
        seen.add(text)
        out.append(text)
    return out


# ---------------------------------------------------------------- selection


def _seed_key(watched: list[WatchedMovie], n: int, sliders: SliderParams) -> str:
    """Deterministic seed: same log + params always -> same seed."""
    ids = tuple(sorted((w.id, w.rating) for w in watched))
    return repr((ids, n, sliders.surprise, sliders.crowd, sliders.era, sliders.indian))


def _select(
    scored: list[tuple[float, ScoreComponents, CatalogMovie]],
    n: int,
    sliders: SliderParams,
    profile: TasteProfile,
    watched: list[WatchedMovie],
) -> list[tuple[float, ScoreComponents, CatalogMovie]]:
    """surprise=0 -> pure top-score; >0 -> seeded weighted sample from the
    top decile, with genres already dominant in the log downweighted."""
    if sliders.surprise <= 0.0 or not scored:
        return scored[:n]

    k = sliders.surprise / 100.0
    pool_n = max(n, math.ceil(len(scored) * PRACTICAL_SURPRISE_POOL_FRACTION))
    pool = scored[:pool_n]
    dominant = set(profile.top_genres[:2])

    def weight_of(item: tuple[float, ScoreComponents, CatalogMovie]) -> float:
        s, _comps, m = item
        w = max(s, 1e-6)
        if dominant and (m.genres & dominant):
            w *= max(1e-6, 1.0 - k * PRACTICAL_SURPRISE_DOMINANCE_PENALTY)
        return max(w, 1e-6)

    rng = random.Random(_seed_key(watched, n, sliders))
    # Efficient weighted sampling without replacement (exponential-key trick):
    # key_i = U_i ** (1/w_i), take the n largest keys. Deterministic given rng.
    keyed = []
    for item in pool:
        u = rng.random()
        w = weight_of(item)
        key = u ** (1.0 / w)
        keyed.append((key, item))
    keyed.sort(key=lambda t: -t[0])
    picked = [item for _, item in keyed[:n]]
    picked.sort(key=lambda t: (-t[0], t[2].id))
    return picked


# ---------------------------------------------------------------- entrypoint


def recommend(
    catalog: dict[int, CatalogMovie],
    watched: list[WatchedMovie],
    n: int,
    sliders: SliderParams,
) -> list[Recommendation]:
    """Top-N unwatched movies (PRAC-1: watched ids are never candidates)."""
    watched_ids = {w.id for w in watched}
    profile = derive_taste_profile(watched)
    pop_pcts = _popularity_percentiles(catalog)
    decades = sorted(m.decade for m in catalog.values() if m.decade is not None)
    fallback_decade = float(decades[len(decades) // 2]) if decades else 2000.0

    scored: list[tuple[float, ScoreComponents, CatalogMovie]] = []
    for mid, m in catalog.items():
        if mid in watched_ids:
            continue
        s, comps = score_movie(m, profile, pop_pcts[mid], sliders, fallback_decade)
        scored.append((s, comps, m))
    scored.sort(key=lambda t: (-t[0], t[2].id))

    chosen = _select(scored, n, sliders, profile, watched)

    out: list[Recommendation] = []
    for s, comps, m in chosen:
        reasons = build_reasons(m, profile, comps, sliders, watched)
        out.append(Recommendation(movie=m, reasons=tuple(reasons), score=s, components=comps))
    return out
