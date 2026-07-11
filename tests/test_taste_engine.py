"""W1 — taste engine core (#14).

Hand-computed fixture: 4 movies, 2 taste ratings.

Movies (counts -> popularity percentile with distinct counts: less/(n-1)):
    M1 "Alpha (1995)"  {Action, Sci-Fi}  count=100 -> p=1.0
    M2 "Beta (1999)"   {Action}          count=50  -> p=1/3
    M3 "Gamma (1972)"  {Drama}           count=10  -> p=0.0
    M4 "Delta (2005)"  {Action, Drama}   count=80  -> p=2/3

Taste ratings: M1=5.0 (s=+1.0), M3=1.0 (s=-1.0).

Derived profile:
    genre raw: Action=+1, Sci-Fi=+1, Drama=-1 -> max|raw|=1 -> weights unchanged
    era_pref (positive-signal weighted): 1990; era_tol = 0 + 10 = 10
    pop_pref: p(M1) = 1.0

Score for M2:
    genre_affinity = w[Action] = 1.0
    era_match = clip(1 - |1990-1990|/(2*10), -1, 1) = 1.0
    pop_match = 1 - 2*|1/3 - 1| = -1/3
    F = 0.5*1 + 0.25*1 + 0.25*(-1/3) = 2/3
    sim(M2,M1) = 0.6*(1/2) + 0.25*1 + 0.15*(1/3) = 0.6
    sim(M2,M3) = 0.6*0 + 0.25*(1/3) + 0.15*(2/3) = 11/60
    N = (0.6*1 + (11/60)*(-1)) / (|1|+|-1|) = (0.6 - 0.18333...)/2 = 0.2083333...
    S = 100*(0.65*F + 0.35*N) = 50.625
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from recsys.taste import (
    TasteRating,
    build_features,
    derive_profile,
    effective_profile,
    recommend,
    score_movie,
    similar_movies,
    similarity,
)

APPROX = dict(rel=1e-9, abs=1e-12)


@pytest.fixture
def movies_df():
    return pd.DataFrame(
        [
            {"movieId": 1, "title": "Alpha (1995)", "genres": "Action|Sci-Fi"},
            {"movieId": 2, "title": "Beta (1999)", "genres": "Action"},
            {"movieId": 3, "title": "Gamma (1972)", "genres": "Drama"},
            {"movieId": 4, "title": "Delta (2005)", "genres": "Action|Drama"},
        ]
    )


@pytest.fixture
def counts():
    return {1: 100, 2: 50, 3: 10, 4: 80}


@pytest.fixture
def features(movies_df, counts):
    return build_features(movies_df, counts)


@pytest.fixture
def taste():
    return [TasteRating(movie_id=1, rating=5.0), TasteRating(movie_id=3, rating=1.0)]


# ---------------------------------------------------------------- features


def test_build_features_decade_genres_percentile(features):
    m1, m3 = features[1], features[3]
    assert m1.decade == 1990
    assert m3.decade == 1970
    assert m1.genres == frozenset({"Action", "Sci-Fi"})
    assert m1.pop_pct == pytest.approx(1.0, **APPROX)
    assert features[2].pop_pct == pytest.approx(1 / 3, **APPROX)
    assert m3.pop_pct == pytest.approx(0.0, **APPROX)


def test_build_features_unparseable_year_and_no_genres():
    df = pd.DataFrame(
        [{"movieId": 9, "title": "Untitled", "genres": "(no genres listed)"}]
    )
    f = build_features(df, {})[9]
    assert f.decade is None
    assert f.genres == frozenset()
    assert 0.0 <= f.pop_pct <= 1.0


def test_percentile_ties_split():
    df = pd.DataFrame(
        [
            {"movieId": 1, "title": "A (2000)", "genres": "Action"},
            {"movieId": 2, "title": "B (2000)", "genres": "Action"},
        ]
    )
    f = build_features(df, {1: 5, 2: 5})
    assert f[1].pop_pct == pytest.approx(0.5, **APPROX)
    assert f[2].pop_pct == pytest.approx(0.5, **APPROX)


def test_tmdb_genres_merged(movies_df, counts):
    content = {1: {"genres": ["Action", "Thriller"]}}
    f = build_features(movies_df, counts, content_features=content)
    assert f[1].genres == frozenset({"Action", "Sci-Fi", "Thriller"})


# ---------------------------------------------------------------- profile


def test_derived_profile_hand_values(features, taste):
    p = derive_profile(taste, features)
    assert p.genre_weights["Action"] == pytest.approx(1.0, **APPROX)
    assert p.genre_weights["Sci-Fi"] == pytest.approx(1.0, **APPROX)
    assert p.genre_weights["Drama"] == pytest.approx(-1.0, **APPROX)
    assert p.era_pref == pytest.approx(1990.0, **APPROX)
    assert p.era_tol == pytest.approx(10.0, **APPROX)
    assert p.pop_pref == pytest.approx(1.0, **APPROX)
    assert p.n_ratings == 2


def test_empty_profile_cold_start(features):
    """WEB-FAIL-5: zero ratings -> neutral profile that still serves."""
    p = derive_profile([], features)
    assert p.genre_weights == {}
    assert p.era_pref is None
    assert p.pop_pref is None
    picks = recommend(features, [], pins={}, n=4)
    assert len(picks) == 4
    assert all(b.cold_start for b in picks)
    assert all(b.score == pytest.approx(0.0, **APPROX) for b in picks)


# ---------------------------------------------------------------- scoring


def test_score_breakdown_hand_computed(features, taste):
    b = score_movie(2, effective_profile(derive_profile(taste, features), {}), taste, features)
    assert b.genre_affinity == pytest.approx(1.0, **APPROX)
    assert b.era_match == pytest.approx(1.0, **APPROX)
    assert b.pop_match == pytest.approx(-1 / 3, **APPROX)
    assert b.feature_match == pytest.approx(2 / 3, **APPROX)
    assert b.neighbor_evidence == pytest.approx(0.625 / 3, **APPROX)
    assert b.score == pytest.approx(50.625, **APPROX)


def test_breakdown_recombines_math2(features, taste):
    """MATH-2: every exposed term recombines to the totals."""
    eff = effective_profile(derive_profile(taste, features), {})
    for mid in (2, 4):
        b = score_movie(mid, eff, taste, features)
        assert b.feature_match == pytest.approx(
            0.5 * b.genre_affinity + 0.25 * b.era_match + 0.25 * b.pop_match, **APPROX
        )
        assert b.neighbor_evidence == pytest.approx(
            sum(t.contribution for t in b.neighbor_terms), **APPROX
        )
        assert b.score == pytest.approx(b.feature_term + b.neighbor_term, **APPROX)
        assert b.score == pytest.approx(
            100.0 * (0.65 * b.feature_match + 0.35 * b.neighbor_evidence), **APPROX
        )
        for t in b.neighbor_terms:
            assert t.contribution == pytest.approx(
                t.similarity * t.signal / sum(abs((r.rating - 3.0) / 2.0) for r in taste),
                **APPROX,
            )


def test_similarity_hand_computed(features):
    sim, terms = similarity(features[2], features[1])
    assert terms["genre_jaccard"] == pytest.approx(0.5, **APPROX)
    assert terms["era_proximity"] == pytest.approx(1.0, **APPROX)
    assert terms["pop_proximity"] == pytest.approx(1 / 3, **APPROX)
    assert sim == pytest.approx(0.6, **APPROX)
    assert sim == pytest.approx(
        0.6 * terms["genre_jaccard"]
        + 0.25 * terms["era_proximity"]
        + 0.15 * terms["pop_proximity"],
        **APPROX,
    )


# ---------------------------------------------------------------- recommend


def test_recommend_excludes_rated_and_ranks(features, taste):
    picks = recommend(features, taste, pins={}, n=10)
    ids = [b.movie_id for b in picks]
    assert set(ids) == {2, 4}
    assert ids == [2, 4]  # S(M2)=50.625 > S(M4)
    assert picks[0].score >= picks[1].score


def test_determinism_math1(features, taste):
    """MATH-1: identical inputs -> identical scores and order."""
    a = recommend(features, taste, pins={}, n=10)
    b = recommend(features, taste, pins={}, n=10)
    assert [(x.movie_id, x.score) for x in a] == [(y.movie_id, y.score) for y in b]


def test_tiebreak_by_movie_id():
    df = pd.DataFrame(
        [
            {"movieId": 7, "title": "Twin A (2000)", "genres": "Action"},
            {"movieId": 5, "title": "Twin B (2000)", "genres": "Action"},
        ]
    )
    f = build_features(df, {5: 3, 7: 3})
    picks = recommend(f, [], pins={}, n=2)
    assert [b.movie_id for b in picks] == [5, 7]


def test_similar_movies_excludes_self(features):
    sims = similar_movies(1, features, n=10)
    assert 1 not in [s.movie_id for s in sims]
    assert len(sims) == 3
    assert sims == sorted(sims, key=lambda s: (-s.similarity, s.movie_id))


# ---------------------------------------------------------------- pins


def test_pins_math4(features, taste):
    """MATH-4: pinned wins outright; unpinned = derived; reset restores."""
    prof = derive_profile(taste, features)
    eff = effective_profile(prof, {"genre:Action": -1.0, "pop": 0.0})
    assert eff.genre_weights["Action"] == pytest.approx(-1.0, **APPROX)  # pinned
    assert eff.genre_weights["Sci-Fi"] == pytest.approx(1.0, **APPROX)  # derived
    assert eff.pop_pref == pytest.approx(0.0, **APPROX)
    assert eff.sliders["genre:Action"].pinned is True
    assert eff.sliders["genre:Action"].derived == pytest.approx(1.0, **APPROX)
    assert eff.sliders["pop"].pinned is True
    # reset == recompute without the pin
    back = effective_profile(prof, {})
    assert back.genre_weights["Action"] == pytest.approx(1.0, **APPROX)
    assert back.sliders["genre:Action"].pinned is False


def test_pin_changes_scores(features, taste):
    base = {b.movie_id: b.score for b in recommend(features, taste, pins={}, n=10)}
    flipped = {
        b.movie_id: b.score
        for b in recommend(features, taste, pins={"genre:Action": -1.0}, n=10)
    }
    assert flipped[2] < base[2]


def test_pins_enable_cold_start_scoring(features):
    """WEB-FAIL-5: pinned weights produce nonzero scores with zero ratings."""
    picks = recommend(features, [], pins={"genre:Drama": 1.0}, n=4)
    scores = {b.movie_id: b.score for b in picks}
    assert scores[3] > scores[2]


# ---------------------------------------------------------------- MATH-3


def test_no_ml_imports_math3():
    """MATH-3: importing the engine pulls in no fitted-model code."""
    code = (
        "import sys; import recsys.taste; "
        "bad = [m for m in sys.modules if m.startswith('recsys.recommenders') "
        "or m.startswith('sklearn')]; "
        "assert not bad, bad"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
    )
    assert proc.returncode == 0, proc.stderr
