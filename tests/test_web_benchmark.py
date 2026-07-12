"""W7 — Benchmark tab (#20): read-only legacy ML comparison table.

The web UI never computes ML metrics (MATH-3); it only renders the
comparison table that the legacy CLI experiment wrote to
``results/comparison.csv``. Missing or unreadable results answer a truthful
empty state, never a 500 (WEB-FAIL-6).

Verification-command: .venv/bin/pytest tests/test_web_benchmark.py -q
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from recsys.web.app import create_app

CSV = (
    "Approach,Precision@10,Recall@10,NDCG@10,RMSE,MAE\n"
    "popularity,0.0900,0.8000,0.5061,—,—\n"
    "svd,0.0900,0.8000,0.5131,1.0957,0.8672\n"
)


def make_client(tmp_path, *, csv: str | None) -> TestClient:
    results = tmp_path / "results"
    if csv is not None:
        results.mkdir()
        (results / "comparison.csv").write_text(csv, encoding="utf-8")
    profile = tmp_path / "taste_profile.json"
    app = create_app(
        mode="fixture", profile_path=profile, results_dir=results
    )
    return TestClient(app)


# -- happy path: table rendered verbatim, read-only ---------------------


def test_benchmark_returns_table(tmp_path):
    c = make_client(tmp_path, csv=CSV)
    r = c.get("/api/benchmark")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["columns"] == [
        "Approach", "Precision@10", "Recall@10", "NDCG@10", "RMSE", "MAE",
    ]
    assert len(body["rows"]) == 2
    assert body["rows"][0][0] == "popularity"
    # Cells pass through verbatim — including the em-dash placeholders.
    assert body["rows"][0][4] == "—"
    assert body["rows"][1][4] == "1.0957"


def test_benchmark_is_deterministic(tmp_path):
    c = make_client(tmp_path, csv=CSV)
    assert c.get("/api/benchmark").json() == c.get("/api/benchmark").json()


# -- WEB-FAIL-6: empty state is truthful, never a 500 --------------------


def test_benchmark_missing_results_is_empty_state_not_500(tmp_path):
    c = make_client(tmp_path, csv=None)
    r = c.get("/api/benchmark")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["rows"] == []
    # The empty state must tell the operator how to produce results.
    assert "recsys" in body["message"]


def test_benchmark_malformed_csv_is_empty_state_not_500(tmp_path):
    c = make_client(tmp_path, csv="")
    r = c.get("/api/benchmark")
    assert r.status_code == 200
    assert r.json()["available"] is False


def test_benchmark_non_utf8_is_empty_state_not_500(tmp_path):
    """P3-4: undecodable bytes must not escape as UnicodeDecodeError -> 500."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "comparison.csv").write_bytes(b"Approach,P\n\xff\xfe garbage,1\n")
    app = create_app(
        mode="fixture", profile_path=tmp_path / "p.json", results_dir=results
    )
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/benchmark")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert "recsys" in body["message"]


def test_benchmark_oversized_field_is_empty_state_not_500(tmp_path):
    """P3-4: a field past csv.field_size_limit raises csv.Error -> must be 200."""
    huge = "x" * 200_000  # > default csv.field_size_limit() of 131072
    results = tmp_path / "results"
    results.mkdir()
    (results / "comparison.csv").write_text(
        f"Approach,P\n{huge},1\n", encoding="utf-8"
    )
    app = create_app(
        mode="fixture", profile_path=tmp_path / "p.json", results_dir=results
    )
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/benchmark")
    assert r.status_code == 200
    assert r.json()["available"] is False


def test_benchmark_ragged_csv_is_empty_state_not_500(tmp_path):
    c = make_client(tmp_path, csv="Approach,P\nonly-one-cell\n,,,,\n")
    r = c.get("/api/benchmark")
    assert r.status_code == 200
    # Ragged rows are dropped; header-only means nothing to show.
    body = r.json()
    assert body["available"] is False or all(
        len(row) == len(body["columns"]) for row in body["rows"]
    )


# -- MATH-3 stays enforced: no ML imports pulled in by W7 ----------------


def test_no_ml_modules_loaded_by_benchmark(tmp_path):
    # Run in a subprocess: the check is on sys.modules, which other tests
    # in the suite legitimately pollute (they import recsys.recommenders).
    # A fresh interpreter proves the benchmark path itself stays ML-free.
    import subprocess
    import sys as _sys

    results = tmp_path / "results"
    results.mkdir()
    (results / "comparison.csv").write_text(CSV, encoding="utf-8")
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from fastapi.testclient import TestClient\n"
        "from recsys.web.app import create_app\n"
        f"app = create_app(mode='fixture',"
        f" profile_path=Path({str(tmp_path / 'p.json')!r}),"
        f" results_dir=Path({str(results)!r}))\n"
        "c = TestClient(app)\n"
        "assert c.get('/api/benchmark').json()['available'] is True\n"
        "bad = [m for m in sys.modules\n"
        "       if m == 'recsys.recommenders' or m.startswith('sklearn')]\n"
        "assert not bad, f'ML modules loaded: {bad}'\n"
    )
    proc = subprocess.run(
        [_sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


# -- page references the tab --------------------------------------------


def test_page_has_benchmark_tab(tmp_path):
    c = make_client(tmp_path, csv=CSV)
    html = c.get("/").text
    assert "/api/benchmark" in html


def test_default_results_dir_used_when_not_passed(tmp_path):
    # Omitting results_dir must not crash app construction (real default).
    app = create_app(mode="fixture", profile_path=tmp_path / "p.json")
    with TestClient(app) as c:
        assert c.get("/api/benchmark").status_code == 200


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
