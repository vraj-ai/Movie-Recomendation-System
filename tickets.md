# Tickets — Movie Recommendation System

Local mirror of the GitHub issue tracker
(`VrajGupta/Movie-Recomendation-System`). The **GitHub issues are the source of
truth**; this file is a readable snapshot. Spec: [issue #1](https://github.com/VrajGupta/Movie-Recomendation-System/issues/1).

Work top-to-bottom by the dependency edges: a ticket is on the **frontier** when
all of its "Blocked by" tickets are closed.

| Issue | Slice | Type | Blocked by | Verification-command |
|-------|-------|------|-----------|----------------------|
| [#7](https://github.com/VrajGupta/Movie-Recomendation-System/issues/7) | 1 — Walking skeleton + popularity baseline | AFK | — (frontier) | `pytest tests/test_data.py tests/test_split.py tests/test_evaluate.py tests/test_popularity.py -q && python -m recsys.cli run --fixture --out results/` |
| [#2](https://github.com/VrajGupta/Movie-Recomendation-System/issues/2) | 2 — TMDB client + enrichment | AFK | #7 | `pytest tests/test_tmdb_client.py tests/test_enrichment.py -q` |
| [#3](https://github.com/VrajGupta/Movie-Recomendation-System/issues/3) | 3 — Content-based recommender | AFK | #7, #2 | `pytest tests/test_content.py -q && python -m recsys.cli run --fixture` |
| [#4](https://github.com/VrajGupta/Movie-Recomendation-System/issues/4) | 4 — SVD recommender + RMSE/MAE | AFK | #7 | `pytest tests/test_svd.py tests/test_evaluate_rmse.py -q` |
| [#5](https://github.com/VrajGupta/Movie-Recomendation-System/issues/5) | 5 — Hybrid recommender (full 4-way table) | AFK | #3, #4 | `pytest tests/test_hybrid.py -q && python -m recsys.cli run --fixture` |
| [#6](https://github.com/VrajGupta/Movie-Recomendation-System/issues/6) | 6 — `recommend` CLI command | AFK | #7 | `pytest tests/test_cli_recommend.py -q && python -m recsys.cli recommend --user 1 -n 10` |

## Dependency graph

```
#7 (skeleton) ──┬──> #2 (TMDB) ──> #3 (content) ──┐
                ├──> #4 (SVD) ────────────────────┴──> #5 (hybrid)
                └──> #6 (recommend)
```

Frontier after #7 lands: **#2, #4, #6 can be worked in parallel**; then #3 (after
#2); then #5 (after #3 + #4).

> Note: `Blocked by` edges are recorded as text references in each issue body.
> GitHub's native issue-dependency links can be added in the UI if desired; the
> text edges above are the authoritative dependency order.

## Red-team follow-up tickets (part3)

Filed by the part3 maker pass. `gh` is flaky on this project; if a ticket below
has no linked GitHub issue it still needs creating with label `ready-for-agent`.
Each ticket names the violated invariant and a runnable Verification-command that
exits 0 exactly when fixed.

| ID | Issue | Title | Invariant | Verification-command | Status |
|----|-------|-------|-----------|----------------------|--------|
| P3-1 | [#9](https://github.com/VrajGupta/Movie-Recomendation-System/issues/9) | Pinned gate `.venv/bin/pytest -q` cannot import `recsys` (no path/config); suite only collects via `python -m pytest` | Test harness / gate integrity | `.venv/bin/pytest -q` | FIXED |
| P3-2 | [#10](https://github.com/VrajGupta/Movie-Recomendation-System/issues/10) | SEC-1 leak: FAIL-2 `RequestException` branch in `tmdb._fetch_with_retry` logs the raw exception, which embeds `?api_key=<KEY>` from the request URL | SEC-1 (key never logged) | `.venv/bin/pytest tests/test_tmdb_client.py -q` | FIXED |
| P3-3 | [#11](https://github.com/VrajGupta/Movie-Recomendation-System/issues/11) | `split.holdout_split` crashes with cryptic `ValueError: No objects to concatenate` when every user has <2 ratings (non-empty df, empty test_parts) | Splitter contract (users <2 ratings → all train) | `.venv/bin/pytest tests/test_split.py -q` | FIXED |

### P3-1 — gate cannot import `recsys`
`.venv/bin/pytest -q` fails collection with `ModuleNotFoundError: No module
named 'recsys'`. Root cause: repo root is not on `sys.path` under the `pytest`
console script (no `pyproject.toml`/`pytest.ini` `pythonpath`, no root conftest,
no editable install). `.venv/bin/python -m pytest -q` passes 145 because `-m`
adds cwd. Fix (in-scope, `tests/**`): prepend the repo root to `sys.path` at the
top of `tests/conftest.py` so the pinned gate and per-area gates all collect.
- Verification: `.venv/bin/pytest -q` exits 0.

### P3-2 — SEC-1 API key leaked into logs on network exceptions
`_fetch_with_retry`'s `except requests.exceptions.RequestException as exc:`
branch logs `exc`. Confirmed empirically that `requests` puts the full URL —
including the `api_key` query parameter — into the exception string, so the key
is written to WARNING logs. The pre-existing `test_api_key_not_logged` only
covers the HTTP-500 path, which never logs the exception. Fix: log the exception
*class name* only (`type(exc).__name__`), never the raw message. New test
reproduces the real `requests`-style exception string and asserts the key is
absent from all log records.
- Verification: `.venv/bin/pytest tests/test_tmdb_client.py -q` exits 0.

### P3-3 — splitter crashes on all-singleton-user datasets
`holdout_split` guards the fully-empty DataFrame (part2) but not the case where
the df is non-empty yet every user has <2 ratings: all groups go to `train_parts`
and `test_parts` stays empty, so `pd.concat(test_parts)` raises the cryptic
`ValueError: No objects to concatenate`. Per the docstring contract these users
"keep all ratings in train", so the correct result is train=all, test=empty. Fix:
return an empty test DataFrame (same columns/dtypes) when `test_parts` is empty.
- Verification: `.venv/bin/pytest tests/test_split.py -q` exits 0.

## Follow-ups deliberately NOT fixed in this pass (documented, low-value / risky)

- **FAIL-3 unbounded `Retry-After`**: a hostile/broken server can return a huge
  `Retry-After` and the client will `time.sleep` for it. Cap to a sane maximum.
  (Known part2 open edge case.)
- **FAIL-7 cache value-type validation**: `_load_cache` checks keys exist but not
  that values have the right types (e.g. `genres` as a string). Wrong-typed but
  structurally-present entries are served silently. (Known part2 open edge case.)
- **Zero-relevant-item users in metric averages**: `evaluate_recommender` averages
  Precision/Recall/NDCG over *all* test users, including users with no held-out
  relevant items. This uniformly deflates every model, so the relative ranking
  (the deliverable) is not distorted; low value. (Known part2 open edge case.)
- **Duplicate `(userId, movieId)` rows summed in CSR** on load. (Known part2 edge.)
- **No linter configured**: consider adding `ruff`/`flake8` in CI. (Known gap.)

---

# Tickets — Effort 2: Taste Engine + Web UI

Local mirror of the GitHub issue tracker (`VrajGupta/Movie-Recomendation-System`).
The **GitHub issues are the source of truth**; this file is a readable snapshot.
Spec anchor: [issue #13](https://github.com/VrajGupta/Movie-Recomendation-System/issues/13)
· `docs/SPEC-TASTE-WEBUI.md` · branch `plan/taste-engine-webui`.
Invariants: `docs/INVARIANTS.md` effort-2 section (MATH-1..4, WEB-PERF-1..3,
WEB-FAIL-1..6, WEB-SEC-1..3). "Blocked by" tracks GitHub issue numbers.

- [ ] W1 — Taste engine core (deterministic profile + scoring + breakdown)
      — [#14](https://github.com/VrajGupta/Movie-Recomendation-System/issues/14)
      · frontier · MATH-1..4, WEB-FAIL-5
      · Verify: `.venv/bin/pytest tests/test_taste_engine.py -q`
- [ ] W2 — Profile store (ratings + pins JSON, quarantine, gitignore)
      — [#15](https://github.com/VrajGupta/Movie-Recomendation-System/issues/15)
      · frontier · WEB-FAIL-3, WEB-FAIL-5, WEB-SEC-3
      · Verify: `.venv/bin/pytest tests/test_profile_store.py -q`
- [ ] W3 — Server skeleton (explicit data mode, status badge API, static shell, search)
      — [#16](https://github.com/VrajGupta/Movie-Recomendation-System/issues/16)
      · frontier · WEB-FAIL-1..2, WEB-SEC-1..2, WEB-PERF-1
      · Verify: `.venv/bin/pytest tests/test_web_app.py -q`
- [ ] W4 — Recommend + Similar APIs (breakdowns wired to engine)
      — [#17](https://github.com/VrajGupta/Movie-Recomendation-System/issues/17)
      · blocked by #14, #15, #16 · MATH-1..4, WEB-FAIL-2/5, WEB-PERF-2
      · Verify: `.venv/bin/pytest tests/test_web_recommend.py -q`
- [ ] W5 — Ratings + profile APIs (mutations recompute, pin semantics)
      — [#18](https://github.com/VrajGupta/Movie-Recomendation-System/issues/18)
      · blocked by #14, #15, #16 · MATH-4, WEB-FAIL-2/3, WEB-PERF-3
      · Verify: `.venv/bin/pytest tests/test_web_profile.py -q`
- [ ] W6 — Web UI: Recommend / Similar / Profile tabs (static page)
      — [#19](https://github.com/VrajGupta/Movie-Recomendation-System/issues/19)
      · blocked by #17, #18 · WEB-FAIL-1/4, WEB-SEC-2
      · Verify: `.venv/bin/pytest tests/test_web_page.py -q`
- [ ] W7 — Benchmark tab + docs + release polish
      — [#20](https://github.com/VrajGupta/Movie-Recomendation-System/issues/20)
      · blocked by #17, #18, #19 · WEB-FAIL-6, MATH-3, WEB-PERF-1
      · Verify: `.venv/bin/pytest -q && .venv/bin/ruff check .`

## Red-team follow-up tickets (part3 — effort-2 web UI pass)

Filed by the part3 maker pass over W1–W7. `gh` remains flaky on this project; if
a ticket below has no linked GitHub issue it still needs creating with label
`ready-for-agent`. Each ticket names the violated invariant and a runnable
Verification-command that exits 0 exactly when fixed.

| ID | Issue | Title | Invariant | Verification-command | Status |
|----|-------|-------|-----------|----------------------|--------|
| P3-4 | _pending (`gh` flaky)_ | `GET /api/benchmark` 500s on non-UTF-8 or oversized-field `comparison.csv`: only `OSError` is caught, so `UnicodeDecodeError` / `csv.Error` escape | WEB-FAIL-6 (malformed results → friendly empty state, never an error page) | `.venv/bin/pytest tests/test_web_benchmark.py -q` | FIXED |
| P3-5 | _pending (`gh` flaky)_ | Quarantine warning never reaches the operator: the `quarantined` status flag is non-sticky (any non-status request consumes it) and `index.html` never renders it | WEB-FAIL-3 ("the UI shows a visible warning") | `.venv/bin/pytest tests/test_web_profile.py -q` | FIXED |
| P3-6 | _pending (`gh` flaky)_ | `python -m recsys.web` ignores `config.WEB_HOST`/`WEB_PORT`: hardcodes port 8765 while config documents 8977, leaving the documented constants dead | Config-as-documented (`config.py` "Web server" block; WEB-SEC-1 comment anchors to these constants) | `.venv/bin/pytest tests/test_web_app.py -q` | FIXED |

### P3-4 — WEB-FAIL-6: malformed comparison.csv can still 500
`get_benchmark` wraps the read in `except OSError:` only. Confirmed empirically
on Python 3.13: a `comparison.csv` with invalid UTF-8 bytes raises
`UnicodeDecodeError` → HTTP 500; a field larger than `csv.field_size_limit()`
(131072) raises `csv.Error` → HTTP 500. WEB-FAIL-6 promises a truthful empty
state for *malformed* results, never an error page. Fix: catch
`(OSError, UnicodeDecodeError, csv.Error)`.
- Verification: `.venv/bin/pytest tests/test_web_benchmark.py -q` exits 0.

### P3-5 — WEB-FAIL-3: quarantine is invisible to the operator
Two halves, confirmed empirically: (a) `store.load()` quarantines on first read
and the flag lives only on that one `StoredProfile` — if the first post-corruption
request is anything but `/api/status` (e.g. `/api/profile`), the subsequent
status call reports `quarantined: False`; (b) even when the flag is true,
`index.html`'s `loadStatus()` renders no warning — the string `quarantined` does
not appear in the page. Fix: make the flag sticky for the server's lifetime
(any load that quarantines latches app-level state reported by `/api/status`)
and wire a visible banner in the UI.
- Verification: `.venv/bin/pytest tests/test_web_profile.py -q` exits 0.

### P3-6 — dead config: WEB_HOST/WEB_PORT never used
`recsys/web/__main__.py` hardcodes `default="127.0.0.1"` and `default=8765`;
`config.py` declares `WEB_HOST = "127.0.0.1"` (annotated as the WEB-SEC-1
anchor) and `WEB_PORT = 8977`. An operator tuning the documented constants sees
no effect, and the two defaults have already drifted (8765 ≠ 8977). Fix: the
parser defaults come from config.
- Verification: `.venv/bin/pytest tests/test_web_app.py -q` exits 0.

### Follow-ups deliberately NOT fixed in this pass (documented)

- **WEB-PERF-2 at real scale (risk, untested)**: `recommend()` scores every
  unrated movie per request in pure Python — O(n_movies × n_taste_ratings)
  similarity calls. Fine on the 10-movie fixture; at ~9.7k movies with a large
  profile the <1 s budget is at risk. Needs profiling on the real dataset and
  likely vectorization — refactor territory, left to the checker/next slice.
- **No timing tests for WEB-PERF-2/WEB-PERF-3**: only WEB-PERF-1 (startup <5 s)
  has a test; latency assertions are flaky in CI and were deliberately skipped.
- **Similar-tab weight labels hardcoded in `index.html`** (`×0.6 / ×0.25 /
  ×0.15`) duplicate `TASTE_SIM_*` config constants — drift risk if the operator
  tunes config. Clean fix needs a small config endpoint; refactor, not a bug fix.
- **Cold-start era slider renders `0.00`**: with zero ratings `era`/`pop` slider
  values are JSON `null`; `sliderRow` shows `0.00 (derived 0.00)` instead of a
  dash. Cosmetic, JS-only, no browser test harness in the repo.
- **`put_pin` accepts junk-but-harmless genre ids** (`"genre: "` whitespace,
  genres absent from the dataset): validated for range only; the pin creates an
  inert slider. Low value.
- **`ProfileStore._quarantine` can itself raise `OSError`** (unwritable parent
  dir, or the corrupt file vanishing between read and rename) and would surface
  as a 500. Narrow race/permission edge; left open.
