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
