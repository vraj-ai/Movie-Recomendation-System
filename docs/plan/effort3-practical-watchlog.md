# Effort 3 — Practical Watch Log (India-scale catalog + polished UI)

Date: 2026-07-12 · Chain: /part1 → /part2 (3 subagents) → /part3 (review agent)

## User requirements (verbatim intent)
1. UI must look really good and polished; currently "there is no art" → real poster art.
2. All Indian movies in the catalog.
3. Practical flow: "I put all the movies I have watched and then it runs to give me a
   recommendation of my next watch or next 10."
4. "I don't want to see the math, only when I want to see it" → **"Show the math" button**.
5. Full ~9.7k MovieLens catalog (ml-latest-small) loaded.
6. Sliders renamed + explained "for what a normal person would see and do".
7. Decisions from waypoint gate: **5-star ratings** for logging; **dark cinematic** theme.

## Network reality (measured 2026-07-12)
- BLOCKED (connection timeout): api.themoviedb.org, files.grouplens.org.
- REACHABLE: *.github.com / raw.githubusercontent.com, huggingface.co (+ us.aws.cdn.hf.co
  redirect target), **image.tmdb.org** (poster CDN!), upload.wikimedia.org, m.media-amazon.com.
- Therefore: datasets come from GitHub mirror + HF dump; posters hotlink image.tmdb.org;
  the TMDB *API* is never called (CAT-1).

## Data sources (verified)
- MovieLens ml-latest-small mirror: `https://raw.githubusercontent.com/smanihwr/ml-latest-small/master/{movies,ratings,links,tags}.csv` (movies.csv returns 200; 9,742 movies).
- TMDB metadata dump: HF dataset `wykonos/movies`,
  `https://huggingface.co/datasets/wykonos/movies/resolve/main/movies_dataset.csv`
  (302 → us.aws.cdn.hf.co; ~722k rows; has id, title, genres, original_language, overview,
  popularity, release_date, vote_average, vote_count, poster_path).
  Fallback if CDN blocked: paginate `https://datasets-server.huggingface.co/rows?dataset=wykonos%2Fmovies&config=default&split=train&offset=N&length=100`.

## Catalog contract (SQLite: `data/catalog.sqlite`) — FROZEN, all agents build to this
```sql
CREATE TABLE movies (
  id INTEGER PRIMARY KEY,          -- stable internal id
  tmdb_id INTEGER UNIQUE,          -- may be NULL for ML rows with no TMDB match
  ml_id INTEGER UNIQUE,            -- MovieLens movieId, NULL for Indian-only rows
  title TEXT NOT NULL,
  year INTEGER,                    -- NULL allowed
  language TEXT NOT NULL,          -- ISO-639-1 original_language, 'xx' if unknown
  genres TEXT NOT NULL DEFAULT '', -- pipe-separated, e.g. 'Drama|Thriller'
  overview TEXT NOT NULL DEFAULT '',
  poster_path TEXT,                -- e.g. '/abc.jpg'; NULL → UI renders styled fallback
  popularity REAL NOT NULL DEFAULT 0,
  vote_average REAL NOT NULL DEFAULT 0,
  vote_count INTEGER NOT NULL DEFAULT 0,
  is_indian INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL CHECK (source IN ('ml','indian','both'))
);
CREATE VIRTUAL TABLE movies_fts USING fts5(title, content='movies', content_rowid='id');
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);  -- counts, build stamp
```
- Indian languages: `{hi, ta, te, ml, kn, bn, mr, pa, gu, or, as}` → is_indian=1.
- Indian rows quality floor: title AND release_date present; keep even 0-vote rows (user
  said ALL Indian movies). Dedupe by tmdb_id; a movie in both sets → source='both'.
- Poster URL construction (frontend): `https://image.tmdb.org/t/p/w342{poster_path}`.
- Expected counts (verify in build report): ml rows = 9,742; indian rows ≥ 8,000.

## API contract (mode `--full`) — FROZEN
- `GET /api/search?q=&limit=20` → `{results:[{id,title,year,language,genres:[...],poster_url,watched:bool,rating:float|null}]}` (FTS prefix match, ranked by vote_count desc).
- `GET/POST /api/watched`; POST body `{id:int, rating:float}` rating ∈ {0.5..5.0 step 0.5}
  → `{count:int}`; `DELETE /api/watched/{id}`. GET → `{items:[{id,title,year,poster_url,rating,added_at}],count}`. Persisted via existing profile store (atomic tmp+replace, quarantine rules unchanged).
- `GET /api/next?n=10&surprise=50&crowd=50&era=50&indian=50` (all 0–100, default 50) →
  `{items:[{id,title,year,language,genres,poster_url,reasons:[str],math:{score:float,components:{genre,language,era,popularity,rating_signal}}}], profile:{top_genres:[...],languages:[...],movies_logged:int}}`
  - PRAC-1: never returns a watched movie.
  - PRAC-2: `reasons` = plain English ("Because you gave Dangal 5 stars", "Matches your
    love of thrillers"); `math` always present for the Show-the-math button.
  - Requires ≥ 3 logged movies, else 409 `{detail:"log_more", needed:int}`.
- Existing fixture/real modes keep working; old endpoints stay (Advanced/benchmark).

## Slider contract (names a normal person understands) — FROZEN
| param | UI name | left label | right label | helper text |
|---|---|---|---|---|
| surprise | Comfort zone | Stick to my taste | Surprise me | "Left: more like the movies you already love. Right: stretch into new genres and styles." |
| crowd | Fame | Hidden gems | Blockbusters | "Left: lesser-known films worth digging up. Right: crowd-pleasers everyone's heard of." |
| era | Era | Older classics | New releases | "Which decades your picks lean toward." |
| indian | Cinema mix | Global | Indian | "Balance between international films and Indian cinema (Hindi, Tamil, Telugu, Malayalam & more)." |

Every slider shows its helper text underneath — no jargon, no math symbols anywhere
in the default UI (PRAC-3).

## Invariants (add to docs/INVARIANTS.md)
- CAT-1: catalog build never calls api.themoviedb.org; reproducible offline from cached downloads in `cache/downloads/`.
- CAT-2: every row has non-empty title + language; is_indian ⇔ language ∈ Indian set.
- PRAC-1: watched movies never appear in /api/next results.
- PRAC-2: every recommendation carries reasons[] (plain English) AND math{}.
- PRAC-3: default UI renders no raw scores/vectors; math only after explicit "Show the math".
- PRAC-4: missing/failed posters render a styled fallback card, never a broken image.
- WEB-SEC-1..3, WEB-FAIL-1..6 remain binding (bind 127.0.0.1, explicit data mode, etc.).

## Tickets
- **W8 (Agent A — data)**: `scripts/build_catalog.py` + downloads + catalog.sqlite +
  `fixtures/catalog-small.sqlite` (25 rows: 15 ml / 10 indian, deterministic) + build report
  in `data/catalog-report.json` + tests `tests/test_catalog_build.py` (schema, dedupe,
  indian-language mapping, fixture integrity).
- **W9 (Agent B — backend)**: `--full` mode in `recsys/web/__main__.py`, catalog-backed
  search/watched/next endpoints in `recsys/web/app.py`, recommender
  `recsys/recommenders/practical.py` (content-based: genre affinity from 5-star-weighted
  watch log + language/era/popularity terms + surprise-controlled diversity), tests
  `tests/test_practical_api.py` against fixtures/catalog-small.sqlite.
- **W10 (Agent C — frontend)**: full redesign of `recsys/web/static/index.html`
  (dark cinematic, poster-forward, onboarding search→star→log flow, hero next watch +
  next-10 grid, human sliders w/ helpers, per-card "Why this?" + "Show the math" button,
  Advanced link to legacy tabs), update UI-coupled tests.
- **W11 (Agent D — /part3)**: fresh-eyes review of W8–W10, fix findings, full suite green,
  review handoff `docs/handoffs/2026-07-12-effort3-review.md`.

## Sequencing & ownership (no file overlaps)
- A owns: scripts/, data/, fixtures/catalog-small.sqlite, tests/test_catalog_build.py.
- B owns: recsys/web/app.py, __main__.py, profile_store.py, recsys/recommenders/practical.py, tests/test_practical_api.py.
- C owns: recsys/web/static/*, UI-coupled existing tests.
- A ∥ B ∥ C (contracts frozen above); agents do NOT commit; integrator commits after
  suite is green; then D reviews; then push + restart server in --full mode.
