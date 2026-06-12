# Data Ingestion: Sackmann + Hermes Supplement

## Data flow

```
Jeff Sackmann tennis_atp (GitHub)          Hermes agent (VPS, TennisAbstract)
        |  authoritative                          |  gap-fill
        v                                         v
tennis_datav2/atp_matches_YYYY.csv     data/tennis_matches.sqlite (matches table)
        \                                        /
         \        scripts/refresh_match_data.py
          \                  |
           v                 v
   (validated overwrite)   tennis_datav2/atp_matches_YYYY_supplement.csv
                    \       /
                     v     v
        DataLoader.load_data()  ->  main file + supplement (deduped)
```

- **Sackmann is authoritative.** `scripts/refresh_match_data.py` downloads the
  current year's `atp_matches_YYYY.csv` (and the previous year's during
  January) and overwrites the local copy, but only if the download parses as
  CSV with the expected columns and has at least as many rows as the local
  file (guards against upstream truncation).
- **Hermes fills the gap** between Sackmann releases. Completed ATP singles
  matches in `data/tennis_matches.sqlite` dated after the Sackmann coverage
  cutoff (`max(tourney_date)` across the refreshed yearly files) are
  normalized to the Sackmann column layout and written to
  `tennis_datav2/atp_matches_{YEAR}_supplement.csv`. Supplement files are
  derived data, rebuilt wholesale on every run — never edit them by hand.
- **Loading:** `DataLoader.load_data()` appends a year's supplement file (if
  present) after the main file and drops exact duplicate
  `(winner_name, loser_name, tourney_date)` pairs, keeping the main-file row.

Run it locally with:

```
.venv/bin/python scripts/refresh_match_data.py            # full refresh
.venv/bin/python scripts/refresh_match_data.py --skip-download   # offline
```

In CI (`.github/workflows/daily-model.yml`) the script runs daily before
retraining. If the repository secret `HERMES_DB_URL` is set, the workflow
first downloads the Hermes sqlite from that URL to `data/tennis_matches.sqlite`;
if the secret is unset or the download fails, the run continues with
Sackmann data only.

## Contract for the Hermes agent

Hermes must write completed matches into the `matches` table of
`data/tennis_matches.sqlite` (schema defined in
`scripts/agent_daily_tennis_scrape.py`). Columns the refresh script depends
on:

| Column        | Required | Format / convention                                        |
|---------------|----------|------------------------------------------------------------|
| `match_key`   | yes      | Stable unique TEXT primary key (scraper uses a sha256 of tour/season/tournament/round/players/score) |
| `tour`        | yes      | Must be `'ATP'` to be ingested                             |
| `status`      | yes      | Must be `'completed'` to be ingested                       |
| `winner_name` | yes      | `"First Last"` full name (TennisAbstract style, e.g. `Nick Kyrgios`) |
| `loser_name`  | yes      | Same format as `winner_name`                               |
| `tournament`  | yes      | Tournament name, e.g. `Stuttgart`                          |
| `match_date`  | strongly recommended | ISO date `YYYY-MM-DD`. If NULL/empty, the refresh falls back to `date(first_seen_utc)` — the day the scraper first saw the result, which can lag the actual match date |
| `first_seen_utc` | yes   | ISO-8601 UTC timestamp (fallback date source)              |
| `round`       | recommended | Sackmann round codes: `R128/R64/R32/R16/QF/SF/F/RR`     |
| `score`       | recommended | Space-separated sets, e.g. `6-3 3-6 6-3`                |
| `best_of`     | optional | `3` or `5`; defaults to `3` when NULL                      |
| `surface`     | optional | `Hard/Clay/Grass/Carpet`; left NaN when NULL               |
| `season`      | optional | Calendar year INTEGER                                      |

Per-side serve stats may be written to `match_side_stats` (one row per
`(match_key, side)` with `side` in `winner`/`loser`). When populated, the
refresh maps `aces`, `double_faults`, `service_points_total`,
`first_serve_in`, `first_serve_points_won`, `second_serve_points_won`,
`break_points_saved`, `break_points_faced` into the Sackmann
`w_*`/`l_*` stat columns. NULL stats are fine — supplement rows then carry
empty stat columns, like early Sackmann rows sometimes do.

Fields the supplement cannot provide (player ids, ranks, hand, height, age,
seeds, draw size, minutes) are left empty; the model's feature pipeline
already tolerates NaN there. `tourney_level` is guessed from the tournament
name (Grand Slams -> `G`, Masters -> `M`, default `A`), and `tourney_id` is
synthesized as `{year}-SUPP-{tournament-slug}` so supplement rows are easy
to identify.

## Supersession (how supplement rows retire)

Every refresh recomputes the coverage cutoff from the (just refreshed)
Sackmann files and rebuilds supplements from scratch:

1. Sackmann publishes new matches -> `max(tourney_date)` moves forward.
2. Agent matches at or before the new cutoff are no longer selected.
3. For matches near the boundary, a fuzzy dedup drops any supplement row
   that matches a Sackmann row on normalized winner/loser **last names**
   (accents stripped, hyphens treated as spaces — handles
   `Jan-Lennard Struff` vs `Jan Lennard Struff`) with dates within 14 days.
4. A year's supplement file is deleted when a run with a readable Hermes DB
   produces no surviving rows for it. If the Hermes DB is missing the
   existing supplement files are left untouched.

So no manual cleanup is needed: once Sackmann covers a period, the
supplement rows for that period disappear on the next refresh, and the
authoritative rows (with full stats, ranks, ids) take their place.
