# Backfilling missing 2025 stats (TennisAbstract profiles)

This repo’s 2025 dataset of record is:

- `ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS.csv`

If some matches are missing `winner_stats` / `loser_stats` blobs (currently a small number in Wimbledon/US Open),
use the backfill script below.

## Script

- `ta_tourney_scrape/tools/backfill_missing_profile_stats_2025.py`

It:
- finds rows where `winner_stats` or `loser_stats` are empty
- scrapes TennisAbstract player profile match logs
- aligns by `(tourney_name, round, winner_name, loser_name, score)`
- writes a new CSV with the same schema

## Run

From repo root:

```bash
python3 ta_tourney_scrape/tools/backfill_missing_profile_stats_2025.py \
  --input ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS.csv \
  --output ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS_BACKFILLED.csv \
  --year 2025
```

By default it targets only `Wimbledon` and `US Open` (conservative). Override:

```bash
python3 ta_tourney_scrape/tools/backfill_missing_profile_stats_2025.py \
  --only-tournaments "Wimbledon,US Open,Australian Open"
```

## Output

- Backfilled CSV: `..._BACKFILLED.csv`
- JSON report: `..._BACKFILLED.backfill_report.json`

## Notes

- This script relies on the existing throttled HTTP client + caching in `ta_tourney_scrape`.
- It is intentionally conservative: it doesn’t try to infer unknown stats.
