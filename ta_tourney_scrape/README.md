# TennisAbstract ATP 2025 Scraper - Hybrid Pipeline

Production-ready Python ETL tool to scrape ATP 2025 completed matches from TennisAbstract and enrich with percentage statistics.

## Problem Statement

TennisAbstract tournament pages are **inconsistent**:
- Some tournaments (e.g., Laver Cup) show detailed per-match stat tables with A%, DF%, 1stIn%, 1st%, 2nd%, BPSvd, DR, etc.
- Other tournaments (e.g., Adelaide, Hong Kong) only list completed matches (round + players + score) with **NO stats**

## Solution: Hybrid Scraping Strategy

This pipeline uses a **two-phase approach**:

1. **Tournament Pages (Primary)**: Authoritative source for match listings and metadata
   - Scrape all completed matches (main draw + qualifying)
   - **Detect** if page has stat tables
   - If stats present: extract them directly
   - If stats absent: flag for profile enrichment

2. **Player Profiles (Fallback)**: Targeted enrichment for matches lacking stats
   - Only scrape profiles for tournaments that need it
   - Extract match-level stats from player match logs
   - Deduplicate (same match appears on both winner and loser profiles)
   - Merge winner-side and loser-side serve stats

## Architecture

```
discover → scrape_tournaments → scrape_profiles → build
   ↓              ↓                    ↓             ↓
tournaments  tournament_matches   profile_matches  final_dataset
   .json         .parquet            .parquet      + report.md
```

## Installation

```bash
cd ta_tourney_scrape
pip install -e .
```

## Usage

### Full Pipeline

```bash
python -m ta_tourney_scrape all \
  --year 2025 \
  --base-csv ../tennis_datav2/atp_matches_2024.csv \
  --output-dir data/intermediate
```

### Step-by-Step

```bash
# 1. Discover tournament URLs
python -m ta_tourney_scrape discover --year 2025

# 2. Scrape tournament pages
python -m ta_tourney_scrape scrape_tournaments --year 2025

# 3. Enrich via player profiles (only for tournaments lacking stats)
python -m ta_tourney_scrape scrape_profiles --year 2025

# 4. Build final dataset with joins and reconciliation
python -m ta_tourney_scrape build \
  --year 2025 \
  --base-csv path/to/base_dataset.csv
```

## Outputs

### Intermediate Files (data/intermediate/)

1. **`ta_tournaments_2025.json`**
   - List of discovered tournament URLs
   - Includes seed list of 28 known ATP tournaments

2. **`ta_tournament_matches_2025.parquet`**
   - All matches scraped from tournament pages
   - Includes flag `has_stats` (True/False)
   - Includes flag `tourney_has_match_stats` at tournament level

3. **`tournaments_needing_profiles_2025.json`**
   - List of tournaments that need profile fallback enrichment

4. **`ta_profile_fallback_hits_2025.parquet`**
   - Matches with stats scraped from player profiles
   - Deduplicated (winner + loser profiles merged)

### Final Outputs (data/)

1. **`atp_matches_2025_enriched.csv`**
   - Union of all 2025 matches with TA stats (where available)
   - Columns: tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date, round, winner_name, loser_name, score, plus TA percent metrics

2. **`atp_matches_2025_joined_to_base.csv`**
   - Your base dataset (left join preserved)
   - TA percent columns appended where matched
   - Computed metrics from base counts for comparison

3. **`ta_reconciliation_report_2025.md`**
   - Coverage by tournament
   - Which tournaments required profile fallback
   - Metric agreement (TA % vs computed % from base counts)
   - Mismatch analysis

## Modules

- **`discover.py`**: Tournament URL discovery (index scraping + seed list + base dataset search)
- **`parse_tourney.py`**: Tournament page parsing with stat table detection
- **`parse_profile.py`**: Player profile scraping for fallback enrichment
- **`dedupe.py`**: Merge winner/loser profile rows, deduplicate
- **`normalize.py`**: Player names, scores, tournament keys normalization
- **`match.py`**: Join TA data to base dataset with fuzzy matching
- **`metrics.py`**: Compute derived metrics from base count stats
- **`export.py`**: Write final CSVs and reconciliation report
- **`http.py`**: HTTP session with caching, rate limiting, retries
- **`cli.py`**: Command-line interface

## Polite Scraping

- **Rate limit**: ≤ 1 req/sec (configurable)
- **Caching**: `requests-cache` with SQLite backend (7-day TTL)
- **Retries**: Exponential backoff, max 3 attempts
- **User-Agent**: Custom UA string
- **robots.txt**: Checked (with fallback if unavailable)

## Key Features

### Stat Table Detection

Automatically detects if a tournament page has per-match stat tables by looking for column headers like:
- A%, DF%, 1stIn, 1st%, 2nd%, BPSvd, DR, SPW, RPW, TPW, Time

If ≥3 stat indicators found → parse stats directly  
If not found → flag for profile enrichment

### Robust Matching

Matches are joined to base dataset using:
1. **Primary key**: (year, tournament_norm, round, winner_norm, loser_norm, score_norm, date)
2. **Relaxed matching**: Allow score variations, missing dates
3. **Join quality tracking**: exact / relaxed_score / fuzzy

### Deduplication

Player profiles show the same match twice (winner's view + loser's view):
- **Winner profile** → winner-side serve stats (A%, DF%, 1stIn%, 1st%, 2nd%, BPSvd)
- **Loser profile** → loser-side serve stats
- **Merge strategy**: Keep one record per match with both sides' stats

## Seed List

The pipeline includes a seed list of 28 known 2025 ATP tournaments:
- Grand Slams (Australian Open, Roland Garros, Wimbledon, US Open)
- Masters 1000 (Indian Wells, Miami, Monte Carlo, Madrid, Rome)
- ATP 500/250 events
- Laver Cup

This ensures the pipeline works even if TennisAbstract's index pages are unavailable.

## Rate Limiting & 429 Errors

TennisAbstract may return `429 Too Many Requests` if scraping too aggressively.

**Solutions**:
1. Increase rate limit interval in `http.py` (e.g., 2 seconds instead of 1)
2. Run scraper in batches (scrape 10 tournaments, wait, scrape next 10)
3. Use cached data (re-runs won't re-download)

## Example: Laver Cup 2025

Tournament page: https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-9210%2FLaver-Cup

**Has stat table**: ✅ Yes  
**Matches scraped**: 9  
**Stats extracted**: DR, A%, 1stIn%, 1st%, 2nd%, BPSvd for both winner and loser

Sample match:
```
Winner: Casper Ruud
Loser: Reilly Opelka
Score: 6-4 7-6(4)
DR: 1.77
Winner stats: ace_pct=1.77%, first_in_pct=9.4%, first_won_pct=79.7%, second_won_pct=88.2%
Loser stats: first_in_pct=13.8%, first_won_pct=67.7%, second_won_pct=81.8%
```

## Example: Hong Kong 2026 (Forecast-Only)

Tournament page: https://www.tennisabstract.com/current/2026ATPHongKong.html

**Has stat table**: ❌ No (only forecasts and upcoming matches)  
**Completed matches**: Listed in text format (R32: Winner d. Loser Score)  
**Stats source**: Would need player profile fallback

## Testing

```bash
# Test discovery
python -m ta_tourney_scrape discover --year 2025

# Test single tournament scraping
python -c "
from ta_tourney_scrape.parse_tourney import scrape_tournament
metadata, matches = scrape_tournament('https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-9210%2FLaver-Cup')
print(f'Tournament: {metadata[\"tourney_name\"]}')
print(f'Has stats: {metadata[\"tourney_has_match_stats\"]}')
print(f'Matches: {len(matches)}')
"

# Test profile scraping
python -c "
from ta_tourney_scrape.parse_profile import scrape_player_profile_matches
matches = scrape_player_profile_matches('Novak Djokovic', 2025)
print(f'Found {len(matches)} matches for Djokovic in 2025')
"
```

## Troubleshooting

### SSL Certificate Errors

If you see `[SSL: CERTIFICATE_VERIFY_FAILED]`, install certificates:

```bash
# macOS
/Applications/Python\ 3.9/Install\ Certificates.command

# Or set environment variable
export PYTHONHTTPSVERIFY=0  # Not recommended for production
```

### No Tournaments Discovered

- Check internet connection
- Verify TennisAbstract is accessible
- Seed list will be used as fallback (28 tournaments)

### Rate Limiting (429 Errors)

- Increase `_rate_limit_interval` in `http.py`
- Use cached data for re-runs
- Scrape in smaller batches

### Missing Stats

- Check if tournament page has stat tables (use `detect_stat_table()`)
- If not, run `scrape_profiles` command
- Some tournaments may not have stats available yet

## Dependencies

- `requests>=2.32.3` - HTTP client
- `beautifulsoup4>=4.12.3` - HTML parsing
- `pandas>=2.2.2` - Data manipulation
- `unidecode>=1.3.8` - Name normalization
- `tqdm>=4.66.2` - Progress bars
- `python-dateutil>=2.9.0` - Date parsing
- `requests-cache>=1.2.1` - HTTP caching
- `tenacity>=8.2.3` - Retries
- `lxml>=5.3.2` - Fast HTML parsing
- `pyarrow>=22.0.0` - Parquet support

## License

MIT

## Author

Built for ATP match prediction and sports betting analysis.
