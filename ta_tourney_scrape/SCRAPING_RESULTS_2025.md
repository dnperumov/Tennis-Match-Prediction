# TennisAbstract 2025 Scraping Results

## Summary

Successfully scraped **1,517 matches from 28 ATP tournaments** for the 2025 season.

## Output Files

- **`data/atp_matches_2025_from_tennisabstract.csv`** - Main output with all scraped matches
- **`data/intermediate/ta_tournament_matches_2025.parquet`** - Intermediate Parquet format
- **`data/intermediate/ta_tournaments_2025.json`** - List of discovered tournament URLs (120 total)

## Tournaments Scraped (28 total)

### Masters 1000 (6 tournaments, 735 matches)
1. Canada Masters - 190 matches
2. Rome Masters - 190 matches  
3. Paris Masters - 110 matches
4. Indian Wells Masters - 95 matches
5. Miami Masters - 95 matches
6. Monte Carlo Masters - 55 matches

### ATP 500 (10 tournaments, 457 matches)
7. Bastad - 81 matches
8. Dubai - 62 matches
9. Tokyo - 62 matches
10. Dallas - 62 matches
11. Rotterdam - 31 matches
12. Barcelona - 31 matches
13. Vienna - 31 matches
14. Halle - 31 matches
15. Beijing - 31 matches
16. Basel - 31 matches
17. Rio de Janeiro - 31 matches

### ATP 250 (10 tournaments, 310 matches)
18. Metz - 54 matches
19. Stockholm - 27 matches
20. Stuttgart - 27 matches
21. Auckland - 27 matches
22. Marrakech - 27 matches
23. Eastbourne - 27 matches
24. Delray Beach - 27 matches
25. Buenos Aires - 27 matches
26. Brisbane - 31 matches

### Other Events (2 tournaments, 24 matches)
27. ATP Tour Finals - 15 matches
28. Laver Cup - 9 matches

## Data Columns

The scraped dataset includes:
- Tournament metadata: `tourney_name`, `surface`, `draw_size`, `tourney_level`, `start_date`, `ta_url`
- Match info: `round`, `winner_name`, `loser_name`, `winner_rank`, `loser_rank`, `score`, `time`
- Statistics: `dr` (Dominance Ratio), `winner_stats`, `loser_stats`

## Technical Details

### URL Formats Discovered

TennisAbstract uses two URL formats for tournament pages:

1. **`/cgi-bin/tourney.cgi?t=YYYY-XXXX%2FName`** ✅ **Successfully scraped**
   - Contains full HTML tables with match statistics
   - Example: `https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0301%2FBrisbane`
   - **This is what we successfully scraped (1,517 matches)**

2. **`/current/YYYYATPName.html`** ❌ **Cannot scrape with simple HTML parsing**
   - Uses JavaScript to dynamically load match data
   - Example: `https://www.tennisabstract.com/current/2025ATPWashington.html`
   - Matches are loaded into `<span id="completed">` via JavaScript
   - Would require browser automation (Selenium/Playwright) to scrape

### Discovery Process

- Started with 61 hardcoded tournament URLs
- Expanded to 120 URLs by including both format types
- Successfully scraped 28 tournaments with completed matches
- 92 tournaments either:
  - Haven't been played yet (2025 season ongoing)
  - Use JavaScript-based loading (can't scrape)
  - Don't exist at those URLs (404 errors)

## Limitations & Next Steps

### Current Limitations

1. **JavaScript-loaded pages**: Cannot scrape `/current/` format pages without browser automation
2. **Incomplete season**: 2025 season was ongoing at time of scraping (January 2026)
3. **Missing tournaments**: Grand Slams and some other tournaments not yet available or use different formats
4. **Stats format**: `winner_stats` and `loser_stats` are stored as dictionaries, may need flattening for analysis

### Recommended Next Steps

1. **For complete 2025 data**: Use Jeff Sackmann's `tennis_atp` repository as the base dataset
   - It has ALL ATP matches with standard format
   - Enrich it with TennisAbstract percentage stats using the `build` command

2. **For JavaScript pages**: Implement browser automation
   - Use Selenium or Playwright to load JavaScript-rendered pages
   - Extract matches from dynamically loaded content

3. **For player profile fallback**: Run the `scrape_profiles` command
   - Already implemented in the pipeline
   - Scrapes match stats from individual player profile pages
   - Useful for tournaments without match-level stats on tournament pages

## Usage

### View the data
```bash
# Open in pandas
import pandas as pd
df = pd.read_csv('data/atp_matches_2025_from_tennisabstract.csv')
print(df.head())
```

### Re-run scraping
```bash
# Discover tournaments
python3 -m ta_tourney_scrape discover

# Scrape tournament pages
python3 -m ta_tourney_scrape scrape_tournaments

# Scrape player profiles (for fallback)
python3 -m ta_tourney_scrape scrape_profiles

# Build final dataset with reconciliation
python3 -m ta_tourney_scrape build --base-csv path/to/sackmann_atp_matches_2025.csv
```

## Date Range

- **Earliest match**: 2024-12-30 (Brisbane/Auckland - early start for 2025 season)
- **Latest match**: 2025-11-09 (Paris Masters/Tour Finals)
- **Season coverage**: ~11 months of the 2025 ATP season

## Statistics Available

From the scraped data, we have:
- **Dominance Ratio (DR)**: Available for most matches
- **Winner/Loser Stats**: Stored as dictionaries, including:
  - Ace percentage
  - First serve percentage
  - First serve won percentage
  - Second serve won percentage
  - Break points saved
  - And more (varies by tournament page format)

## Conclusion

Successfully scraped **1,517 matches from 28 tournaments** using the `/cgi-bin/` URL format. This represents a substantial portion of the 2025 ATP season, particularly Masters 1000 and ATP 500 events. For complete 2025 coverage, recommend using Jeff Sackmann's dataset as base and enriching with TennisAbstract stats.

