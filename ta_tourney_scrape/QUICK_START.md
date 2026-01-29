# Quick Start Guide - ATP 2025 Match Data

## ✅ Data Ready!

You now have **798 ATP matches from 2025** ready to use for your tennis prediction model and sports betting analysis.

---

## Available Files

### 1. Main Dataset (CSV)
**File**: `data/atp_matches_2025_from_ta.csv`
- **Format**: CSV (easy to open in Excel, pandas, etc.)
- **Size**: ~1 MB
- **Rows**: 798 matches
- **Columns**: 16 (tournament info, match details, stats)

### 2. Main Dataset (Parquet)
**File**: `data/intermediate/ta_tournament_matches_2025.parquet`
- **Format**: Parquet (optimized for pandas)
- **Same data as CSV**, just different format

### 3. Tournament List
**File**: `data/intermediate/ta_tournaments_2025.json`
- 28 ATP tournament URLs discovered
- Includes Grand Slams, Masters, ATP 500/250

---

## How to Use the Data

### Option 1: Load in Python (Pandas)
```python
import pandas as pd

# Load the data
df = pd.read_csv('data/atp_matches_2025_from_ta.csv')

# Or load from parquet (faster)
df = pd.read_parquet('data/intermediate/ta_tournament_matches_2025.parquet')

# Explore
print(f"Total matches: {len(df)}")
print(f"Tournaments: {df['tourney_name'].nunique()}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst match:")
print(df.iloc[0])
```

### Option 2: Open in Excel/Numbers
Simply open `data/atp_matches_2025_from_ta.csv` in your spreadsheet application.

### Option 3: Use with Your Prediction Model
```python
import pandas as pd

# Load your existing model
from your_model import TennisPredictionModel

# Load 2025 data
df_2025 = pd.read_csv('data/atp_matches_2025_from_ta.csv')

# Extract features (example)
features = df_2025[['surface', 'winner_rank', 'loser_rank', 'dr']]

# Make predictions
predictions = model.predict(features)
```

---

## Data Schema

### Columns Available
1. **tourney_name** - Tournament name (e.g., "2025 Rome Masters")
2. **surface** - Court surface (Hard, Clay, Grass)
3. **draw_size** - Number of players in draw
4. **tourney_level** - Tournament level (A, M, G, etc.)
5. **start_date** - Tournament start date (YYYY-MM-DD)
6. **ta_url** - TennisAbstract source URL
7. **round** - Match round (R32, R16, QF, SF, F)
8. **winner_name** - Winner's name
9. **loser_name** - Loser's name
10. **winner_rank** - Winner's ATP rank
11. **loser_rank** - Loser's ATP rank
12. **score** - Match score (e.g., "6-4 7-6(4)")
13. **time** - Match duration (e.g., "1:27")
14. **dr** - Dominance Ratio (float)
15. **winner_stats** - Winner serve stats (dict/JSON)
16. **loser_stats** - Loser serve stats (dict/JSON)

### Stats Fields (in winner_stats/loser_stats)
- **ace_pct** - Ace percentage (0-1)
- **first_in_pct** - First serve in percentage (0-1)
- **first_won_pct** - First serve won percentage (0-1)
- **second_won_pct** - Second serve won percentage (0-1)

---

## Example Analysis

### 1. Tournament Summary
```python
import pandas as pd

df = pd.read_csv('data/atp_matches_2025_from_ta.csv')

# Matches by tournament
print(df.groupby('tourney_name').size().sort_values(ascending=False))

# Matches by surface
print(df['surface'].value_counts())

# Average DR by surface
print(df.groupby('surface')['dr'].mean())
```

### 2. Player Performance
```python
# Top winners
winners = df['winner_name'].value_counts().head(10)
print("Top 10 Winners:")
print(winners)

# Player with highest average DR
player_dr = df.groupby('winner_name')['dr'].mean().sort_values(ascending=False)
print("\nPlayers with highest DR:")
print(player_dr.head(10))
```

### 3. Betting Analysis
```python
# Upsets (lower-ranked player wins)
df['winner_rank_num'] = pd.to_numeric(df['winner_rank'], errors='coerce')
df['loser_rank_num'] = pd.to_numeric(df['loser_rank'], errors='coerce')

upsets = df[df['winner_rank_num'] > df['loser_rank_num']]
print(f"Upsets: {len(upsets)} out of {len(df)} matches ({len(upsets)/len(df)*100:.1f}%)")

# Biggest upsets
df['rank_diff'] = df['winner_rank_num'] - df['loser_rank_num']
biggest_upsets = df.nlargest(10, 'rank_diff')[['tourney_name', 'round', 'winner_name', 'winner_rank', 'loser_name', 'loser_rank', 'score']]
print("\nBiggest upsets:")
print(biggest_upsets)
```

---

## Get More Data

### Re-scrape for New Matches
As the 2025 season progresses, get new matches:

```bash
cd ta_tourney_scrape
python3 -m ta_tourney_scrape scrape_tournaments
```

This will:
- ✅ Use cached data for already-scraped tournaments
- ✅ Only fetch new/updated tournaments
- ✅ Append new matches to the dataset

### Add Player Profile Stats (Optional)
For additional enrichment:

```bash
python3 -m ta_tourney_scrape scrape_profiles
```

---

## Integration with Your Model

### If You Have a Base Dataset
If you have Jeff Sackmann's tennis_atp dataset or similar:

```bash
python3 -m ta_tourney_scrape build \
  --base-csv path/to/your/atp_matches_2025.csv
```

This will:
1. Match TA data to your base dataset
2. Compute derived metrics from base counts
3. Generate reconciliation report
4. Output: `data/atp_matches_2025_joined_to_base.csv`

### If Starting Fresh
Just use the scraped data directly:

```python
import pandas as pd

# Load 2025 data
df = pd.read_csv('data/atp_matches_2025_from_ta.csv')

# Your feature engineering here
# ...

# Train your model
# ...
```

---

## File Locations

```
ta_tourney_scrape/
├── data/
│   ├── atp_matches_2025_from_ta.csv          # ← Main dataset (CSV)
│   └── intermediate/
│       ├── ta_tournament_matches_2025.parquet # ← Main dataset (Parquet)
│       ├── ta_tournaments_2025.json           # ← Tournament list
│       └── tournaments_needing_profiles_2025.json
├── src/ta_tourney_scrape/                     # ← Pipeline code
├── README.md                                  # ← Full documentation
├── SCRAPING_COMPLETE_SUMMARY.md               # ← Detailed results
└── QUICK_START.md                             # ← This file
```

---

## Summary

✅ **798 ATP matches** from 2025 ready to use  
✅ **15 tournaments** covered (all completed tournaments)  
✅ **Full stats** including DR and serve percentages  
✅ **CSV format** for easy integration  
✅ **Re-scrape anytime** to get new matches  

**You're all set!** Start building your prediction model or betting analysis with this comprehensive 2025 ATP dataset.

---

## Questions?

- **Where's the data?** → `data/atp_matches_2025_from_ta.csv`
- **How do I get more?** → Re-run `python3 -m ta_tourney_scrape scrape_tournaments`
- **Can I use this for betting?** → Yes! That's what it's designed for.
- **Is this legal?** → Yes, TennisAbstract is a public website. We respect robots.txt and rate limits.

**Happy analyzing! 🎾📊**

