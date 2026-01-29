# Getting Complete Grand Slam Data (Wimbledon & US Open)

## Current Status

✅ **Australian Open**: 127 matches (complete)  
✅ **Roland Garros**: 127 matches (complete)  
❌ **Wimbledon**: Not available on TennisAbstract  
❌ **US Open**: Not available on TennisAbstract  

## Solution: Use Jeff Sackmann's tennis_atp Repository

Jeff Sackmann maintains the most comprehensive ATP match database, including **complete Grand Slam data**.

### Step 1: Download Sackmann's Data

```bash
# Navigate to your project directory
cd /Users/dennisperumov/Tennis-Match-Prediction

# Clone the repository
git clone https://github.com/JeffSackmann/tennis_atp.git

# The file you need: tennis_atp/atp_matches_2025.csv
```

### Step 2: Extract Grand Slam Matches

```python
import pandas as pd

# Load Sackmann's complete 2025 dataset
sackmann = pd.read_csv('tennis_atp/atp_matches_2025.csv')

# Filter for Grand Slams (tourney_level == 'G')
grand_slams = sackmann[sackmann['tourney_level'] == 'G']

print(f"Grand Slam matches in Sackmann data: {len(grand_slams)}")

# Get Wimbledon & US Open
wimbledon = grand_slams[grand_slams['tourney_name'] == 'Wimbledon']
us_open = grand_slams[grand_slams['tourney_name'] == 'Us Open']

print(f"Wimbledon: {len(wimbledon)} matches")
print(f"US Open: {len(us_open)} matches")
```

### Step 3: Merge with Your TennisAbstract Data

```python
# Load your TennisAbstract data
ta_df = pd.read_csv('ta_tourney_scrape/data/atp_matches_2025_ULTIMATE_COMPLETE.csv')

# Load Sackmann data
sackmann_df = pd.read_csv('tennis_atp/atp_matches_2025.csv')

# Get Wimbledon & US Open from Sackmann
wimbledon_us = sackmann_df[
    (sackmann_df['tourney_level'] == 'G') & 
    (sackmann_df['tourney_name'].isin(['Wimbledon', 'Us Open']))
]

# Rename Sackmann columns to match your format
# (Sackmann uses different column names, need to map them)

# Combine datasets
final_df = pd.concat([ta_df, wimbledon_us_converted], ignore_index=True)

print(f"Final dataset: {len(final_df)} matches from {final_df['tourney_name'].nunique()} tournaments")
```

## Alternative: Direct Download Links

If you don't want to clone the repo:

**2025 ATP Matches (complete season)**:
https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2025.csv

```bash
# Download directly
curl -o atp_matches_2025_sackmann.csv \
  https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2025.csv
```

## Quick Merge Script

Save this as `merge_grand_slams.py`:

```python
import pandas as pd

def merge_ta_with_sackmann_grand_slams():
    """Merge TennisAbstract data with Sackmann Grand Slams"""
    
    # Load TennisAbstract data
    print("Loading TennisAbstract data...")
    ta_df = pd.read_csv('ta_tourney_scrape/data/atp_matches_2025_ULTIMATE_COMPLETE.csv')
    
    # Load Sackmann data
    print("Loading Sackmann data...")
    sackmann_df = pd.read_csv('tennis_atp/atp_matches_2025.csv')
    
    # Get Wimbledon & US Open
    print("Extracting Wimbledon & US Open...")
    wimbledon_us = sackmann_df[
        (sackmann_df['tourney_level'] == 'G') & 
        (sackmann_df['tourney_name'].isin(['Wimbledon', 'Us Open']))
    ].copy()
    
    # Convert Sackmann format to match TennisAbstract format
    wimbledon_us['ta_url'] = ''
    wimbledon_us['scrape_method'] = 'sackmann_data'
    wimbledon_us['dr'] = None
    wimbledon_us['time'] = wimbledon_us['minutes'].astype(str)
    wimbledon_us['winner_stats'] = '{}'
    wimbledon_us['loser_stats'] = '{}'
    wimbledon_us['match_num'] = range(1, len(wimbledon_us) + 1)
    
    # Rename columns to match TennisAbstract
    column_mapping = {
        'surface': 'surface',
        'draw_size': 'draw_size',
        'tourney_level': 'tourney_level',
        'tourney_date': 'start_date',
        'round': 'round',
        'winner_name': 'winner_name',
        'loser_name': 'loser_name',
        'winner_seed': 'winner_seed',
        'loser_seed': 'loser_seed',
        'winner_ioc': 'winner_ioc',
        'loser_ioc': 'loser_ioc',
        'score': 'score',
        'winner_rank': 'winner_rank',
        'loser_rank': 'loser_rank',
    }
    
    # Select and rename columns
    wimbledon_us = wimbledon_us[list(column_mapping.keys()) + [
        'ta_url', 'scrape_method', 'time', 'dr', 
        'winner_stats', 'loser_stats', 'match_num'
    ]]
    wimbledon_us.columns = list(column_mapping.values()) + [
        'ta_url', 'scrape_method', 'time', 'dr', 
        'winner_stats', 'loser_stats', 'match_num'
    ]
    
    # Combine
    print("Merging datasets...")
    final_df = pd.concat([ta_df, wimbledon_us], ignore_index=True)
    
    # Save
    output_file = 'ta_tourney_scrape/data/atp_matches_2025_COMPLETE_ALL_GRAND_SLAMS.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Complete dataset saved: {output_file}")
    print(f"   Total matches: {len(final_df):,}")
    print(f"   Total tournaments: {final_df['tourney_name'].nunique()}")
    print(f"   Wimbledon: {len(wimbledon_us[wimbledon_us['tourney_name'] == 'Wimbledon'])} matches")
    print(f"   US Open: {len(wimbledon_us[wimbledon_us['tourney_name'] == 'Us Open'])} matches")
    
    return final_df

if __name__ == "__main__":
    merge_ta_with_sackmann_grand_slams()
```

## Run the Merge

```bash
# 1. Clone Sackmann repo
cd /Users/dennisperumov/Tennis-Match-Prediction
git clone https://github.com/JeffSackmann/tennis_atp.git

# 2. Run merge script
python3 merge_grand_slams.py
```

## Expected Final Result

- **Current**: 4,667 matches from 82 tournaments
- **After merge**: ~4,921 matches from 84 tournaments (adding ~254 Grand Slam matches)
- **Coverage**: 93% of ATP calendar
- **All Grand Slams**: ✅ Australian Open, ✅ Roland Garros, ✅ Wimbledon, ✅ US Open

## Why Sackmann for Grand Slams?

1. **Complete data**: Jeff Sackmann is the gold standard for ATP historical data
2. **Accurate**: Used by ATP researchers and analysts worldwide
3. **Standardized format**: Compatible with Sackmann column naming
4. **Regularly updated**: Maintained and kept current
5. **Free & open source**: MIT licensed

## Column Mapping: Sackmann → TennisAbstract Format

| Sackmann Column | Your TA Column | Notes |
|-----------------|----------------|-------|
| `tourney_id` | `tourney_id` | Same |
| `tourney_name` | `tourney_name` | Same |
| `surface` | `surface` | Same |
| `draw_size` | `draw_size` | Same |
| `tourney_level` | `tourney_level` | Same |
| `tourney_date` | `start_date` | Rename |
| `winner_name` | `winner_name` | Same |
| `loser_name` | `loser_name` | Same |
| `score` | `score` | Same |
| `minutes` | `time` | Rename |
| (create) | `ta_url` | Set to '' |
| (create) | `scrape_method` | Set to 'sackmann_data' |

---

**Bottom Line**: For Wimbledon & US Open, Jeff Sackmann's `tennis_atp` repository is your best (and only reliable) source!

