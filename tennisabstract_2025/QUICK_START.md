# Quick Start Guide

## ✅ What's Ready

- ✅ Column list configured (50 columns)
- ✅ All code implemented
- ✅ Sample URL added for testing

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
cd tennisabstract_2025
pip install -e .
```

### Step 2: Download Player Database

Download `atp_players.csv` from:
- https://github.com/JeffSackmann/tennis_atp

Place it in:
```
data/sackmann/atp_players.csv
```

### Step 3: Run the Pipeline

**For 2025:**
```bash
python -m ta2025 run --year 2025
```

**For 2026 (with sample URL):**
```bash
python -m ta2025 run --year 2026
```

**Or test first:**
```bash
python test_scraper.py
```

## 📊 Output

After running, you'll get:

- `data/raw/tennisabstract_2025_matches.jsonl` - Raw scraped data
- `data/processed/atp_matches_2025.csv` - Final CSV output
- `data/raw/missing_player_ids.csv` - Players that need manual mapping

## 🔧 Troubleshooting

**"Player database not found"**
→ Download `atp_players.csv` to `data/sackmann/`

**"No tournaments discovered"**
→ Check TennisAbstract website or add URLs to fallback list in `discover.py`

**"Missing player IDs"**
→ Add entries to `mappings/name_overrides.csv`

## 📝 Column List

The tool uses **50 columns** in this exact order:
- tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date
- match_num, winner_id, winner_seed, winner_entry, winner_name, winner_hand
- winner_ht, winner_ioc, winner_age, loser_id, loser_seed, loser_entry
- loser_name, loser_hand, loser_ht, loser_ioc, loser_age, score, best_of
- round, minutes, w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon
- w_SvGms, w_bpSaved, w_bpFaced, l_ace, l_df, l_svpt, l_1stIn, l_1stWon
- l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced, winner_rank, winner_rank_points
- loser_rank, loser_rank_points

All configured and ready to use!

