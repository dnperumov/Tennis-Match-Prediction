# Setup Instructions

## Step 1: Install Dependencies

```bash
cd tennisabstract_2025
pip install -e .
```

Or manually:

```bash
pip install requests beautifulsoup4 pandas unidecode tqdm python-dateutil lxml
```

## Step 2: Get Required Files

### Player Database

Download `atp_players.csv` from Jeff Sackmann's repository:
- URL: https://github.com/JeffSackmann/tennis_atp
- Place it in: `data/sackmann/atp_players.csv`

The file should have at least these columns:
- `player_id`
- `name`

### Column List

**IMPORTANT**: You need to provide the exact column list for the output CSV.

**Option 1**: Edit `src/ta2025/cli.py` and set `COLUMN_LIST`:

```python
COLUMN_LIST = [
    'tourney_id',
    'tourney_name',
    'surface',
    # ... (all columns in exact order)
]
```

**Option 2**: Create a file with column names (one per line) and use `--column-list-file`:

```bash
python -m ta2025 build-csv --column-list-file COLUMN_LIST.txt
```

**Option 3**: Use the reference from an existing `atp_matches_YYYY.csv` file:

```bash
# Extract column names from existing file
head -1 data/sackmann/atp_matches_2024.csv | tr ',' '\n' > COLUMN_LIST.txt
```

## Step 3: Configure Name Overrides (Optional)

If you know some players won't match automatically, add them to `mappings/name_overrides.csv`:

```csv
original_name,player_id
"John Smith",123456
"Jane Doe",789012
```

## Step 4: Test Discovery

Test that tournament discovery works:

```python
from ta2025.discover import discover_tournaments
urls = discover_tournaments(year=2025)
print(f"Found {len(urls)} tournaments")
```

## Step 5: Run the Pipeline

Once the column list is configured:

```bash
python -m ta2025 run --year 2025
```

## Troubleshooting

### "Column list not configured" error

You must provide the column list before running `build-csv` or `run`. See Step 2 above.

### "Player database not found" warning

Make sure `data/sackmann/atp_players.csv` exists. Download it from the Jeff Sackmann repository.

### No tournaments discovered

- Check that TennisAbstract website structure hasn't changed
- Add fallback tournament URLs to `src/ta2025/discover.py` in `FALLBACK_TOURNAMENTS_2025`

### Missing player IDs

After running, check `data/raw/missing_player_ids.csv` and add entries to `mappings/name_overrides.csv` as needed.

