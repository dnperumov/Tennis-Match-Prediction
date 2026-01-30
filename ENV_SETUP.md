# Environment setup (local)

This repo contains two Python projects:

1) `Tennis-Match-Prediction/` (ML pipeline)
2) `Tennis-Match-Prediction/tennisabstract_2025/` (2025 TennisAbstract scraper)

## Requirements

On Ubuntu/Debian you typically need:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv
```

Then from the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### TennisAbstract 2025 scraper

```bash
cd tennisabstract_2025
pip install -e .
# then
python -m ta2025 run --year 2025
```

## Quick audit step

Before training, audit the produced 2025 CSV for missingness (script lives in `sasha-builds`):

```bash
python3 /path/to/sasha-builds/tools/tennis_audit/audit_atp_csv.py \
  tennisabstract_2025/data/processed/atp_matches_2025.csv
```
