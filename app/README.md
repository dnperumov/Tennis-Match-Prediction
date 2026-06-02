# Tennis Betting Research Dashboard

Research-only tooling for model monitoring and historical betting-edge analysis. This does **not** place bets or connect to sportsbooks.

## Run the daily research model/backtest

```bash
.venv/bin/python scripts/betting_research_pipeline.py --years 2021 2022 2023 2024 2025 --test-year 2025
```

Outputs:

- `data/betting_research/latest_metrics.json`
- `data/betting_research/latest_backtest_predictions.csv`
- `models/betting_research/latest_model.pkl`

The backtest uses tennis-data.co.uk historical ATP odds and restricts features to pre-match information: rank, points, surface, series, round, rolling form, surface form, and prior H2H.

## Run the Streamlit app

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.headless true --server.port 8501
```

Tabs:

- **Data**: DB/export coverage and stat completeness.
- **Model**: latest metrics/model artifact status.
- **Edge backtest**: flat-stake threshold tests vs Bet365 closing odds.
- **Rows**: current enriched 2026 match export.

## Daily automation

Hermes cron jobs in this environment:

- `agent-led daily ATP match/stat scrape`: daily scrape/update job.
- `daily tennis model retrain and betting research backtest`: runs after scrape to retrain and refresh backtest outputs.

## Current result warning

The first research backtest did **not** find a profitable flat-stake edge. The market odds beat the current model on log loss and Brier score. Treat this as a baseline to improve, not a betting system.
