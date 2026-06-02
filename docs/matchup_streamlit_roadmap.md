# Matchup Streamlit roadmap

Generated: 2026-06-02

Research-only. The app and model artifacts are for tennis match-outcome prediction and model-quality research. They do not place bets, execute trades, or connect to sportsbook/order endpoints.

## Built in this pass

### Real schedule/date overlay infrastructure

Added `scripts/apply_schedule_overlay.py` so reliable official/manual schedule CSVs can correct collapsed fallback dates without rewriting the export schema.

Supported overlay matching:

- exact `match_key`, or
- fuzzy tournament/round/player-pair matching

The script writes:

- corrected `match_date`
- `schedule_source`
- `schedule_date_quality = real_schedule_overlay`
- JSON report under `data/reports/schedule_overlay_report.json`
- backup of the previous export unless `--no-backup` is passed

Example:

```bash
.venv/bin/python scripts/apply_schedule_overlay.py data/schedules/roland_garros_schedule.csv
```

### Live feature/routing layer

Added `src/tennis_ml/predict/live_features.py` and wired it into the matchup engine. It now builds a no-lookahead feature snapshot with:

- schedule/date quality audit
- current-tournament fatigue/load
- rolling serve/return/total-points profiles
- overall Elo
- surface Elo
- BO5/Slam Elo
- entry context parsing (`Q`, `WC`, `LL`, `PR`)
- calibrated route selection

### Matchup prediction engine MVP

Added `src/tennis_ml/predict/matchup_engine.py` with a stable public API:

```python
predict_matchup(player1, player2, context=None, export_path=None, metrics_path=None, advanced_report_path=None, odds=None)
```

The engine returns:

- player win probabilities
- pick and confidence label
- model source
- optional no-vig market probability from decimal odds
- model edge versus market when odds are supplied
- reason breakdown
- risk flags
- caveats/data-coverage notes
- player comparison summaries
- artifact metadata

The current model source is `transparent_current_export_profile_v1`: a deterministic, auditable fallback based on the current enriched export. It uses current-export player summaries such as overall win rate, surface win rate, recent form, total-points-won profile, return-points-won profile, and first-serve-won profile. It conservatively shrinks probabilities toward the market when market odds are supplied, otherwise toward 50%.

### Streamlit matchup page

Extended `app/streamlit_app.py` with a new first tab: **Matchup Breakdown**.

The tab includes:

- player dropdowns from the current export
- tournament/surface/round/best-of-five context controls
- optional decimal odds inputs
- probability cards
- pick/confidence display
- model-vs-market edge display
- reason breakdown
- risk flags
- data caveats
- player comparison table
- raw payload expander for auditability

Existing Data, Model, Edge backtest, Niche research, and Rows tabs remain in place.

### Tests

Added `tests/test_matchup_engine.py` covering:

- probability bounds and p1+p2 sum
- no-vig market edge from decimal odds
- missing-data caveats instead of crashes
- deterministic confidence labels

## Current limitation

This is an MVP product layer, not the final high-accuracy model. It exposes the matchup workflow and gives Dennis a usable dashboard shell, while keeping the model transparent and conservative until stronger live prematch features are available.

## Next accuracy-improvement path

### 1. Better current match dates and schedules

Roland Garros/current-tournament rows currently can collapse to weak date granularity depending on fallback source. The next data improvement is reliable per-match date/time and schedule ingestion so current-tournament fatigue and no-lookahead live predictions are meaningful.

### 2. Serve/return rolling stats

Add no-lookahead rolling features for:

- hold percentage
- break percentage
- first-serve points won
- second-serve points won
- return points won
- break points created/saved
- opponent-adjusted serve/return profiles

These should update only after completed matches and should not use same-match stats for prematch prediction.

### 3. Clay/surface Elo and BO5/Slam features

Build stronger player-strength features:

- overall Elo
- surface Elo
- clay Elo
- tournament-tier Elo
- Grand Slam/best-of-five Elo
- recent-form weighted Elo
- Slam round/stage experience

The Grand Slam benchmark is already the strongest segment, so Slam-specific routing should be prioritized.

### 4. Entry context

Fill the currently neutral placeholder feature hooks:

- qualifier
- wildcard
- lucky loser
- protected ranking
- Challenger/lower-tour form
- ranking momentum

These are especially important for early rounds and ATP250-type weak clusters.

### 5. Current tournament fatigue

Add current-event state before matchup inference:

- sets played
- games played
- minutes played
- five-setters
- rest days
- walkovers/retirements
- travel/country/continent switches

This should feed both the model and the Streamlit explanation layer.

### 6. Calibration/reliability work

Continue the conservative no-lookahead recalibration path:

- fit calibration only on historical out-of-sample/train-period predictions
- test on held-out years/events
- report log loss, Brier, accuracy, and reliability bins
- specifically monitor the known 0.4-0.6 underprediction bucket
- reject calibrators that improve one bucket but worsen global probability quality

## Product roadmap for the Streamlit app

1. Keep **Matchup Breakdown** as the front door.
2. Add a **Player Profile** tab for surface/form/Slam/serve-return trends.
3. Add a **Head-to-Head** tab with same-surface H2H and similar-opponent context.
4. Add model diagnostic visuals for accuracy/log loss/Brier/calibration by year, surface, round, and tournament.
5. Add a current-tournament tracker tab that compares pre-match picks to completed results with caveats.
6. Keep the research-only edge monitor clearly separated from prediction-quality diagnostics.
