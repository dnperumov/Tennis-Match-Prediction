# Tennis model improvement pipeline: 10-feature implementation

Generated: 2026-06-01

Research-only. This repo must not place bets or send financial orders. API credentials must stay in local environment variables or secret files; do not paste keys into chat or commit them.

## What was implemented

### 1. Pre-match odds snapshots / CLV infrastructure

Added `scripts/odds_snapshot_store.py`.

It creates a SQLite store at:

- `data/odds/odds_snapshots.sqlite`

Tables/views:

- `odds_snapshots`: one row per match/side/source/timestamp
- `latest_match_clv`: computes open vs close probability movement per match/side

Supported commands:

```bash
.venv/bin/python scripts/odds_snapshot_store.py init
.venv/bin/python scripts/odds_snapshot_store.py import-csv path/to/odds_snapshots.csv
.venv/bin/python scripts/odds_snapshot_store.py export-clv --out data/odds/latest_clv.csv
.venv/bin/python scripts/odds_snapshot_store.py kalshi-snapshot --ticker <TICKER> --match-id <ID> --player1 A --player2 B --side player1 --dry-run
```

Kalshi integration is read-only snapshot infrastructure. It does **not** sign orders, place trades, or include order endpoints.

### 2. Travel + schedule fatigue features

Added rolling no-lookahead features:

- `p1_travel_km`, `p2_travel_km`, `travel_km_diff`
- `p1_country_switch`, `p2_country_switch`, `country_switch_diff`
- `p1_continent_switch`, `p2_continent_switch`, `continent_switch_diff`
- `p1_title_and_continent_switch`, `p2_title_and_continent_switch`

A partial built-in tournament location map was added. Unknown locations safely fall back to zero-distance/unknown metadata. Expanding the tournament geocode table is a high-priority data improvement.

### 3. Qualifier / wildcard / lucky loser / protected-ranking feature hooks

Added model columns:

- `p1_qualifier`, `p2_qualifier`, `qualifier_diff`
- `p1_wildcard`, `p2_wildcard`, `wildcard_diff`
- `p1_lucky_loser`, `p2_lucky_loser`, `lucky_loser_diff`
- `p1_protected_ranking`, `p2_protected_ranking`, `protected_ranking_diff`

Current values are placeholders until draw/entry-source ETL is added. The model can now consume those fields once available.

### 4. Challenger / lower-tour form feature hooks

Added neutral placeholder columns:

- `p1_challenger_form`, `p2_challenger_form`, `challenger_form_diff`
- `p1_challenger_title_30d`, `p2_challenger_title_30d`, `challenger_title_30d_diff`

These are ready for Challenger-result ingestion.

### 5. Injury / retirement / withdrawal proxies

Added rolling features:

- `p1_retired_within_30`, `p2_retired_within_30`, `retired_within_30_diff`
- `p1_long_layoff_45`, `p2_long_layoff_45`, `long_layoff_45_diff`

Retirement/walkover is inferred from match comments when present. Long layoff is inferred from rest days.

### 6. Segment-tuned residual shrinkage

Added:

- `residual_overlay_segment_tuned_p1`

Instead of one global residual shrink, weak segments now use separate shrink factors:

- early ATP250: `0.35`
- post-title/final: `0.40`
- surface switch: `0.45`
- Grand Slam: `0.55`
- top-10 match: `0.65`
- early surface switch: `0.35`

### 7. CLV target readiness

`odds_snapshot_store.py` now stores the data needed to train future CLV/movement targets:

- `opening_prob`
- `closing_prob`
- `clv_prob_delta`
- source/book/market ticker metadata
- timestamps and minutes until match

Historical backtest does not yet train a CLV target because real snapshots do not exist yet.

### 8. Style matchup proxies

Added rolling score-derived proxies:

- `set_win_pct_diff`
- `game_win_pct_diff`
- `tiebreak_rate_diff`
- `straight_set_win_rate_diff`

These are weaker than true serve/return matchup stats, but they are no-lookahead and available from current score data.

### 9. Tournament-specific context

Added rolling tournament/surface/round history:

- `tournament_favorite_win_rate`
- `tournament_upset_rate`

These are computed only from prior matches, avoiding same-day leakage.

### 10. Underperformance-risk filter

Added:

- `underperformance_risk`
- `residual_overlay_filtered_p1`

The filter predicts when advanced features are likely to trail market and falls back to market probability when risk is high.

## Latest verified walk-forward result

Test years: 2022-2026, trained only on prior years.

Top models after this pass:

- `market_no_vig`: log loss `0.588846`
- `residual_overlay_segment_tuned`: log loss `0.589747`
- `residual_overlay_filtered`: log loss `0.589883`
- `residual_overlay`: log loss `0.590227`
- `blend_market_advanced_25`: log loss `0.592753`
- `advanced_features`: log loss `0.619931`

Segment-tuned residual overlay is now the best model-produced probability, but market still wins overall.

## Segment diagnostics

Selected segment log loss:

- early round: market `0.5936`, tuned overlay `0.5958`
- early ATP250: market `0.6242`, tuned overlay `0.6267`
- post-title/final: market `0.5347`, tuned overlay `0.5345`
- Grand Slam: market `0.5071`, tuned overlay `0.5075`
- top-10 match: market `0.4858`, tuned overlay `0.4860`
- surface switch: market `0.6013`, tuned overlay `0.6041`
- continent switch: market `0.5876`, tuned overlay `0.5898`
- long layoff: market `0.5790`, tuned overlay `0.5804`

The strongest new result is that post-title/final is slightly better than market in this run. Top-10 is roughly market-level. Early ATP250 remains close but still trails market.

## Betting-threshold research result

Best paper-threshold from the segment-tuned residual overlay:

- threshold: `0.08`
- bets: `73`
- profit: `+19.62 units`

This is still research-only and not deployable. It needs forward paper tracking and CLV evidence.

## Safe Kalshi/API setup

Do not paste API keys into chat and do not commit them.

Use local environment variables or a `.env` file excluded from git. Example names:

```bash
export KALSHI_BASE_URL="https://api.elections.kalshi.com"
export KALSHI_API_KEY_ID="..."
export KALSHI_PRIVATE_KEY_PATH="$HOME/.secrets/kalshi_private_key.pem"
```

Current code only uses the public/read-only market snapshot path and does not implement order placement.

## Next data tasks

1. Expand tournament location geocode coverage.
2. Add draw-entry ETL for qualifier/wildcard/lucky-loser/protected-ranking flags.
3. Add Challenger results ingestion.
4. Add robust injury/withdrawal feed.
5. Start collecting odds snapshots daily and evaluate CLV before any live decision-making.
