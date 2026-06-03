# Advanced feature/model research

Generated from `scripts/advanced_feature_model_research.py` on `2026-06-01T20:01:12Z`.

Research only. No bet execution.

## What changed in this pass

Applied the prior findings instead of only reporting them:

1. Added a true **market residual overlay**:
   - target: `result - implied_p1_no_vig`
   - prediction: `market_probability + shrink * predicted_residual`
   - conservative shrink: `0.55`
   - aggressive shrink: `0.85`

2. Added specialist residual overlays for the weak clusters:
   - early ATP250
   - post-title / post-final within 7-14 days
   - surface switch
   - Grand Slam
   - top-10 match
   - early-round + surface-switch

3. Kept all feature generation no-lookahead:
   - rows are emitted before updating states with the current date
   - same-day matches do not leak into each other

4. Added `market_no_vig` as an explicit baseline in the output metrics.

## Artifacts

- Script: `scripts/advanced_feature_model_research.py`
- Report JSON: `data/betting_research/latest_advanced_feature_model_research.json`
- Predictions CSV: `data/betting_research/latest_advanced_feature_predictions.csv`

## Walk-forward setup

- source years: 2019-2026
- test years: 2022, 2023, 2024, 2025, 2026
- train: all years before each test year

## Overall OOS results

By log loss:

- `market_no_vig`: `0.588846`
  - Brier: `0.202618`
- `residual_overlay`: `0.589966`
  - vs market: `+0.001120`
  - vs base: `-0.035463`
  - Brier: `0.203232`
- `residual_overlay_aggressive`: `0.592276`
  - vs market: `+0.003430`
  - vs base: `-0.033153`
- `blend_market_advanced_25`: `0.592669`
  - vs market: `+0.003823`
  - vs base: `-0.032760`
- `market_aware`: `0.592875`
  - vs market: `+0.004029`
- `advanced_features`: `0.620373`
  - vs base: `-0.005056`
- `base_features`: `0.625428`

## Key conclusion

The new residual overlay is now the best model-produced probability, but it still does **not** beat the closing no-vig market overall.

Important: it gets much closer to market than the previous blend/market-aware models:

- old best model-produced: `blend_market_advanced_25`, log loss `0.592669`
- new residual overlay: `0.589966`
- improvement: `0.002703` log-loss points

That is a real improvement, but not enough to claim edge.

## Yearly log loss

- 2022:
  - market: `0.5824`
  - residual overlay: `0.5865`
  - blend 25: `0.5859`
- 2023:
  - market: `0.5898`
  - residual overlay: `0.5895`
  - residual beat market by a tiny amount
- 2024:
  - market: `0.5850`
  - residual overlay: `0.5843`
  - residual beat market by a tiny amount
- 2025:
  - market: `0.6014`
  - residual overlay: `0.6025`
- 2026:
  - market: `0.5822`
  - residual overlay: `0.5841`

The overlay beat market in 2023 and 2024, but lost in 2022, 2025, and 2026. Not robust enough yet.

## Segment results after applying residual overlay

### Early rounds

- market: `0.5936`
- advanced-only: `0.6311`
- residual overlay: `0.5967`

Huge improvement vs advanced-only, but still slightly worse than market.

### 1st Round

- market: `0.5935`
- advanced-only: `0.6313`
- residual overlay: `0.5965`

Same story: residual overlay largely fixes the model’s early-round overconfidence/underconfidence, but market remains better.

### Early ATP250

- market: `0.6242`
- advanced-only: `0.6654`
- residual overlay: `0.6289`

This was one of the biggest weak clusters. The specialist overlay helps a lot.

### Post-title / post-final

- market: `0.5347`
- advanced-only: `0.5639`
- residual overlay: `0.5353`

This is the strongest evidence that the specific title/final hangover adjustment worked. It almost fully closes the gap to market.

### Surface switch

- market: `0.6013`
- advanced-only: `0.6375`
- residual overlay: `0.6053`

Large improvement, still short of market.

### Grand Slam

- market: `0.5071`
- advanced-only: `0.5419`
- residual overlay: `0.5075`

Grand Slam overlay basically closes the gap.

### Top-10 match

- market: `0.4858`
- advanced-only: `0.5075`
- residual overlay: `0.4848`

This is the best specialist result: residual overlay slightly beats market in top-10 matches.

## Betting threshold backtest

Still not deployable.

Best residual-overlay threshold:

- threshold: `0.08`
- bets: `102`
- profit: `+2.26 units`
- ROI: `+2.22%`
- hit rate: `54.9%`
- avg odds: `2.09`

This is better behaved than the previous longshot-heavy blend result, but too small and threshold-mined to trust.

## Interpretation

The improvements worked in the right direction:

- advanced-only model found real tennis signal but was too far from market
- residual overlay transformed that signal into a market-adjacent correction
- weak clusters improved materially:
  - early rounds
  - early ATP250
  - post-title/final
  - surface switch
  - Slams
  - top-10 matches

But the closing market remains extremely hard to beat. The current model should be treated as a **research overlay / paper-tracking model**, not a betting system.

## Next changes to apply

1. Use actual pre-match odds snapshots, not closing B365 odds, so we can measure CLV.
2. Train the residual model on CLV direction when odds snapshots exist.
3. Add richer schedule/travel features:
   - previous tournament country/continent
   - distance between events
   - same-country/same-continent flag
   - days since prior tournament ended
4. Add player-entry context:
   - qualifier
   - lucky loser
   - wildcard
   - protected ranking
   - Implemented in the feature builder when optional draw columns such as `WEntry`/`LEntry`, `winner_entry`/`loser_entry`, or `Winner Entry`/`Loser Entry` are present; missing entry data safely remains all-zero.
5. Add Challenger/current-form context for ATP250 early rounds.
   - Implemented as no-lookahead rolling Challenger form/title features when Challenger rows are included in the input feed; the current tennis-data ATP-only feed still has no Challenger source rows, so these remain neutral until that ETL is added.
6. Add withdrawal/retirement/injury flags.
7. Paper-track only these model families:
   - `market_no_vig`
   - `residual_overlay`
   - segment-specific residual overlay for top-10 and post-title/final matches

## Bottom line

The applied changes improved the model materially. The best production candidate is now:

`residual_overlay = market_no_vig + 0.55 * predicted_market_residual`

But it is still not a standalone edge. The next real unlock is better pre-match market timing + travel/injury/entry features, especially for early ATP250 and post-title/final spots.
