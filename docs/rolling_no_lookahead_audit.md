# Rolling-window / no-lookahead audit

This repo's betting-research pipeline is intentionally structured so match-level prediction features are available **before** the match being evaluated.

## What is rolling

In `scripts/betting_research_pipeline.py`, `make_side_dataset()` iterates matches sorted by date/tournament/round/player. For every match it:

1. Groups matches by date.
2. Reads the current pre-match `PlayerState` for both players from state ending before that date.
3. Builds features from prior state only:
   - `p1_prior_win_pct`, `p2_prior_win_pct`
   - `p1_last5_pct`, `p2_last5_pct`
   - `p1_surface_pct`, `p2_surface_pct`
   - `h2h_diff`
   - prior match counts
4. Only **after** writing every feature row for that date, it updates player/H2H states with that date's results.

Because Tennis-data rows have dates but not match start times, same-day results are intentionally not allowed to leak into another match on that same date. This is stricter than row-by-row chronological updating and better matches a daily pre-match prediction workflow.

## What is not used as predictive input

The betting-research model does **not** use same-match post-match stat columns such as aces, double faults, first-serve percentage, return points won, break points, or minutes. Those are useful for updating rolling player profiles after completion, but using them to predict the same row would be leakage.

## Walk-forward evaluation

`scripts/niche_submarket_research.py` uses yearly walk-forward tests:

- For test year `Y`, training rows are strictly `date.year < Y`.
- Test rows are `date.year == Y`.
- Rolling features inside the test year may incorporate earlier completed matches from the same year, matching daily real-world operation.

## Betting side evaluation

For each test match, the niche script computes:

- `edge_p1 = model_p1 - no_vig_implied_p1`
- `edge_p2 = (1 - model_p1) - no_vig_implied_p2`

It evaluates the side with the larger model-vs-market edge, while still allowing no bet if the edge is below threshold.

## Current warning

The current model does not beat the market on broad OOS tests. Any positive niche slice should be treated as a hypothesis only unless it persists across years with enough bets and survives forward paper trading / closing-line-value checks.
