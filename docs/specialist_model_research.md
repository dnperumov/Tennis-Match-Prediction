# Specialist model research: surface, Grand Slam, and top-10 buckets

Generated from `scripts/specialist_model_research.py` on `2026-06-01T18:53:27Z`.

Research only. No bet execution or sportsbook connectivity.

## Setup

Data source: Tennis-data ATP odds/results files, 2019-2026.

Walk-forward test years: 2022, 2023, 2024, 2025, 2026.

Feature policy: rolling/no-lookahead features only. Player/H2H state is read before the match date is updated; same-day matches do not update each other because the source has dates but no start times.

Models compared:

- `global`: one model across all ATP rows.
- `surface_specialist`: separate models for Hard, Clay, Grass.
- `grand_slam_specialist`: separate Grand Slam vs non-Grand-Slam models.
- `top10_specialist`: separate no-top-10 / one-top-10 / both-top-10 models.
- `surface_x_top10_specialist`: separate surface x top-10 bucket models.
- `surface_x_slam_specialist`: separate surface x Grand Slam bucket models.

Minimum specialist bucket training rows: 450.

## Overall result

Market no-vig probabilities still beat every model on log loss/Brier. Specialization did **not** create a deployable betting edge.

Overall model comparison by out-of-sample log loss:

- `top10_specialist`: log loss `0.627760`, delta vs global `-0.000012`, Brier `0.219077`.
- `global`: log loss `0.627772`, Brier `0.218220`.
- `grand_slam_specialist`: log loss `0.628623`, delta `+0.000851`.
- `surface_specialist`: log loss `0.631340`, delta `+0.003568`.
- `surface_x_top10_specialist`: log loss `0.631607`, delta `+0.003835`.
- `surface_x_slam_specialist`: log loss `0.636492`, delta `+0.008720`.

Interpretation:

- Top-10 specialization is basically tied with global on log loss, but worse on Brier. It is not enough to claim improvement.
- Pure surface models overfit / lose too much sample size overall.
- Grand Slam specialization alone does not fix Grand Slams.
- Combined specialist buckets generally overfit due to small bucket sizes.

## Betting threshold result

All broad candidate-side threshold strategies remained negative. Best-by-profit examples:

- `global`: best threshold `0.10`, profit `-589.06`, ROI `-14.60%`.
- `top10_specialist`: best threshold `0.10`, profit `-541.31`, ROI `-12.73%`.
- `surface_x_top10_specialist`: best threshold `0.10`, profit `-547.53`, ROI `-11.63%`.

Specialists reduce losses in some threshold scans but remain meaningfully negative.

## Yearly specialist stability

Top-10 specialization helped in 2022, then faded:

- 2022: `top10_specialist` log loss delta vs global `-0.0062`.
- 2023: `+0.0001`.
- 2024: `+0.0027`.
- 2025: `+0.0014`.
- 2026: `+0.0041`.

Surface specialization was unstable:

- 2022: approximately tied.
- 2023: `+0.0095` worse.
- 2024: `+0.0052` worse.
- 2025: `-0.0004` slight improvement.
- 2026: `+0.0027` worse.

Conclusion: specialist models are not consistently improving OOS probability quality yet.

## Where the global model is going wrong

Segments where the global model most underperforms the market by log loss:

1. `Clay | grand_slam`
   - Rows: `508`
   - Model log loss: `0.5446`
   - Market log loss: `0.4900`
   - Gap: `+0.0545`
2. `Grass | grand_slam`
   - Rows: `508`
   - Model log loss: `0.5678`
   - Market log loss: `0.5149`
   - Gap: `+0.0529`
3. `Hard | one_top10`
   - Rows: `1368`
   - Model log loss: `0.5368`
   - Market log loss: `0.4849`
   - Gap: `+0.0519`
4. `both_top10`
   - Rows: `247`
   - Model log loss: `0.6227`
   - Market log loss: `0.5733`
   - Gap: `+0.0495`
5. `grand_slam`
   - Rows: `2150`
   - Model log loss: `0.5530`
   - Market log loss: `0.5071`
   - Gap: `+0.0459`
6. `one_top10`
   - Rows: `2274`
   - Model log loss: `0.5188`
   - Market log loss: `0.4763`
   - Gap: `+0.0425`
7. `1st Round`
   - Rows: `5178`
   - Model log loss: `0.6351`
   - Market log loss: `0.5935`
   - Gap: `+0.0417`

Interpretation:

- The market is especially strong for Grand Slams and top-player matches.
- The model has good discrimination in those areas (`ROC AUC` often high), but probability calibration / sharpness is still behind the market.
- For early rounds and ATP250/no-top-10 buckets, the model has lower discrimination and probably lacks player-quality/context features.

## Where specialization helped locally

Best local log-loss improvements vs global:

- `Hard | one_top10`, using `surface_specialist`: delta `-0.0199`.
- `Hard | one_top10`, using `surface_x_top10_specialist`: delta `-0.0159`.
- `Hard | one_top10`, using `top10_specialist`: delta `-0.0127`.
- `one_top10`, using `top10_specialist`: delta `-0.0077`.
- `Clay | grand_slam`, using `grand_slam_specialist`: delta `-0.0064`.
- `Hard | non_grand_slam`, using `surface_x_slam_specialist`: delta `-0.0054`.

These are useful signals, but they do not overcome the broad market gap.

## Recommendations

### 1. Do not deploy pure specialist models yet

Current best construction is still:

- global model as baseline probability;
- top10/surface/slam specialization as diagnostic overlays;
- maybe blend specialist probabilities with global probabilities instead of replacing global.

A safer next experiment:

```text
p_final = 0.75 * p_global + 0.25 * p_specialist
```

Tune blend weights walk-forward by bucket. This should reduce overfitting from small specialist samples.

### 2. Add market-aware modeling, but only for research

The market is beating the model heavily. Instead of predicting match winner from scratch, train residual models:

```text
target = result
features include no_vig_market_probability plus rolling/model features
or
residual_target = result - no_vig_market_probability
```

Goal: predict where market is wrong, not rebuild the whole market.

### 3. Add better Grand Slam features

Grand Slams are a model weakness because best-of-five and tournament context matter.

Candidate features:

- best-of-five historical win rate;
- Grand Slam-only rolling win rate;
- Grand Slam round-specific experience;
- major-specific player performance: AO/RG/Wimbledon/USO prior results;
- days since last match / rest advantage;
- prior match length / sets played in tournament;
- left in tournament fatigue: cumulative minutes/sets/games in current event;
- five-set record and deciding-set record.

### 4. Add surface-specific skill features, not just surface-specific models

Surface specialist models lost sample size. Better: keep global model and add richer surface features:

- rolling surface Elo;
- surface-specific hold% / break% from completed matches;
- surface-specific tiebreak rate;
- clay/grass/hard recent form over last 52 weeks;
- player surface preference residual: surface win% minus overall win%;
- opponent-adjusted surface wins.

### 5. Add top-player features

Top-10 matches are where market is strongest and current rank features are too crude.

Candidate features:

- Elo difference and Elo uncertainty;
- top-10/top-20 career record;
- record vs top-10 in last 52 weeks;
- record as favorite/underdog;
- pressure metrics: tiebreak win%, deciding set win%, break-point conversion/saved;
- serve dominance and return dominance rolling by surface.

### 6. Add tournament-stage and first-round context

The model underperforms in 1st/2nd rounds. Likely missing:

- travel/rest since last tournament;
- first match after surface switch;
- indoor/outdoor if available;
- altitude if available;
- home-country/continent;
- qualifier/lucky loser/wildcard flag;
- injury/retirement recent history;
- recent match volume/fatigue.

### 7. Model construction next steps

Recommended next experiment order:

1. Add Elo family features:
   - overall Elo;
   - surface Elo;
   - Grand Slam Elo or best-of-five Elo;
   - recent-form Elo.
2. Add market-residual model:
   - model predicts improvement over no-vig market, not raw win probability.
3. Add blended specialists:
   - global + top10 specialist;
   - global + Grand Slam specialist;
   - global + hard-one-top10 specialist.
4. Evaluate with:
   - walk-forward log loss/Brier vs market;
   - calibration by probability bin;
   - CLV/paper tracking before considering any real betting decisions.

## Artifacts

- Script: `scripts/specialist_model_research.py`
- Latest report: `data/betting_research/latest_specialist_model_research.json`
- Latest predictions: `data/betting_research/latest_specialist_predictions.csv`
