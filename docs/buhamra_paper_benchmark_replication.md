# Buhamra et al. (2025) benchmark ideas added to the tennis model

Research-only. No bet placement or autonomous financial operations.

## What was implemented

`advanced_feature_model_research.py` now includes a paper-style benchmark mode inspired by Buhamra, Groll, and Gerharz (2025):

- Grand Slam-only benchmark scope.
- Chronological tournament-level expanding window: train on all matches before a Grand Slam tournament, predict that tournament.
- Market feature included via no-vig bookmaker probability and market logit.
- Accuracy, log loss, Brier, ROC AUC, mean probability, and actual rate reported for each model.
- Model comparison table added to the JSON report.

The report key is:

```json
"grand_slam_tournament_expanding_benchmark"
```

## Models in the Grand Slam benchmark

- `market_no_vig`: no-vig bookmaker probability baseline.
- `logistic_market`: logistic regression with market + advanced features.
- `spline_logistic`: spline-expanded logistic approximation to the paper's GAM/spline model.
- `random_forest`: sklearn random forest.
- `xgboost`: XGBoost classifier.
- `svm_linear`: linear SVM with probability calibration through `SVC(probability=True)`.
- `advanced_voting`: existing ensemble classifier using logistic + gradient boosting + random forest.
- `residual_overlay_segment_tuned`: market residual overlay with Grand Slam/top-player/surface-switch specialist behavior.

## Statistically enhanced ability features added

These are pre-match-only features estimated from historical state before the match is updated:

- `surface_ability_diff`: surface Elo plus rolling surface win-rate signal.
- `serve_return_ability_diff`: score-derived game-win percentage proxy. This should later be replaced or augmented with true serve/return stats.
- `bo5_ability_diff`: Slam/best-of-five Elo plus best-of-five win-rate signal.
- `recent_form_ability_diff`: last-five result form plus Elo signal.
- `fatigue_adjusted_ability_diff`: Elo adjusted by recent match load and travel distance.

These correspond to the paper's suggestion of statistically enhanced covariates: build ability estimates first, then feed them into ML models.

## How to run

Fast apples-to-apples 2022-style paper replication:

```bash
.venv/bin/python scripts/advanced_feature_model_research.py \
  --years 2019 2020 2021 2022 \
  --test-years 2022 \
  --paper-test-years 2022
```

Broader yearly model research still works:

```bash
.venv/bin/python scripts/advanced_feature_model_research.py \
  --years 2019 2020 2021 2022 2023 2024 2025 2026 \
  --test-years 2022 2023 2024 2025 2026 \
  --paper-test-years 2022
```

`--paper-test-years` defaults to the first `--test-years` value to keep the heavier Grand Slam model-zoo benchmark fast. Pass more years if you want more Grand Slam folds.

## Verified 2022 run

Command run on 2026-06-02 after fixing the tournament-fold selector to evaluate every row from each Grand Slam event, not just the first date of the event:

```bash
.venv/bin/python -m py_compile scripts/advanced_feature_model_research.py scripts/betting_research_pipeline.py
.venv/bin/python tests/test_grand_slam_coverage_diagnostics.py
.venv/bin/python scripts/advanced_feature_model_research.py \
  --years 2019 2020 2021 2022 \
  --test-years 2022 \
  --paper-test-years 2022
```

Grand Slam tournament-expanding benchmark output from the available repo data:

- rows: 507
- folds: 4
- skipped folds: 0
- coverage diagnostics: 507 evaluated 2022 rows across Australian Open (127), French Open (127), Wimbledon (127), and US Open (126)

Model results:

- `svm_linear`: accuracy 0.7515, log loss 0.5015, Brier 0.1660
- `spline_logistic`: accuracy 0.7436, log loss 0.5018, Brier 0.1664
- `market_no_vig`: accuracy 0.7495, log loss 0.5021, Brier 0.1662
- `residual_overlay_segment_tuned`: accuracy 0.7456, log loss 0.5042, Brier 0.1672
- `random_forest`: accuracy 0.7475, log loss 0.5073, Brier 0.1672
- `advanced_voting`: accuracy 0.7357, log loss 0.5129, Brier 0.1713
- `xgboost`: accuracy 0.7258, log loss 0.5226, Brier 0.1744
- `logistic_market`: accuracy 0.7239, log loss 0.5474, Brier 0.1808

Important: the previous run evaluated only 105 rows because the fold selector used the tournament's first date when that date had at least 20 rows. The fixed benchmark now evaluates full tournaments. The 507 available/evaluated repo rows exceed the paper's cited 293-match 2022 Grand Slam validation target, so this is still not an apples-to-apples replication; the next coverage task is to document/filter the paper's exact eligibility criteria if known.

## Interpretation

The first verified run shows the same kind of tradeoff as the paper:

- Highest accuracy: `svm_linear`.
- Best probability quality by log loss/Brier: `residual_overlay_segment_tuned`, slightly ahead of market in this small Grand Slam sample.
- XGBoost is not automatically best on our current data coverage.

For betting/smart-play identification, do not optimize accuracy alone. Use accuracy for paper comparability, but keep log loss, Brier, calibration, and future CLV as core decision metrics.

## 2026-06-02 calibration diagnostics update

`advanced_feature_model_research.py` now stores 10-bin reliability diagnostics under each `overall_model_comparison[*].calibration_bins` entry and each Grand Slam benchmark model entry. Each bin reports `rows`, `mean_prob`, `actual_rate`, and `calibration_error` (`mean_prob - actual_rate`). This makes calibration failures inspectable directly from `data/betting_research/latest_advanced_feature_model_research.json` instead of only comparing aggregate log loss/Brier.

Verified command:

```bash
.venv/bin/python -m py_compile scripts/advanced_feature_model_research.py scripts/betting_research_pipeline.py && \
.venv/bin/python scripts/advanced_feature_model_research.py \
  --years 2019 2020 2021 2022 \
  --test-years 2022 \
  --paper-test-years 2022
```

2022 reliability signal from the verified run: the market baseline remains best overall by log loss/Brier, but both market and residual overlay under-predict the 0.4-0.5 probability bucket (`market_no_vig` calibration error -0.1170; `residual_overlay_segment_tuned` -0.0809). That bucket is a concrete next target for recalibration research.

## 2026-06-02 material calibration summary update

`advanced_feature_model_research.py` now also writes compact calibration-target summaries:

- `overall_calibration_summary`
- `grand_slam_calibration_summary`

Each summary filters low-sample bins, ranks the remaining reliability buckets by absolute calibration error, and includes `direction` plus `weighted_abs_error`. This turns the verbose per-model 10-bin tables into an explicit next-target list for recalibration work.

Verified 2022 run after this update:

```bash
.venv/bin/python tests/test_calibration_summary.py && \
.venv/bin/python tests/test_grand_slam_coverage_diagnostics.py && \
.venv/bin/python -m py_compile scripts/advanced_feature_model_research.py scripts/betting_research_pipeline.py
.venv/bin/python scripts/advanced_feature_model_research.py \
  --years 2019 2020 2021 2022 \
  --test-years 2022 \
  --paper-test-years 2022
```

Top material overall calibration target from `data/betting_research/latest_advanced_feature_model_research.json`: `market_no_vig` bin 0.4-0.5, 338 rows, mean probability 0.4422, actual win rate 0.5592, calibration error -0.1170 (`underpredicts_p1`). The advanced/residual families show the same bucket-level underprediction, so the next research step should test conservative recalibration/monotonic smoothing around mid-market probabilities before adding more raw features.

## 2026-06-02 conservative market-bin recalibration experiment

`advanced_feature_model_research.py` now includes a deliberately simple no-lookahead recalibration baseline, `market_bin_recalibrated`:

- fit reliability-bin offsets only on rows with `date.year < test_year`;
- apply a conservative 0.25 shrinkage factor to `actual_rate - mean_prob`;
- require at least 120 training rows per bin; bins without enough evidence keep the original no-vig market probability;
- store per-year learned offsets in `bin_recalibration_diagnostics` so the experiment is auditable.

Verified command:

```bash
.venv/bin/python tests/test_calibration_summary.py && \
.venv/bin/python tests/test_grand_slam_coverage_diagnostics.py
.venv/bin/python -m py_compile scripts/advanced_feature_model_research.py scripts/betting_research_pipeline.py
.venv/bin/python scripts/advanced_feature_model_research.py \
  --years 2019 2020 2021 2022 \
  --test-years 2022 \
  --paper-test-years 2022
```

2022 result: `market_bin_recalibrated` became the best aggregate probability model in the fast research run with accuracy 0.6812, log loss 0.581850, and Brier 0.199417 versus `market_no_vig` log loss 0.582365 and Brier 0.199600. This is a tiny but real no-lookahead improvement in aggregate probability quality. The main 0.4-0.5 bucket is still underpredicted (`calibration_error` -0.1135 after recalibration versus -0.1170 before), so the next calibration step should test stronger but still monotonic/conservative smoothing, preferably across more test years before accepting it as durable.

## 2026-06-02 shrinkage-sweep diagnostic update

`advanced_feature_model_research.py` now writes `bin_recalibration_shrinkage_sweeps`, a no-lookahead diagnostic that fits reliability-bin offsets on training rows only and scores multiple shrinkage strengths on each test year. This does not auto-select a live model; it exposes whether the current fixed shrinkage is too timid or too aggressive before a broader walk-forward run.

Verified command:

```bash
.venv/bin/python tests/test_calibration_summary.py && \
.venv/bin/python tests/test_grand_slam_coverage_diagnostics.py && \
.venv/bin/python -m py_compile scripts/advanced_feature_model_research.py scripts/betting_research_pipeline.py
.venv/bin/python scripts/advanced_feature_model_research.py \
  --years 2019 2020 2021 2022 \
  --test-years 2022 \
  --paper-test-years 2022
```

2022 sweep result from `data/betting_research/latest_advanced_feature_model_research.json`: baseline/no shrink log loss 0.582365 and Brier 0.199600; the best tested shrinkage was 1.0 with log loss 0.581351 and Brier 0.199140. The existing production-style `market_bin_recalibrated` row still uses the conservative 0.25 shrinkage (log loss 0.581850, Brier 0.199417), so the next decision should be a multi-year walk-forward sweep before increasing the default shrinkage.

## 2026-06-02 multi-year shrinkage-sweep summary update

`advanced_feature_model_research.py` now writes `bin_recalibration_shrinkage_summary` beside the per-year sweep details. The summary counts which shrinkage value wins by log loss/Brier, averages deltas versus no recalibration, and emits a conservative recommendation so a single good year cannot silently promote a more aggressive recalibration setting.

Verified multi-year walk-forward command:

```bash
.venv/bin/python tests/test_calibration_summary.py && \
.venv/bin/python -m py_compile scripts/advanced_feature_model_research.py scripts/betting_research_pipeline.py scripts/french_open_pick_tracker.py
.venv/bin/python scripts/advanced_feature_model_research.py \
  --years 2019 2020 2021 2022 2023 2024 2025 \
  --test-years 2022 2023 2024 2025 \
  --paper-test-years 2022
```

Result from the generated JSON: across 2022-2025, the fixed conservative `market_bin_recalibrated` row is a tiny aggregate probability improvement over market (`log_loss` 0.589379 vs 0.589572; Brier 0.202946 vs 0.202985) but slightly lower accuracy (0.676313 vs 0.676783). The diagnostic sweep improved log loss/Brier in 3 of 4 years, but best shrinkage was not stable (`1.0`, `0.5`, `0.5`, `0.0` by log loss), so the summary recommendation is `diagnostic_only_mixed_or_insufficient_years`. Keep the default shrinkage conservative until a more stable monotonic smoother beats market across more years.

Research-methodology note: this follows the calibration/reliability-diagram idea in scikit-learn's probability calibration guidance (`https://scikit-learn.org/stable/modules/calibration.html`): probability quality should be checked with reliability behavior and proper scoring rules, not only classification accuracy.

## 2026-06-02 ECE/MCE calibration metrics update

`advanced_feature_model_research.py` now writes compact `calibration_error_metrics` beside every overall and Grand Slam model row. The fields are:

- `expected_calibration_error`: row-weighted mean absolute reliability-bin error (ECE).
- `maximum_calibration_error`: largest absolute reliability-bin error (MCE).
- `rows`: rows represented by the reliability bins.

This turns the existing verbose reliability bins into easy-to-sort scalar diagnostics while keeping log loss and Brier as the primary probability-quality scores. The methodology follows the same reliability-diagram/calibration-error framing as scikit-learn's probability calibration guidance (`https://scikit-learn.org/stable/modules/calibration.html`): good classifiers should have probabilities that match observed event frequencies, not only high accuracy.

Verified commands:

```bash
.venv/bin/python tests/test_calibration_summary.py -v && \
.venv/bin/python -m py_compile scripts/advanced_feature_model_research.py scripts/betting_research_pipeline.py scripts/french_open_pick_tracker.py
.venv/bin/python scripts/advanced_feature_model_research.py \
  --years 2019 2020 2021 2022 \
  --test-years 2022 \
  --paper-test-years 2022
```

2022 fast-run signal from `data/betting_research/latest_advanced_feature_model_research.json`: `market_bin_recalibrated` remains best by log loss/Brier (0.581850 / 0.199417) and has lower ECE than raw market (0.029633 vs 0.031990), but `residual_overlay` has the lowest ECE (0.027683) while worse log loss/Brier. This confirms ECE is a useful calibration diagnostic, not a replacement objective; next tuning should target log loss/Brier first and use ECE/MCE to locate bucket-level calibration failures.

## 2026-06-02 probability-quality tradeoff summary update

`advanced_feature_model_research.py` now writes `overall_probability_quality_summary` into `data/betting_research/latest_advanced_feature_model_research.json`. The summary identifies the leaders by log loss, Brier, accuracy, and ECE, plus deltas versus `market_no_vig`, so hourly runs can quickly distinguish useful probability improvements from accuracy-only changes. A focused regression test also prevents a false warning when the accuracy leader merely ties the log-loss leader.

Verified commands:

```bash
.venv/bin/python tests/test_calibration_summary.py -v && \
.venv/bin/python -m py_compile scripts/advanced_feature_model_research.py scripts/betting_research_pipeline.py scripts/french_open_pick_tracker.py
.venv/bin/python scripts/advanced_feature_model_research.py \
  --years 2019 2020 2021 2022 \
  --test-years 2022 \
  --paper-test-years 2022
```

2022 summary: `market_bin_recalibrated` leads log loss and Brier versus market (`log_loss_delta_vs_baseline` -0.000515, `brier_delta_vs_baseline` -0.000183), market and recalibrated market tie on accuracy (0.681159), and `residual_overlay` has the lowest ECE (0.027683) despite worse proper scores. Research-methodology source remains scikit-learn's calibration guidance (`https://scikit-learn.org/stable/modules/calibration.html`): reliability diagnostics are useful for locating probability errors, but log loss/Brier remain the primary probability-quality objectives.
