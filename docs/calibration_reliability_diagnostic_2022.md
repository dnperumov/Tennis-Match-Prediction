# 2022 calibration reliability diagnostic

Research-only. No bet placement, trading, sportsbook execution, or financial operations.

Generated from `data/betting_research/latest_advanced_feature_model_research.json` (`generated_at`: 2026-06-02T14:44:02.253166+00:00).

## Why this matters

The current 2022 advanced-feature run shows that the no-vig market baseline still has the best overall probability quality. The clearest concrete calibration target is the 0.4-0.5 predicted-probability bucket: several models under-predict actual player-1 win rate there.

## Overall 2022 model status

| rank | model | rows | accuracy | log loss | Brier | log-loss delta vs market |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `market_no_vig` | 2622 | 0.6812 | 0.5824 | 0.1996 | +0.0000 |
| 2 | `residual_overlay_segment_tuned` | 2622 | 0.6808 | 0.5846 | 0.2008 | +0.0022 |
| 3 | `residual_overlay_filtered` | 2622 | 0.6808 | 0.5846 | 0.2008 | +0.0022 |
| 4 | `residual_overlay` | 2622 | 0.6781 | 0.5861 | 0.2014 | +0.0038 |
| 5 | `blend_market_advanced_25` | 2622 | 0.6777 | 0.5861 | 0.2012 | +0.0038 |
| 6 | `residual_overlay_aggressive` | 2622 | 0.6732 | 0.5908 | 0.2033 | +0.0085 |
| 7 | `market_aware` | 2622 | 0.6735 | 0.5925 | 0.2043 | +0.0101 |
| 8 | `blend_market_advanced_50` | 2622 | 0.6747 | 0.5931 | 0.2044 | +0.0108 |

Interpretation: lower log loss/Brier is better. On this run, residual overlays are close but still trail market by about 0.0022 log loss overall.

## Largest reliability errors in the top models

| model | bucket | rows | mean predicted p1 | actual p1 win rate | calibration error |
|---|---:|---:|---:|---:|---:|
| `market_no_vig` | 0.4-0.5 | 338 | 0.4422 | 0.5592 | -0.1170 |
| `market_no_vig` | 0.8-0.9 | 170 | 0.8380 | 0.8824 | -0.0443 |
| `market_no_vig` | 0.5-0.6 | 373 | 0.5490 | 0.5818 | -0.0328 |
| `residual_overlay_segment_tuned` | 0.4-0.5 | 384 | 0.4425 | 0.5234 | -0.0809 |
| `residual_overlay_segment_tuned` | 0.5-0.6 | 369 | 0.5504 | 0.5881 | -0.0377 |
| `residual_overlay_segment_tuned` | 0.8-0.9 | 179 | 0.8422 | 0.8715 | -0.0293 |
| `residual_overlay_filtered` | 0.4-0.5 | 384 | 0.4425 | 0.5234 | -0.0809 |
| `residual_overlay_filtered` | 0.5-0.6 | 369 | 0.5504 | 0.5881 | -0.0377 |
| `residual_overlay_filtered` | 0.8-0.9 | 178 | 0.8423 | 0.8708 | -0.0284 |
| `residual_overlay` | 0.4-0.5 | 376 | 0.4440 | 0.5346 | -0.0906 |
| `residual_overlay` | 0.5-0.6 | 371 | 0.5507 | 0.5903 | -0.0396 |
| `residual_overlay` | 0.8-0.9 | 176 | 0.8438 | 0.8693 | -0.0255 |
| `blend_market_advanced_25` | 0.4-0.5 | 415 | 0.4427 | 0.5325 | -0.0899 |
| `blend_market_advanced_25` | 0.2-0.3 | 295 | 0.2503 | 0.2034 | +0.0469 |
| `blend_market_advanced_25` | 0.8-0.9 | 164 | 0.8396 | 0.8841 | -0.0445 |

Negative calibration error means the model under-predicted player 1; positive means over-predicted. The repeated 0.4-0.5 under-prediction is the most stable, high-volume issue:

- `market_no_vig`: 338 rows, mean 0.4422, actual 0.5592, error -0.1170.
- `residual_overlay_segment_tuned`: 384 rows, mean 0.4425, actual 0.5234, error -0.0809.
- `residual_overlay`: 376 rows, mean 0.4440, actual 0.5346, error -0.0906.
- `blend_market_advanced_25`: 415 rows, mean 0.4427, actual 0.5325, error -0.0899.

## Grand Slam paper-style benchmark cross-check

Full-tournament Grand Slam benchmark coverage remains 507 evaluated rows across 4 folds; skipped folds: 0.

| rank | model | rows | accuracy | log loss | Brier | log-loss delta vs GS market |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `svm_linear` | 507 | 0.7515 | 0.5015 | 0.1660 | -0.0006 |
| 2 | `spline_logistic` | 507 | 0.7436 | 0.5018 | 0.1664 | -0.0003 |
| 3 | `market_no_vig` | 507 | 0.7495 | 0.5021 | 0.1662 | +0.0000 |
| 4 | `residual_overlay_segment_tuned` | 507 | 0.7456 | 0.5042 | 0.1672 | +0.0021 |
| 5 | `random_forest` | 507 | 0.7475 | 0.5073 | 0.1672 | +0.0052 |
| 6 | `advanced_voting` | 507 | 0.7357 | 0.5129 | 0.1713 | +0.0107 |
| 7 | `xgboost` | 507 | 0.7258 | 0.5226 | 0.1744 | +0.0204 |
| 8 | `logistic_market` | 507 | 0.7239 | 0.5474 | 0.1808 | +0.0453 |

Grand Slam note: `svm_linear` slightly beats market log loss in the 2022 Grand Slam benchmark (-0.0006), but the available repo sample has 507 rows versus the paper target of 293, so this remains a coverage-mismatch diagnostic rather than an apples-to-apples paper replication.

## Next concrete improvement suggested by this diagnostic

Add a calibration experiment that trains on pre-2022 out-of-sample predictions and tests on 2022, with special attention to the 0.4-0.6 range. Candidate methods: isotonic calibration, Platt/logit calibration, and a conservative bucket-aware residual correction. The test should report overall log loss/Brier and bucket-level reliability so we do not improve one bucket by degrading global probability quality.

## Source evidence

- `data/betting_research/latest_advanced_feature_model_research.json`
- `data/betting_research/latest_metrics.json`
- `docs/buhamra_paper_benchmark_replication.md`
