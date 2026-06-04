#!/usr/bin/env python3
"""Regression tests for calibration summary diagnostics."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from advanced_feature_model_research import (  # noqa: E402
    apply_bin_recalibration,
    calibration_error_metrics,
    evaluate_bin_recalibration_shrinkage_sweep,
    fit_bin_recalibration,
    segment_errors,
    disagreement_fallback_routing_diagnostic,
    summarize_calibration_diagnostics,
    interaction_segment_errors,
    interaction_model_market_disagreement_segments,
    model_market_agreement_segments,
    market_favorite_pressure_segments,
    model_market_disagreement_segments,
    multivariate_segment_errors,
    no_lookahead_blend_weight_diagnostic,
    no_lookahead_agreement_sizing_blend_diagnostic,
    no_lookahead_market_favorite_pressure_blend_diagnostic,
    no_lookahead_disagreement_margin_blend_diagnostic,
    no_lookahead_underperformance_risk_threshold_diagnostic,
    build_calibrated_market_blend_diagnostics,
    disagreement_margin_segments,
    player_involvement_segments,
    summarize_probability_quality_tradeoffs,
    summarize_segment_strengths,
    summarize_segment_weaknesses,
    summarize_shrinkage_sweeps,
)


class CalibrationSummaryTest(unittest.TestCase):
    def test_bin_recalibration_learns_train_offsets_and_preserves_unseen_bins(self) -> None:
        import pandas as pd

        train = pd.DataFrame({
            "result": [1, 0, 0, 1],
            "market_p1": [0.44, 0.46, 0.62, 0.64],
        })

        table = fit_bin_recalibration(train, "market_p1", bins=10, min_rows=2, shrink=1.0)
        test = pd.DataFrame({"market_p1": [0.45, 0.63, 0.82]})

        recalibrated = apply_bin_recalibration(test, "market_p1", table, bins=10)

        self.assertAlmostEqual(float(recalibrated.iloc[0]), 0.50)
        self.assertAlmostEqual(float(recalibrated.iloc[1]), 0.50)
        self.assertAlmostEqual(float(recalibrated.iloc[2]), 0.82)

    def test_calibration_error_metrics_reports_weighted_ece_and_mce(self) -> None:
        bins = [
            {"rows": 100, "calibration_error": -0.10},
            {"rows": 300, "calibration_error": 0.02},
            {"rows": 0, "calibration_error": 0.90},
        ]

        metrics = calibration_error_metrics(bins)

        self.assertEqual(metrics["rows"], 400)
        self.assertAlmostEqual(metrics["expected_calibration_error"], 0.04)
        self.assertAlmostEqual(metrics["maximum_calibration_error"], 0.10)

    def test_summarize_calibration_diagnostics_prioritizes_material_miscalibration(self) -> None:
        model_rows = [
            {
                "model": "market_no_vig",
                "log_loss": 0.58,
                "brier": 0.20,
                "calibration_bins": [
                    {"bin": 0, "rows": 10, "mean_prob": 0.05, "actual_rate": 0.30, "calibration_error": -0.25},
                    {"bin": 4, "rows": 200, "mean_prob": 0.45, "actual_rate": 0.55, "calibration_error": -0.10},
                ],
            },
            {
                "model": "residual_overlay_segment_tuned",
                "log_loss": 0.59,
                "brier": 0.21,
                "calibration_bins": [
                    {"bin": 6, "rows": 150, "mean_prob": 0.65, "actual_rate": 0.61, "calibration_error": 0.04},
                ],
            },
        ]

        summary = summarize_calibration_diagnostics(model_rows, min_rows=50, top_n=2)

        self.assertEqual(
            summary["worst_bins"][0],
            {
                "model": "market_no_vig",
                "bin": 4,
                "prob_min": 0.4,
                "prob_max": 0.5,
                "rows": 200,
                "mean_prob": 0.45,
                "actual_rate": 0.55,
                "calibration_error": -0.10,
                "direction": "underpredicts_p1",
                "abs_calibration_error": 0.10,
                "weighted_abs_error": 20.0,
            },
        )
        self.assertEqual(summary["worst_bins"][1]["model"], "residual_overlay_segment_tuned")
        self.assertEqual(summary["excluded_low_sample_bins"], 1)
        self.assertEqual(summary["min_rows"], 50)

    def test_summarize_calibration_diagnostics_handles_empty_input(self) -> None:
        self.assertEqual(
            summarize_calibration_diagnostics([], min_rows=25, top_n=3),
            {
                "min_rows": 25,
                "top_n": 3,
                "excluded_low_sample_bins": 0,
                "worst_bins": [],
            },
        )

    def test_summarize_probability_quality_tradeoffs_identifies_leaders_and_deltas(self) -> None:
        rows = [
            {
                "model": "market_no_vig",
                "accuracy": 0.68,
                "log_loss": 0.582,
                "brier": 0.200,
                "calibration_error_metrics": {"expected_calibration_error": 0.032},
            },
            {
                "model": "market_bin_recalibrated",
                "accuracy": 0.681,
                "log_loss": 0.581,
                "brier": 0.199,
                "calibration_error_metrics": {"expected_calibration_error": 0.029},
            },
            {
                "model": "accuracy_only_model",
                "accuracy": 0.70,
                "log_loss": 0.610,
                "brier": 0.215,
                "calibration_error_metrics": {"expected_calibration_error": 0.055},
            },
        ]

        summary = summarize_probability_quality_tradeoffs(rows, baseline_model="market_no_vig")

        self.assertEqual(summary["baseline_model"], "market_no_vig")
        self.assertEqual(summary["best_by_log_loss"]["model"], "market_bin_recalibrated")
        self.assertEqual(summary["best_by_brier"]["model"], "market_bin_recalibrated")
        self.assertEqual(summary["best_by_accuracy"]["model"], "accuracy_only_model")
        self.assertEqual(summary["best_by_ece"]["model"], "market_bin_recalibrated")
        self.assertAlmostEqual(summary["best_by_log_loss"]["delta_vs_baseline"], -0.001)
        self.assertEqual(
            summary["warning"],
            "accuracy leader is not the log-loss leader; prioritize calibrated probability quality over accuracy-only gains",
        )

    def test_calibrated_market_blend_diagnostics_use_bin_recalibrated_baseline(self) -> None:
        import pandas as pd

        preds = pd.DataFrame({
            "date": pd.to_datetime([
                "2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04",
                "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04",
            ]),
            "result": [1, 1, 0, 0, 1, 1, 0, 0],
            "model_p1": [0.70, 0.72, 0.30, 0.28, 0.70, 0.72, 0.30, 0.28],
            "implied_p1_no_vig": [0.52, 0.52, 0.48, 0.48, 0.52, 0.52, 0.48, 0.48],
            "market_bin_recalibrated_p1": [0.64, 0.64, 0.36, 0.36, 0.64, 0.64, 0.36, 0.36],
        })

        diagnostics = build_calibrated_market_blend_diagnostics(
            preds,
            model_prob_cols=["model_p1"],
            min_train_rows=2,
        )

        self.assertIn("model_p1_market_favorite_pressure_blend_vs_calibrated_market", diagnostics)
        self.assertIn("model_p1_agreement_sizing_blend_vs_calibrated_market", diagnostics)
        self.assertIn("model_p1_disagreement_margin_blend_vs_calibrated_market", diagnostics)
        pressure = diagnostics["model_p1_market_favorite_pressure_blend_vs_calibrated_market"]
        self.assertEqual(pressure["baseline_probability_col"], "market_bin_recalibrated_p1")
        self.assertEqual(pressure["routed_rows"], 4)
        self.assertLessEqual(
            pressure["overall_blended_metrics"]["log_loss"],
            pressure["overall_market_metrics"]["log_loss"],
        )

    def test_underperformance_risk_threshold_uses_prior_oos_years_only(self) -> None:
        import pandas as pd

        preds = pd.DataFrame({
            "date": pd.to_datetime([
                "2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04",
                "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04",
            ]),
            "result": [1, 0, 1, 0, 1, 0, 1, 0],
            "model_p1": [0.90, 0.10, 0.90, 0.10, 0.90, 0.10, 0.90, 0.10],
            "market_p1": [0.55, 0.45, 0.55, 0.45, 0.55, 0.45, 0.55, 0.45],
            "implied_p1_no_vig": [0.55, 0.45, 0.55, 0.45, 0.55, 0.45, 0.55, 0.45],
            "risk": [0.20, 0.20, 0.90, 0.90, 0.20, 0.20, 0.90, 0.90],
        })

        diagnostic = no_lookahead_underperformance_risk_threshold_diagnostic(
            preds,
            model_prob_col="model_p1",
            risk_col="risk",
            baseline_prob_col="market_p1",
            candidate_thresholds=[0.0, 0.5, 1.0],
            min_train_rows=4,
        )

        self.assertEqual(diagnostic["yearly"][0]["selected_threshold"], 1.0)
        self.assertEqual(diagnostic["yearly"][0]["prior_oos_train_rows"], 0)
        self.assertEqual(diagnostic["yearly"][1]["selected_threshold"], 1.0)
        self.assertEqual(diagnostic["yearly"][1]["prior_oos_train_rows"], 4)
        self.assertEqual(diagnostic["routed_probability_col"], "model_p1_underperformance_risk_threshold")
        self.assertEqual(diagnostic["overall_routed_metrics"]["rows"], 8)
        self.assertLess(
            diagnostic["overall_routed_metrics"]["log_loss"],
            diagnostic["overall_market_metrics"]["log_loss"],
        )

    def test_player_involvement_segments_scores_both_player_sides_with_year_stability(self) -> None:
        import pandas as pd

        rows = []
        for year in [2022, 2023, 2024]:
            for i in range(4):
                rows.append({
                    "date": f"{year}-01-{i + 1:02d}",
                    "year": year,
                    "player1": "Target A",
                    "player2": f"Opponent {year}-{i}",
                    "result": 0,
                    "model_p1": 0.82,
                    "implied_p1_no_vig": 0.55,
                })
                rows.append({
                    "date": f"{year}-02-{i + 1:02d}",
                    "year": year,
                    "player1": f"Opponent B {year}-{i}",
                    "player2": "Target A",
                    "result": 1,
                    "model_p1": 0.18,
                    "implied_p1_no_vig": 0.45,
                })
        rows.append({
            "date": "2024-03-01",
            "year": 2024,
            "player1": "Sparse Player",
            "player2": "Other",
            "result": 1,
            "model_p1": 0.80,
            "implied_p1_no_vig": 0.60,
        })
        preds = pd.DataFrame(rows)

        segments = player_involvement_segments(preds, "model_p1", min_rows=6)

        self.assertEqual(segments[0]["segment_col"], "player")
        self.assertEqual(segments[0]["segment"], "Target A")
        self.assertEqual(segments[0]["rows"], 24)
        self.assertEqual(segments[0]["player_rows_as_p1"], 12)
        self.assertEqual(segments[0]["player_rows_as_p2"], 12)
        self.assertEqual(segments[0]["years"], [2022, 2023, 2024])
        self.assertEqual(segments[0]["years_model_lags_market_log_loss"], 3)
        self.assertGreater(segments[0]["model_minus_market_log_loss"], 0)
        self.assertTrue(all(row["segment"] != "Sparse Player" for row in segments))

    def test_segment_errors_includes_rank_odds_rest_fatigue_and_context_buckets(self) -> None:
        import pandas as pd

        rows = []
        for i in range(90):
            rows.append({
                "result": 1 if i % 3 else 0,
                "model_p1": 0.70 if i % 3 else 0.80,
                "implied_p1_no_vig": 0.82,
                "surface": "Clay",
                "series": "ATP250",
                "court": "Outdoor",
                "round_group": "early",
                "round": "1st Round",
                "is_early_round": 1,
                "any_top10": 0,
                "both_top10": 0,
                "early_after_title_p1": 0,
                "early_after_title_p2": 0,
                "p1_title_within_14": 0,
                "p2_title_within_14": 0,
                "p1_final_within_7": 0,
                "p2_final_within_7": 0,
                "p1_surface_switch": 1,
                "p2_surface_switch": 0,
                "rank_diff": 75,
                "rest_diff": -4,
                "matches_last7_diff": 3,
            })
        diagnostics = segment_errors(pd.DataFrame(rows), "model_p1")
        segments_by_col = {row["segment_col"]: row["segment"] for row in diagnostics}

        self.assertEqual(segments_by_col["rank_diff_bucket"], "p1_much_lower_rank")
        self.assertEqual(segments_by_col["market_prob_bucket"], "heavy_p1_favorite")
        self.assertEqual(segments_by_col["rest_diff_bucket"], "p1_less_rest")
        self.assertEqual(segments_by_col["matches_last7_diff_bucket"], "p1_heavier_load")
        self.assertEqual(segments_by_col["surface_switch_any"], "True")
        self.assertEqual(segments_by_col["post_title_or_final_any"], "False")

    def test_market_favorite_pressure_segments_bucket_model_favorite_underpricing(self) -> None:
        import pandas as pd

        rows = []
        for year in [2022, 2023, 2024]:
            for i in range(70):
                rows.append({
                    "date": f"{year}-04-{(i % 9) + 1:02d}",
                    "result": 1,
                    "model_p1": 0.57,
                    "implied_p1_no_vig": 0.74,
                })
            for i in range(10):
                rows.append({
                    "date": f"{year}-05-{(i % 9) + 1:02d}",
                    "result": 0,
                    "model_p1": 0.43,
                    "implied_p1_no_vig": 0.26,
                })

        diagnostics = market_favorite_pressure_segments(pd.DataFrame(rows), "model_p1", min_rows=150)
        by_col = {row["segment_col"]: row for row in diagnostics}

        self.assertEqual(by_col["model_vs_market_favorite_pressure_bucket"]["segment"], "model_underprices_market_favorite_gt10pct")
        self.assertEqual(by_col["model_vs_market_favorite_pressure_bucket"]["rows"], 240)
        self.assertAlmostEqual(by_col["model_vs_market_favorite_pressure_bucket"]["mean_market_favorite_prob"], 0.74)
        self.assertAlmostEqual(by_col["model_vs_market_favorite_pressure_bucket"]["mean_model_favorite_prob"], 0.57)
        self.assertEqual(by_col["model_vs_market_favorite_pressure_bucket"]["market_favorite_hit_rate"], 1.0)
        self.assertGreater(by_col["model_vs_market_favorite_pressure_bucket"]["model_minus_market_log_loss"], 0)
        self.assertEqual(by_col["model_vs_market_favorite_pressure_bucket"]["years_model_lags_market_log_loss"], 3)

    def test_model_market_agreement_segments_scores_probability_sizing_not_pick_overrides(self) -> None:
        import pandas as pd

        rows = []
        for year in [2022, 2023, 2024]:
            # Market and model both pick p1, but the model is more overconfident and
            # should be diagnosed separately from true pick overrides.
            for i in range(50):
                rows.append({
                    "date": f"{year}-02-{(i % 9) + 1:02d}",
                    "result": 1 if i % 2 else 0,
                    "model_p1": 0.82,
                    "implied_p1_no_vig": 0.60,
                    "series": "ATP250",
                    "round_group": "early",
                })
            # Disagreement rows in the same segment must be excluded from the
            # agreement diagnostic and counted for auditability.
            for i in range(10):
                rows.append({
                    "date": f"{year}-03-{(i % 9) + 1:02d}",
                    "result": 0,
                    "model_p1": 0.62,
                    "implied_p1_no_vig": 0.48,
                    "series": "ATP250",
                    "round_group": "early",
                })

        diagnostics = model_market_agreement_segments(
            pd.DataFrame(rows),
            "model_p1",
            segment_cols=["series", "round_group"],
            min_rows=120,
        )

        by_col = {row["segment_col"]: row for row in diagnostics}
        self.assertEqual(by_col["series"]["agreement_rows"], 150)
        self.assertEqual(by_col["series"]["disagreement_rows_excluded"], 30)
        self.assertEqual(by_col["series"]["model_pick_accuracy"], by_col["series"]["market_pick_accuracy"])
        self.assertGreater(by_col["series"]["model_minus_market_log_loss"], 0)
        self.assertGreater(by_col["series"]["model_minus_market_brier"], 0)
        self.assertEqual(by_col["series"]["year_count"], 3)

    def test_disagreement_margin_segments_scores_only_model_market_overrides_by_gap_bucket(self) -> None:
        import pandas as pd

        rows = []
        for year in [2022, 2023, 2024]:
            for i in range(60):
                rows.append({
                    "date": f"{year}-02-{(i % 9) + 1:02d}",
                    "result": 1 if i % 3 else 0,
                    "model_p1": 0.58,
                    "implied_p1_no_vig": 0.47,
                })
            for i in range(20):
                rows.append({
                    "date": f"{year}-03-{(i % 9) + 1:02d}",
                    "result": 1,
                    "model_p1": 0.62,
                    "implied_p1_no_vig": 0.51,
                })

        diagnostics = disagreement_margin_segments(pd.DataFrame(rows), "model_p1", min_rows=100)

        by_col = {row["segment_col"]: row for row in diagnostics}
        self.assertEqual(by_col["model_market_gap_bucket"]["segment"], "medium_gap_7_12pct")
        self.assertEqual(by_col["model_market_gap_bucket"]["disagreement_rows"], 180)
        self.assertEqual(by_col["model_market_gap_bucket"]["agreement_rows_excluded"], 60)
        self.assertEqual(by_col["model_market_direction"]["segment"], "model_prefers_p1_market_prefers_p2")
        self.assertEqual(by_col["model_market_gap_bucket"]["year_count"], 3)

    def test_segment_errors_adds_year_stability_diagnostics(self) -> None:
        import pandas as pd

        rows = []
        for year in [2022, 2023, 2024]:
            for i in range(90):
                rows.append({
                    "date": f"{year}-01-0{(i % 9) + 1}",
                    "result": 1 if i % 2 else 0,
                    "model_p1": 0.70 if i % 2 else 0.20,
                    "implied_p1_no_vig": 0.50,
                    "surface": "Grass",
                    "series": "ATP250",
                    "court": "Outdoor",
                    "round_group": "early",
                    "round": "1st Round",
                    "is_early_round": 1,
                    "any_top10": 0,
                    "both_top10": 0,
                    "early_after_title_p1": 0,
                    "early_after_title_p2": 0,
                })

        grass = next(row for row in segment_errors(pd.DataFrame(rows), "model_p1") if row["segment_col"] == "surface")

        self.assertEqual(grass["years"], [2022, 2023, 2024])
        self.assertEqual(grass["year_count"], 3)
        self.assertEqual(grass["min_year_rows"], 90)
        self.assertEqual(grass["years_model_beats_market_log_loss"], 3)
        self.assertEqual(grass["years_model_beats_market_brier"], 3)

    def test_no_lookahead_disagreement_margin_blend_uses_prior_bucket_weights_only(self) -> None:
        import pandas as pd

        rows = []
        # 2022 teaches the medium-gap/model-prefers-p1 override bucket is harmful;
        # the 2023 same-bucket rows should be shrunk to market, while the first year
        # and agreement rows keep the model probability.
        for i in range(80):
            rows.append({
                "date": "2022-01-01",
                "result": 0,
                "model_p1": 0.58,
                "implied_p1_no_vig": 0.47,
            })
        for i in range(40):
            rows.append({
                "date": "2022-01-02",
                "result": 1,
                "model_p1": 0.64,
                "implied_p1_no_vig": 0.70,
            })
        for i in range(20):
            rows.append({
                "date": "2023-01-01",
                "result": 0,
                "model_p1": 0.58,
                "implied_p1_no_vig": 0.47,
            })
        for i in range(10):
            rows.append({
                "date": "2023-01-02",
                "result": 1,
                "model_p1": 0.64,
                "implied_p1_no_vig": 0.70,
            })

        diagnostic = no_lookahead_disagreement_margin_blend_diagnostic(
            pd.DataFrame(rows),
            "model_p1",
            candidate_weights=[0.0, 1.0],
            min_train_rows=50,
        )

        self.assertEqual(diagnostic["routed_rows"], 20)
        year_2022, year_2023 = diagnostic["yearly"]
        self.assertEqual(year_2022["routed_rows"], 0)
        self.assertEqual(year_2023["routed_rows"], 20)
        self.assertEqual(year_2023["selected_segments"][0]["selected_model_weight"], 0.0)
        self.assertLess(
            year_2023["blended_metrics"]["log_loss"],
            year_2023["model_metrics"]["log_loss"],
        )
        self.assertLess(diagnostic["overall_blended_minus_model_log_loss"], 0.0)

    def test_agreement_sizing_blend_uses_prior_year_agreement_buckets_only(self) -> None:
        import pandas as pd

        rows = []
        # 2022 establishes that when model and market agree on p1 but the model is
        # much more confident, shrinking to market is better. 2023 can use that
        # prior-year bucket; disagreement rows and unsupported agreement buckets stay untouched.
        for year in [2022, 2023]:
            for i in range(20):
                rows.append({
                    "date": f"{year}-01-{(i % 9) + 1:02d}",
                    "result": 0,
                    "model_p1": 0.82,
                    "implied_p1_no_vig": 0.56,
                })
            for i in range(10):
                rows.append({
                    "date": f"{year}-02-{(i % 9) + 1:02d}",
                    "result": 1,
                    "model_p1": 0.65,
                    "implied_p1_no_vig": 0.45,
                })

        diagnostic = no_lookahead_agreement_sizing_blend_diagnostic(
            pd.DataFrame(rows),
            "model_p1",
            candidate_weights=[0.0, 1.0],
            min_train_rows=15,
        )

        self.assertEqual(diagnostic["routed_rows"], 20)
        self.assertEqual(diagnostic["yearly"][0]["routed_rows"], 0)
        self.assertEqual(diagnostic["yearly"][1]["routed_rows"], 20)
        self.assertEqual(diagnostic["yearly"][1]["selected_segments"][0]["selected_model_weight"], 0.0)
        self.assertLess(diagnostic["yearly"][1]["blended_metrics"]["log_loss"], diagnostic["yearly"][1]["model_metrics"]["log_loss"])
        self.assertLess(diagnostic["overall_blended_minus_model_log_loss"], 0.0)

    def test_no_lookahead_blend_weight_diagnostic_selects_weights_from_prior_years_only(self) -> None:
        import pandas as pd

        rows = []
        # 2022 shows the feature model is better than market, so 2023 may blend toward it.
        for i in range(20):
            result = 1 if i % 2 else 0
            rows.append({
                "date": f"2022-01-{(i % 9) + 1:02d}",
                "result": result,
                "model_p1": 0.80 if result else 0.20,
                "implied_p1_no_vig": 0.55 if result else 0.45,
            })
        # 2023 has the same pattern; the policy should use the prior-year-selected model weight.
        for i in range(20):
            result = 1 if i % 2 else 0
            rows.append({
                "date": f"2023-01-{(i % 9) + 1:02d}",
                "result": result,
                "model_p1": 0.80 if result else 0.20,
                "implied_p1_no_vig": 0.55 if result else 0.45,
            })
        # 2024 reverses; the weight is still selected from prior OOS rows, not 2024 leakage.
        for i in range(20):
            result = 1 if i % 2 else 0
            rows.append({
                "date": f"2024-01-{(i % 9) + 1:02d}",
                "result": result,
                "model_p1": 0.20 if result else 0.80,
                "implied_p1_no_vig": 0.55 if result else 0.45,
            })

        diagnostic = no_lookahead_blend_weight_diagnostic(
            pd.DataFrame(rows),
            "model_p1",
            candidate_weights=[0.0, 0.5, 1.0],
            min_train_rows=10,
        )

        self.assertEqual(diagnostic["model"], "model_p1")
        self.assertEqual(diagnostic["yearly"][0]["year"], 2022)
        self.assertEqual(diagnostic["yearly"][0]["selected_model_weight"], 0.0)
        self.assertEqual(diagnostic["yearly"][1]["year"], 2023)
        self.assertEqual(diagnostic["yearly"][1]["selected_model_weight"], 1.0)
        self.assertEqual(diagnostic["yearly"][2]["year"], 2024)
        self.assertEqual(diagnostic["yearly"][2]["selected_model_weight"], 1.0)
        self.assertGreater(diagnostic["yearly"][2]["blended_metrics"]["log_loss"], diagnostic["yearly"][2]["market_metrics"]["log_loss"])
        self.assertGreater(diagnostic["overall_blended_metrics"]["log_loss"], diagnostic["overall_oracle_best_weight_metrics"]["log_loss"])

    def test_disagreement_fallback_routing_uses_prior_year_segments_only(self) -> None:
        import pandas as pd

        rows = []
        # 2022 establishes an ATP250|early disagreement segment where the model
        # flips away from the market and loses badly; the 2023 policy may use it.
        for i in range(12):
            result = 0 if i < 9 else 1
            rows.append({
                "date": f"2022-02-{(i % 9) + 1:02d}",
                "result": result,
                "model_p1": 0.65,
                "implied_p1_no_vig": 0.45,
                "series": "ATP250",
                "round_group": "early",
            })
        for i in range(12):
            result = 0 if i < 9 else 1
            rows.append({
                "date": f"2023-02-{(i % 9) + 1:02d}",
                "result": result,
                "model_p1": 0.65,
                "implied_p1_no_vig": 0.45,
                "series": "ATP250",
                "round_group": "early",
            })
        # Different 2023 segment should stay untouched even when model disagrees.
        for i in range(12):
            rows.append({
                "date": f"2023-03-{(i % 9) + 1:02d}",
                "result": 1,
                "model_p1": 0.65,
                "implied_p1_no_vig": 0.45,
                "series": "ATP500",
                "round_group": "early",
            })

        diagnostic = disagreement_fallback_routing_diagnostic(
            pd.DataFrame(rows),
            "model_p1",
            segment_groups=[("series", "round_group")],
            min_train_rows=10,
            min_years=1,
            min_stable_year_share=1.0,
        )

        self.assertEqual(diagnostic["model"], "model_p1")
        self.assertEqual(diagnostic["routed_rows"], 12)
        self.assertEqual(diagnostic["yearly"][0]["year"], 2022)
        self.assertEqual(diagnostic["yearly"][0]["routed_rows"], 0)
        self.assertEqual(diagnostic["yearly"][1]["year"], 2023)
        self.assertEqual(diagnostic["yearly"][1]["routed_rows"], 12)
        self.assertLess(diagnostic["overall_routed_metrics"]["log_loss"], diagnostic["overall_original_metrics"]["log_loss"])
        self.assertIn("ATP250 | early", diagnostic["yearly"][1]["segments_flagged"])

    def test_interaction_segment_errors_finds_stable_two_way_contexts(self) -> None:
        import pandas as pd

        rows = []
        for year in [2022, 2023, 2024]:
            for i in range(90):
                rows.append({
                    "date": f"{year}-02-{(i % 9) + 1:02d}",
                    "result": 1 if i % 2 else 0,
                    "model_p1": 0.70 if i % 2 else 0.20,
                    "implied_p1_no_vig": 0.50,
                    "surface": "Grass",
                    "series": "ATP250",
                    "court": "Outdoor",
                    "round_group": "early",
                    "round": "1st Round",
                    "is_early_round": 1,
                    "any_top10": 0,
                    "both_top10": 0,
                    "early_after_title_p1": 0,
                    "early_after_title_p2": 0,
                    "rank_diff": 60,
                    "rest_diff": -4,
                    "matches_last7_diff": 3,
                    "p1_surface_switch": 1,
                    "p2_surface_switch": 0,
                    "p1_title_within_14": 0,
                    "p2_title_within_14": 0,
                    "p1_final_within_7": 0,
                    "p2_final_within_7": 0,
                })

        diagnostics = interaction_segment_errors(
            pd.DataFrame(rows),
            "model_p1",
            interaction_pairs=[("series", "rank_diff_bucket")],
            min_rows=80,
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["segment_col"], "series__rank_diff_bucket")
        self.assertEqual(diagnostics[0]["segment"], "ATP250 | p1_much_lower_rank")
        self.assertEqual(diagnostics[0]["rows"], 270)
        self.assertEqual(diagnostics[0]["year_count"], 3)
        self.assertEqual(diagnostics[0]["years_model_beats_market_log_loss"], 3)
        self.assertLess(diagnostics[0]["model_minus_market_log_loss"], 0)

    def test_multivariate_segment_errors_finds_three_way_contexts(self) -> None:
        import pandas as pd

        rows = []
        for year in [2022, 2023, 2024]:
            for i in range(85):
                rows.append({
                    "date": f"{year}-04-{(i % 9) + 1:02d}",
                    "result": 1 if i % 2 else 0,
                    "model_p1": 0.68 if i % 2 else 0.30,
                    "implied_p1_no_vig": 0.50,
                    "series": "ATP250",
                    "round_group": "early",
                    "rank_diff": 70,
                    "surface": "Clay",
                })

        diagnostics = multivariate_segment_errors(
            pd.DataFrame(rows),
            "model_p1",
            segment_groups=[("series", "round_group", "rank_diff_bucket")],
            min_rows=80,
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["segment_col"], "series__round_group__rank_diff_bucket")
        self.assertEqual(diagnostics[0]["segment"], "ATP250 | early | p1_much_lower_rank")
        self.assertEqual(diagnostics[0]["segment_columns"], ["series", "round_group", "rank_diff_bucket"])
        self.assertEqual(diagnostics[0]["rows"], 255)
        self.assertEqual(diagnostics[0]["year_count"], 3)
        self.assertLess(diagnostics[0]["model_minus_market_log_loss"], 0)

    def test_multivariate_segment_errors_handles_missing_bucket_values(self) -> None:
        import pandas as pd
        import numpy as np

        rows = []
        for i in range(130):
            rows.append({
                "date": "2024-05-01",
                "result": 1 if i % 2 else 0,
                "model_p1": 0.60 if i % 2 else 0.40,
                "implied_p1_no_vig": 0.50,
                "series": "ATP250",
                "round_group": "early",
                "rank_diff": np.nan,
            })

        diagnostics = multivariate_segment_errors(
            pd.DataFrame(rows),
            "model_p1",
            segment_groups=[("series", "round_group", "rank_diff_bucket")],
            min_rows=120,
        )

        self.assertEqual(diagnostics[0]["segment"], "ATP250 | early | nan")
        self.assertEqual(diagnostics[0]["rows"], 130)

    def test_interaction_segment_errors_handles_single_class_year_buckets(self) -> None:
        import pandas as pd

        rows = []
        for year, result in [(2022, 1), (2023, 0), (2024, 1)]:
            for i in range(90):
                rows.append({
                    "date": f"{year}-03-{(i % 9) + 1:02d}",
                    "result": result,
                    "model_p1": 0.80 if result else 0.20,
                    "implied_p1_no_vig": 0.50,
                    "surface": "Hard",
                    "series": "ATP250",
                    "round_group": "early",
                    "rank_diff": 60,
                })

        diagnostics = interaction_segment_errors(
            pd.DataFrame(rows),
            "model_p1",
            interaction_pairs=[("series", "rank_diff_bucket")],
            min_rows=80,
        )

        self.assertEqual(diagnostics[0]["year_count"], 3)
        self.assertEqual(len(diagnostics[0]["yearly_model_minus_market"]), 3)

    def test_summarize_segment_strengths_keeps_only_stable_market_beating_segments(self) -> None:
        segment_rows = [
            {
                "segment_col": "surface",
                "segment": "Clay",
                "rows": 300,
                "accuracy": 0.70,
                "log_loss": 0.55,
                "market_log_loss": 0.58,
                "brier": 0.18,
                "market_brier": 0.20,
                "model_minus_market_log_loss": -0.03,
                "model_minus_market_brier": -0.02,
                "year_count": 3,
                "min_year_rows": 100,
                "years_model_beats_market_log_loss": 3,
                "years_model_beats_market_brier": 3,
            },
            {
                "segment_col": "round_group",
                "segment": "late",
                "rows": 90,
                "accuracy": 0.72,
                "log_loss": 0.52,
                "market_log_loss": 0.56,
                "brier": 0.17,
                "market_brier": 0.19,
                "model_minus_market_log_loss": -0.04,
                "model_minus_market_brier": -0.02,
                "year_count": 3,
                "min_year_rows": 30,
                "years_model_beats_market_log_loss": 3,
                "years_model_beats_market_brier": 3,
            },
            {
                "segment_col": "market_prob_bucket",
                "segment": "near_pickem",
                "rows": 450,
                "accuracy": 0.55,
                "log_loss": 0.68,
                "market_log_loss": 0.69,
                "brier": 0.24,
                "market_brier": 0.25,
                "model_minus_market_log_loss": -0.01,
                "model_minus_market_brier": -0.01,
                "year_count": 3,
                "min_year_rows": 150,
                "years_model_beats_market_log_loss": 1,
                "years_model_beats_market_brier": 3,
            },
            {
                "segment_col": "series",
                "segment": "ATP250",
                "rows": 400,
                "accuracy": 0.61,
                "log_loss": 0.62,
                "market_log_loss": 0.59,
                "brier": 0.22,
                "market_brier": 0.20,
                "model_minus_market_log_loss": 0.03,
                "model_minus_market_brier": 0.02,
                "year_count": 3,
                "min_year_rows": 120,
                "years_model_beats_market_log_loss": 0,
                "years_model_beats_market_brier": 0,
            },
        ]

        summary = summarize_segment_strengths(segment_rows, min_rows=100, top_n=3, min_years=3)

        self.assertEqual(summary["min_rows"], 100)
        self.assertEqual(summary["min_years"], 3)
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["excluded_low_sample_segments"], 1)
        self.assertEqual(summary["excluded_unstable_segments"], 1)
        self.assertEqual(summary["top_segments"][0]["segment"], "Clay")
        self.assertAlmostEqual(summary["top_segments"][0]["weighted_log_loss_improvement"], 9.0)

    def test_model_market_disagreement_segments_focuses_on_model_pick_flips(self) -> None:
        import pandas as pd

        rows = []
        for year in [2022, 2023, 2024]:
            for i in range(70):
                rows.append({
                    "date": f"{year}-06-{(i % 9) + 1:02d}",
                    "result": 1,
                    "model_p1": 0.62,
                    "implied_p1_no_vig": 0.44,
                    "series": "ATP250",
                    "round_group": "early",
                    "surface": "Grass",
                    "rank_diff": -20,
                    "rest_diff": 0,
                    "matches_last7_diff": 0,
                    "p1_surface_switch": 0,
                    "p2_surface_switch": 0,
                    "p1_title_within_14": 0,
                    "p2_title_within_14": 0,
                    "p1_final_within_7": 0,
                    "p2_final_within_7": 0,
                })
                rows.append({
                    "date": f"{year}-06-{(i % 9) + 1:02d}",
                    "result": 0,
                    "model_p1": 0.38,
                    "implied_p1_no_vig": 0.56,
                    "series": "ATP250",
                    "round_group": "early",
                    "surface": "Grass",
                    "rank_diff": 20,
                    "rest_diff": 0,
                    "matches_last7_diff": 0,
                    "p1_surface_switch": 0,
                    "p2_surface_switch": 0,
                    "p1_title_within_14": 0,
                    "p2_title_within_14": 0,
                    "p1_final_within_7": 0,
                    "p2_final_within_7": 0,
                })
            for i in range(20):
                rows.append({
                    "date": f"{year}-07-{(i % 9) + 1:02d}",
                    "result": 1,
                    "model_p1": 0.70,
                    "implied_p1_no_vig": 0.80,
                    "series": "ATP250",
                    "round_group": "early",
                    "surface": "Grass",
                })

        diagnostics = model_market_disagreement_segments(
            pd.DataFrame(rows),
            "model_p1",
            segment_cols=["series", "round_group"],
            min_rows=100,
        )

        self.assertEqual(len(diagnostics), 2)
        by_col = {row["segment_col"]: row for row in diagnostics}
        self.assertEqual(by_col["series"]["rows"], 420)
        self.assertEqual(by_col["series"]["disagreement_rows"], 420)
        self.assertEqual(by_col["series"]["agreement_rows_excluded"], 60)
        self.assertEqual(by_col["series"]["year_count"], 3)
        self.assertEqual(by_col["series"]["years_model_beats_market_log_loss"], 3)
        self.assertLess(by_col["series"]["model_minus_market_log_loss"], 0)
        self.assertGreater(by_col["series"]["model_pick_accuracy"], by_col["series"]["market_pick_accuracy"])

    def test_interaction_model_market_disagreement_segments_finds_two_way_override_contexts(self) -> None:
        import pandas as pd

        rows = []
        for year in [2022, 2023, 2024]:
            for i in range(60):
                rows.append({
                    "date": f"{year}-08-{(i % 9) + 1:02d}",
                    "result": 1,
                    "model_p1": 0.62,
                    "implied_p1_no_vig": 0.44,
                    "series": "ATP250",
                    "round_group": "early",
                    "surface": "Clay",
                    "rank_diff": -30,
                    "rest_diff": 0,
                    "matches_last7_diff": 0,
                    "p1_surface_switch": 0,
                    "p2_surface_switch": 0,
                    "p1_title_within_14": 0,
                    "p2_title_within_14": 0,
                    "p1_final_within_7": 0,
                    "p2_final_within_7": 0,
                })
                rows.append({
                    "date": f"{year}-08-{(i % 9) + 1:02d}",
                    "result": 0,
                    "model_p1": 0.38,
                    "implied_p1_no_vig": 0.56,
                    "series": "ATP250",
                    "round_group": "early",
                    "surface": "Clay",
                    "rank_diff": 30,
                    "rest_diff": 0,
                    "matches_last7_diff": 0,
                    "p1_surface_switch": 0,
                    "p2_surface_switch": 0,
                    "p1_title_within_14": 0,
                    "p2_title_within_14": 0,
                    "p1_final_within_7": 0,
                    "p2_final_within_7": 0,
                })
            for i in range(25):
                rows.append({
                    "date": f"{year}-09-{(i % 9) + 1:02d}",
                    "result": 1,
                    "model_p1": 0.70,
                    "implied_p1_no_vig": 0.80,
                    "series": "ATP250",
                    "round_group": "early",
                    "surface": "Clay",
                })

        diagnostics = interaction_model_market_disagreement_segments(
            pd.DataFrame(rows),
            "model_p1",
            interaction_pairs=[("series", "round_group")],
            min_rows=100,
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["segment_col"], "series__round_group")
        self.assertEqual(diagnostics[0]["segment"], "ATP250 | early")
        self.assertEqual(diagnostics[0]["rows"], 360)
        self.assertEqual(diagnostics[0]["disagreement_rows"], 360)
        self.assertEqual(diagnostics[0]["agreement_rows_excluded"], 75)
        self.assertEqual(diagnostics[0]["year_count"], 3)
        self.assertEqual(diagnostics[0]["years_model_beats_market_log_loss"], 3)
        self.assertGreater(diagnostics[0]["model_pick_accuracy"], diagnostics[0]["market_pick_accuracy"])

    def test_summarize_segment_weaknesses_keeps_only_stable_market_lagging_segments(self) -> None:
        segment_rows = [
            {
                "segment_col": "market_prob_bucket",
                "segment": "heavy_p1_favorite",
                "rows": 420,
                "accuracy": 0.63,
                "log_loss": 0.71,
                "market_log_loss": 0.58,
                "brier": 0.24,
                "market_brier": 0.19,
                "model_minus_market_log_loss": 0.13,
                "model_minus_market_brier": 0.05,
                "year_count": 5,
                "min_year_rows": 60,
                "years_model_lags_market_log_loss": 5,
                "years_model_lags_market_brier": 4,
            },
            {
                "segment_col": "surface",
                "segment": "Clay",
                "rows": 500,
                "accuracy": 0.66,
                "log_loss": 0.62,
                "market_log_loss": 0.60,
                "brier": 0.21,
                "market_brier": 0.20,
                "model_minus_market_log_loss": 0.02,
                "model_minus_market_brier": 0.01,
                "year_count": 5,
                "min_year_rows": 80,
                "years_model_lags_market_log_loss": 2,
                "years_model_lags_market_brier": 5,
            },
            {
                "segment_col": "round_group",
                "segment": "early",
                "rows": 90,
                "accuracy": 0.61,
                "log_loss": 0.65,
                "market_log_loss": 0.60,
                "brier": 0.22,
                "market_brier": 0.20,
                "model_minus_market_log_loss": 0.05,
                "model_minus_market_brier": 0.02,
                "year_count": 3,
                "min_year_rows": 30,
                "years_model_lags_market_log_loss": 3,
                "years_model_lags_market_brier": 3,
            },
            {
                "segment_col": "series",
                "segment": "ATP500",
                "rows": 300,
                "accuracy": 0.69,
                "log_loss": 0.55,
                "market_log_loss": 0.58,
                "brier": 0.18,
                "market_brier": 0.20,
                "model_minus_market_log_loss": -0.03,
                "model_minus_market_brier": -0.02,
                "year_count": 3,
                "min_year_rows": 100,
                "years_model_lags_market_log_loss": 0,
                "years_model_lags_market_brier": 0,
            },
        ]

        summary = summarize_segment_weaknesses(segment_rows, min_rows=100, top_n=2, min_years=3)

        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["excluded_low_sample_segments"], 1)
        self.assertEqual(summary["excluded_unstable_segments"], 1)
        self.assertEqual(summary["top_segments"][0]["segment"], "heavy_p1_favorite")
        self.assertAlmostEqual(summary["top_segments"][0]["weighted_log_loss_damage"], 54.6)
        self.assertEqual(summary["top_segments"][0]["hypothesis_label"], "stable_market_lagging_segment")

    def test_summarize_probability_quality_tradeoffs_does_not_warn_on_accuracy_tie(self) -> None:
        rows = [
            {
                "model": "market_no_vig",
                "accuracy": 0.6811594202898551,
                "log_loss": 0.5823648180463947,
                "brier": 0.1995997417581301,
                "calibration_error_metrics": {"expected_calibration_error": 0.03198951399252753},
            },
            {
                "model": "market_bin_recalibrated",
                "accuracy": 0.6811594202898551,
                "log_loss": 0.5818499979469696,
                "brier": 0.1994167161959878,
                "calibration_error_metrics": {"expected_calibration_error": 0.029632928158857347},
            },
        ]

        summary = summarize_probability_quality_tradeoffs(rows, baseline_model="market_no_vig")

        self.assertEqual(summary["best_by_log_loss"]["model"], "market_bin_recalibrated")
        self.assertIsNone(summary["warning"])

    def test_shrinkage_sweep_reports_best_no_lookahead_recalibration(self) -> None:
        import pandas as pd

        train = pd.DataFrame({
            "result": [1, 1, 1, 1, 0, 0, 0, 0],
            "market_p1": [0.41, 0.42, 0.43, 0.44, 0.61, 0.62, 0.63, 0.64],
        })
        test = pd.DataFrame({
            "result": [1, 1, 0, 0],
            "market_p1": [0.42, 0.44, 0.62, 0.64],
        })

        sweep = evaluate_bin_recalibration_shrinkage_sweep(
            train,
            test,
            prob_col="market_p1",
            bins=10,
            min_rows=4,
            shrink_values=[0.0, 0.5, 1.0],
        )

        self.assertEqual(sweep["source_model"], "market_no_vig")
        self.assertEqual([row["shrink"] for row in sweep["candidates"]], [0.0, 0.5, 1.0])
        self.assertEqual(sweep["best_by_log_loss"]["shrink"], 1.0)
        self.assertLess(sweep["best_by_log_loss"]["log_loss"], sweep["baseline"]["log_loss"])
        self.assertLess(sweep["best_by_brier"]["brier"], sweep["baseline"]["brier"])

    def test_market_favorite_pressure_blend_uses_prior_oos_bucket_weights(self) -> None:
        import pandas as pd

        rows = []
        for year in [2022, 2023]:
            for i in range(6):
                rows.append({
                    "date": f"{year}-01-{i + 1:02d}",
                    "result": 1,
                    "model_p1": 0.56,
                    "implied_p1_no_vig": 0.78,
                })
            for i in range(6):
                rows.append({
                    "date": f"{year}-02-{i + 1:02d}",
                    "result": 0,
                    "model_p1": 0.44,
                    "implied_p1_no_vig": 0.22,
                })

        diagnostic = no_lookahead_market_favorite_pressure_blend_diagnostic(
            pd.DataFrame(rows),
            model_prob_col="model_p1",
            candidate_weights=[0.0, 1.0],
            min_train_rows=10,
        )

        self.assertEqual(diagnostic["blended_probability_col"], "model_p1_market_favorite_pressure_blend")
        self.assertEqual(diagnostic["yearly"][0]["prior_oos_pressure_rows"], 0)
        self.assertEqual(diagnostic["yearly"][1]["prior_oos_pressure_rows"], 12)
        self.assertEqual(diagnostic["yearly"][1]["selected_segments"][0]["selected_model_weight"], 0.0)
        self.assertEqual(diagnostic["routed_rows"], 12)
        self.assertLess(
            diagnostic["overall_blended_metrics"]["log_loss"],
            diagnostic["overall_model_metrics"]["log_loss"],
        )

    def test_summarize_shrinkage_sweeps_reports_multi_year_stability(self) -> None:
        sweeps = [
            {
                "year": 2022,
                "baseline": {"shrink": 0.0, "log_loss": 0.60, "brier": 0.21},
                "best_by_log_loss": {"shrink": 1.0, "log_loss": 0.58, "brier": 0.20},
                "best_by_brier": {"shrink": 0.75, "log_loss": 0.581, "brier": 0.199},
            },
            {
                "year": 2023,
                "baseline": {"shrink": 0.0, "log_loss": 0.59, "brier": 0.205},
                "best_by_log_loss": {"shrink": 0.5, "log_loss": 0.592, "brier": 0.206},
                "best_by_brier": {"shrink": 0.5, "log_loss": 0.592, "brier": 0.203},
            },
        ]

        summary = summarize_shrinkage_sweeps(sweeps)

        self.assertEqual(summary["years"], [2022, 2023])
        self.assertEqual(summary["best_log_loss_shrink_counts"], {"1.0": 1, "0.5": 1})
        self.assertAlmostEqual(summary["avg_log_loss_delta_vs_baseline"], -0.009)
        self.assertAlmostEqual(summary["avg_brier_delta_vs_baseline"], -0.0065)
        self.assertEqual(summary["years_improved_log_loss"], 1)
        self.assertEqual(summary["years_improved_brier"], 2)
        self.assertEqual(summary["recommendation"], "diagnostic_only_mixed_or_insufficient_years")


if __name__ == "__main__":
    unittest.main()
