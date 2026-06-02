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
    summarize_calibration_diagnostics,
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
