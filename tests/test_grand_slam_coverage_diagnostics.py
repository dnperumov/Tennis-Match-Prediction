#!/usr/bin/env python3
"""Focused regression tests for Grand Slam benchmark coverage diagnostics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from advanced_feature_model_research import (  # noqa: E402
    calibration_bins,
    grand_slam_coverage_diagnostics,
    grand_slam_tournament_folds,
)


class GrandSlamCoverageDiagnosticsTest(unittest.TestCase):
    def test_reports_expected_2022_paper_gap_by_tournament(self) -> None:
        gs_rows = pd.DataFrame(
            [
                {"year": 2022, "tournament": "Australian Open", "date": "2022-01-17"},
                {"year": 2022, "tournament": "Australian Open", "date": "2022-01-18"},
                {"year": 2022, "tournament": "Roland Garros", "date": "2022-05-22"},
                {"year": 2021, "tournament": "Wimbledon", "date": "2021-06-28"},
            ]
        )
        evaluated = [{"year": 2022, "tournament": "Australian Open", "rows": 2}]
        skipped = [{"year": 2022, "tournament": "Roland Garros", "test_rows": 1}]

        diag = grand_slam_coverage_diagnostics(gs_rows, [2022], evaluated, skipped)

        self.assertEqual(diag["paper_expected_rows_by_year"]["2022"], 293)
        self.assertEqual(diag["available_rows_by_year"]["2022"], 3)
        self.assertEqual(diag["evaluated_rows_by_year"]["2022"], 2)
        self.assertEqual(diag["missing_vs_paper_by_year"]["2022"], 291)
        self.assertEqual(diag["available_missing_vs_paper_by_year"]["2022"], 290)
        self.assertEqual(diag["evaluated_tournament_count_by_year"]["2022"], 1)
        self.assertEqual(diag["skipped_tournament_count_by_year"]["2022"], 1)
        self.assertEqual(diag["coverage_ratio_vs_paper_by_year"]["2022"], 2 / 293)
        self.assertEqual(diag["available_coverage_ratio_vs_paper_by_year"]["2022"], 3 / 293)
        self.assertEqual(diag["tournaments_by_year"]["2022"]["Australian Open"]["available_rows"], 2)
        self.assertEqual(diag["tournaments_by_year"]["2022"]["Australian Open"]["status"], "evaluated")
        self.assertEqual(diag["tournaments_by_year"]["2022"]["Roland Garros"]["status"], "skipped")

    def test_tournament_folds_include_all_rows_from_same_grand_slam_event(self) -> None:
        gs_rows = pd.DataFrame(
            [
                {"year": 2021, "tournament": "Wimbledon", "date": "2021-06-28", "result": 1},
                {"year": 2022, "tournament": "Australian Open", "date": "2022-01-17", "result": 1},
                {"year": 2022, "tournament": "Australian Open", "date": "2022-01-18", "result": 0},
            ]
        )

        folds = list(grand_slam_tournament_folds(gs_rows, [2022]))

        self.assertEqual(len(folds), 1)
        fold = folds[0]
        self.assertEqual(fold["year"], 2022)
        self.assertEqual(fold["tournament"], "Australian Open")
        self.assertEqual(len(fold["test"]), 2)
        self.assertEqual(len(fold["train"]), 1)
        self.assertLess(fold["train"]["date"].max(), fold["test"]["date"].min())
    def test_calibration_bins_report_bin_counts_error_and_direction(self) -> None:
        preds = pd.DataFrame(
            {
                "result": [0, 1, 1, 0, 1, 1],
                "model_p1": [0.10, 0.20, 0.35, 0.55, 0.75, 0.90],
            }
        )

        bins = calibration_bins(preds, "model_p1", bins=4)

        self.assertEqual([b["bin"] for b in bins], [0, 1, 2, 3])
        self.assertEqual([b["rows"] for b in bins], [2, 1, 1, 2])
        self.assertAlmostEqual(bins[0]["mean_prob"], 0.15)
        self.assertAlmostEqual(bins[0]["actual_rate"], 0.5)
        self.assertAlmostEqual(bins[0]["calibration_error"], -0.35)
        self.assertAlmostEqual(bins[3]["mean_prob"], 0.825)
        self.assertAlmostEqual(bins[3]["actual_rate"], 1.0)
        self.assertAlmostEqual(bins[3]["calibration_error"], -0.175)


if __name__ == "__main__":
    unittest.main()
