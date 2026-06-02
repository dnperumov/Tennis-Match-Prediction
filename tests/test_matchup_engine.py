#!/usr/bin/env python3
"""Tests for the lightweight matchup prediction engine."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ml.predict import confidence_label, predict_matchup  # noqa: E402


def _sample_export(tmp_path: Path) -> Path:
    path = tmp_path / "export.csv"
    pd.DataFrame(
        [
            {
                "match_date": "2026-01-01",
                "tourney_name": "Test Open",
                "surface": "Clay",
                "round": "R32",
                "winner_name": "Player A",
                "loser_name": "Player B",
                "winner_total_points_won_pct": 56.0,
                "loser_total_points_won_pct": 44.0,
                "winner_return_points_won_pct": 42.0,
                "loser_return_points_won_pct": 31.0,
                "winner_serve_first_won_pct": 75.0,
                "loser_serve_first_won_pct": 64.0,
            },
            {
                "match_date": "2026-01-02",
                "tourney_name": "Test Open",
                "surface": "Clay",
                "round": "R16",
                "winner_name": "Player A",
                "loser_name": "Player C",
                "winner_total_points_won_pct": 53.0,
                "loser_total_points_won_pct": 47.0,
                "winner_return_points_won_pct": 39.0,
                "loser_return_points_won_pct": 34.0,
                "winner_serve_first_won_pct": 72.0,
                "loser_serve_first_won_pct": 68.0,
            },
            {
                "match_date": "2026-01-03",
                "tourney_name": "Test Open",
                "surface": "Hard",
                "round": "R32",
                "winner_name": "Player B",
                "loser_name": "Player C",
                "winner_total_points_won_pct": 52.0,
                "loser_total_points_won_pct": 48.0,
                "winner_return_points_won_pct": 36.0,
                "loser_return_points_won_pct": 33.0,
                "winner_serve_first_won_pct": 70.0,
                "loser_serve_first_won_pct": 66.0,
            },
        ]
    ).to_csv(path, index=False)
    return path


class MatchupEngineTest(unittest.TestCase):
    def test_matchup_probabilities_are_bounded_and_sum_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pred = predict_matchup(
                "Player A",
                "Player B",
                context={"surface": "Clay", "tournament": "Test Open"},
                export_path=_sample_export(tmp_path),
                metrics_path=tmp_path / "missing_metrics.json",
                advanced_report_path=tmp_path / "missing_advanced.json",
            ).to_dict()

        self.assertGreaterEqual(pred["p1_probability"], 0.03)
        self.assertLessEqual(pred["p1_probability"], 0.97)
        self.assertGreaterEqual(pred["p2_probability"], 0.03)
        self.assertLessEqual(pred["p2_probability"], 0.97)
        self.assertLess(abs(pred["p1_probability"] + pred["p2_probability"] - 1.0), 1e-6)
        self.assertEqual(pred["pick"], "Player A")

    def test_market_edge_from_decimal_odds_is_no_vig(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pred = predict_matchup(
                "Player A",
                "Player B",
                context={"surface": "Clay"},
                export_path=_sample_export(tmp_path),
                odds={"p1_odds": 2.0, "p2_odds": 2.0},
            ).to_dict()

        self.assertEqual(pred["market_p1_probability"], 0.5)
        self.assertEqual(pred["edge_p1"], round(pred["p1_probability"] - 0.5, 6))

    def test_missing_data_returns_caveats_instead_of_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pred = predict_matchup(
                "Unknown 1",
                "Unknown 2",
                context={"surface": "Grass"},
                export_path=tmp_path / "does_not_exist.csv",
            ).to_dict()

        self.assertEqual(pred["p1_probability"], 0.5)
        self.assertEqual(pred["p2_probability"], 0.5)
        self.assertTrue(pred["caveats"])

    def test_confidence_labels_are_deterministic(self) -> None:
        self.assertEqual(confidence_label(0.51), "low")
        self.assertEqual(confidence_label(0.57), "lean")
        self.assertEqual(confidence_label(0.63), "medium")
        self.assertEqual(confidence_label(0.74), "high")


if __name__ == "__main__":
    unittest.main()

