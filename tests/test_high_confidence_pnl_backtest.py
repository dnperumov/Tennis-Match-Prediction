#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from high_confidence_pnl_backtest import build_paper_bets, summarize_bets


class HighConfidencePnlBacktestTest(unittest.TestCase):
    def test_builds_flat_stake_profit_for_picked_sides(self):
        df = pd.DataFrame([
            {
                "date": "2022-01-01",
                "player1": "A",
                "player2": "B",
                "residual_overlay_segment_tuned_p1": 0.70,
                "implied_p1_no_vig": 0.60,
                "p1_odds": 1.90,
                "p2_odds": 2.10,
                "underperformance_risk": 0.10,
                "round_group": "late",
                "result": 1,
            },
            {
                "date": "2022-01-02",
                "player1": "C",
                "player2": "D",
                "residual_overlay_segment_tuned_p1": 0.30,
                "implied_p1_no_vig": 0.40,
                "p1_odds": 2.30,
                "p2_odds": 1.80,
                "underperformance_risk": 0.10,
                "round_group": "late",
                "result": 1,
            },
        ])

        bets = build_paper_bets(df, stake=10.0)

        self.assertEqual(len(bets), 2)
        self.assertEqual(bets.loc[0, "bet_side"], "p1")
        self.assertAlmostEqual(bets.loc[0, "profit"], 9.0)
        self.assertEqual(bets.loc[1, "bet_side"], "p2")
        self.assertAlmostEqual(bets.loc[1, "profit"], -10.0)
        self.assertAlmostEqual(bets["running_profit"].iloc[-1], -1.0)

    def test_min_raw_edge_filters_negative_expected_value_prices(self):
        df = pd.DataFrame([
            {
                "residual_overlay_segment_tuned_p1": 0.66,
                "implied_p1_no_vig": 0.60,
                "p1_odds": 1.40,
                "p2_odds": 3.20,
                "underperformance_risk": 0.10,
                "round_group": "late",
                "result": 1,
            },
            {
                "residual_overlay_segment_tuned_p1": 0.66,
                "implied_p1_no_vig": 0.60,
                "p1_odds": 1.80,
                "p2_odds": 2.20,
                "underperformance_risk": 0.10,
                "round_group": "late",
                "result": 1,
            },
        ])

        bets = build_paper_bets(df, min_raw_edge=0.05)

        self.assertEqual(len(bets), 1)
        self.assertAlmostEqual(bets.iloc[0]["decimal_odds"], 1.80)

    def test_summary_reports_roi_and_year_breakdown(self):
        df = pd.DataFrame([
            {
                "date": "2022-01-01",
                "residual_overlay_segment_tuned_p1": 0.70,
                "implied_p1_no_vig": 0.60,
                "p1_odds": 2.00,
                "p2_odds": 2.00,
                "underperformance_risk": 0.10,
                "round_group": "late",
                "result": 1,
            },
        ])
        bets = build_paper_bets(df, stake=5.0)
        summary = summarize_bets(bets, total_rows=len(df), stake=5.0, min_raw_edge=0.0)

        self.assertEqual(summary["bets"], 1)
        self.assertAlmostEqual(summary["profit"], 5.0)
        self.assertAlmostEqual(summary["roi"], 1.0)
        self.assertEqual(summary["by_year"][0]["year"], 2022)


if __name__ == "__main__":
    unittest.main()
