import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.high_confidence_strategy import apply_strategy_filter, summarize_strategy


class HighConfidenceStrategyTest(unittest.TestCase):
    def test_strategy_requires_model_market_agreement_without_extreme_favorites(self) -> None:
        df = pd.DataFrame(
            [
                # Qualifies: both model and market like p1, not a huge favorite.
                {"result": 1, "implied_p1_no_vig": 0.64, "market_no_vig_p1": 0.64, "residual_overlay_segment_tuned_p1": 0.68, "round_group": "middle", "underperformance_risk": 0.30},
                # Excluded: too one-sided, even though it is likely correct.
                {"result": 1, "implied_p1_no_vig": 0.86, "market_no_vig_p1": 0.86, "residual_overlay_segment_tuned_p1": 0.89, "round_group": "middle", "underperformance_risk": 0.20},
                # Excluded: model disagrees with market side.
                {"result": 0, "implied_p1_no_vig": 0.61, "market_no_vig_p1": 0.61, "residual_overlay_segment_tuned_p1": 0.42, "round_group": "middle", "underperformance_risk": 0.20},
                # Excluded: early-round model-market override risk.
                {"result": 1, "implied_p1_no_vig": 0.57, "market_no_vig_p1": 0.57, "residual_overlay_segment_tuned_p1": 0.62, "round_group": "early", "underperformance_risk": 0.70},
            ]
        )

        selected = apply_strategy_filter(df)

        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected.iloc[0]["strategy_prob"], 0.68)
        self.assertEqual(selected.iloc[0]["strategy_pick"], 1)

    def test_summary_reports_accuracy_coverage_and_exclusion_reasons(self) -> None:
        df = pd.DataFrame(
            [
                {"result": 1, "implied_p1_no_vig": 0.64, "market_no_vig_p1": 0.64, "residual_overlay_segment_tuned_p1": 0.68, "round_group": "middle", "underperformance_risk": 0.30},
                {"result": 0, "implied_p1_no_vig": 0.35, "market_no_vig_p1": 0.35, "residual_overlay_segment_tuned_p1": 0.31, "round_group": "late", "underperformance_risk": 0.20},
                {"result": 1, "implied_p1_no_vig": 0.86, "market_no_vig_p1": 0.86, "residual_overlay_segment_tuned_p1": 0.89, "round_group": "middle", "underperformance_risk": 0.20},
            ]
        )

        summary = summarize_strategy(df)

        self.assertEqual(summary["selected_rows"], 2)
        self.assertAlmostEqual(summary["coverage"], 2 / 3)
        self.assertAlmostEqual(summary["accuracy"], 1.0)
        self.assertGreater(summary["exclusion_reasons"]["extreme_market_favorite"], 0)


if __name__ == "__main__":
    unittest.main()
