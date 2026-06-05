import math
import unittest

import pandas as pd

from scripts.ability_pressure_diagnostic import (
    add_ability_consensus_segments,
    add_favorite_perspective_columns,
    bucket_numeric_signal,
    metrics_for,
    no_lookahead_multi_segment_fallback_routing,
    no_lookahead_segment_fallback_routing,
    stable_segment_summary,
)


class AbilityPressureDiagnosticTests(unittest.TestCase):
    def test_favorite_perspective_flips_player_diff_and_pressure_for_p2_favorites(self):
        df = pd.DataFrame(
            {
                "result": [1, 0],
                "market_bin_recalibrated_p1": [0.70, 0.30],
                "residual_overlay_filtered_p1": [0.75, 0.25],
                "elo_diff": [100, 80],
            }
        )

        out = add_favorite_perspective_columns(
            df,
            baseline_prob_col="market_bin_recalibrated_p1",
            model_prob_col="residual_overlay_filtered_p1",
            signal_cols=["elo_diff"],
        )

        self.assertEqual(out["favorite_side"].tolist(), ["p1", "p2"])
        self.assertEqual(out["favorite_won"].tolist(), [1, 1])
        self.assertAlmostEqual(out.loc[0, "favorite_elo_diff"], 100)
        self.assertAlmostEqual(out.loc[1, "favorite_elo_diff"], -80)
        self.assertAlmostEqual(out.loc[0, "model_minus_baseline_favorite_prob"], 0.05)
        self.assertAlmostEqual(out.loc[1, "model_minus_baseline_favorite_prob"], 0.05)

    def test_bucket_numeric_signal_uses_missing_sentinel(self):
        got = bucket_numeric_signal(pd.Series([-250, -25, 0, 25, 250, float("nan")]), cuts=[-100, -20, 20, 100], labels=["very_low", "low", "neutral", "high", "very_high"])
        self.assertEqual(got.tolist(), ["very_low", "low", "neutral", "high", "very_high", "missing"])

    def test_metrics_for_handles_single_class_rows(self):
        df = pd.DataFrame({"actual": [1, 1, 1], "model": [0.55, 0.65, 0.75], "baseline": [0.50, 0.60, 0.70]})
        got = metrics_for(df, "model", "baseline", actual_col="actual")
        self.assertEqual(got["rows"], 3)
        self.assertTrue(math.isfinite(got["log_loss"]))
        self.assertTrue(math.isfinite(got["market_log_loss"]))

    def test_stable_segment_summary_requires_multiyear_proper_score_stability(self):
        rows = []
        for year in [2022, 2023, 2024]:
            for _ in range(4):
                rows.append({"year": year, "segment": "stable_good", "actual": 1, "model": 0.80, "baseline": 0.70})
                rows.append({"year": year, "segment": "stable_bad", "actual": 1, "model": 0.60, "baseline": 0.70})
        df = pd.DataFrame(rows)
        good = stable_segment_summary(df, "segment", "model", "baseline", actual_col="actual", min_rows=6, min_years=3, min_stable_year_share=0.66, direction="beats")
        bad = stable_segment_summary(df, "segment", "model", "baseline", actual_col="actual", min_rows=6, min_years=3, min_stable_year_share=0.66, direction="lags")
        self.assertEqual(good[0]["segment"], "stable_good")
        self.assertEqual(bad[0]["segment"], "stable_bad")
        self.assertEqual(good[0]["years_model_beats_market_log_loss"], 3)
        self.assertEqual(bad[0]["years_model_lags_market_brier"], 3)

    def test_ability_consensus_segments_combine_multiple_favorite_side_signals(self):
        df = pd.DataFrame(
            {
                "baseline_favorite_prob": [0.62, 0.78, 0.55],
                "model_minus_baseline_favorite_prob": [0.05, -0.06, 0.0],
                "favorite_elo_diff": [180, -180, 10],
                "favorite_surface_elo_diff": [90, -90, float("nan")],
                "favorite_serve_return_ability_diff": [0.05, -0.05, 0.0],
                "favorite_fatigue_adjusted_ability_diff": [-0.01, -0.05, 0.0],
            }
        )

        out = add_ability_consensus_segments(
            df,
            signal_cols=["elo_diff", "surface_elo_diff", "serve_return_ability_diff", "fatigue_adjusted_ability_diff"],
        )

        self.assertEqual(out["favorite_ability_support_count"].tolist(), [3, 0, 0])
        self.assertEqual(out["favorite_ability_oppose_count"].tolist(), [0, 4, 0])
        self.assertEqual(out["favorite_ability_consensus_bucket"].tolist(), ["ability_strongly_supports_favorite", "ability_strongly_opposes_favorite", "ability_split_or_neutral"])
        self.assertEqual(
            out["ability_consensus_pressure_segment"].tolist()[0],
            "ability_strongly_supports_favorite | modest_fav | model_higher_on_favorite",
        )

    def test_no_lookahead_segment_fallback_routes_only_after_prior_stable_lag(self):
        rows = []
        # 2022/2023 establish a stable bad segment, but cannot route themselves.
        for year in [2022, 2023, 2024]:
            for _ in range(8):
                rows.append({"year": year, "segment": "bad", "result": 1, "model": 0.55, "baseline": 0.80})
                rows.append({"year": year, "segment": "good", "result": 1, "model": 0.82, "baseline": 0.70})
        df = pd.DataFrame(rows)

        got = no_lookahead_segment_fallback_routing(
            df,
            segment_col="segment",
            model_col="model",
            baseline_col="baseline",
            min_train_rows=12,
            min_train_years=2,
            min_stable_year_share=1.0,
        )

        self.assertEqual(got["routed_rows"], 8)
        self.assertEqual(got["yearly_routing"][-1]["year"], 2024)
        self.assertEqual(got["yearly_routing"][-1]["flagged_segments"], ["bad"])
        self.assertLess(got["routed_metrics"]["log_loss"], got["model_metrics"]["log_loss"])
        self.assertEqual(got["baseline_probability_col"], "baseline")
    def test_no_lookahead_multi_segment_fallback_routes_union_once_per_row(self):
        rows = []
        # 2022/2023 establish bad segments in two different diagnostics. 2024 is
        # the first year that may be routed, and rows matching either segment
        # should be routed once, not double-counted.
        for year in [2022, 2023, 2024]:
            for _ in range(8):
                rows.append({"year": year, "seg_a": "bad_a", "seg_b": "ok_b", "result": 1, "model": 0.55, "baseline": 0.80})
                rows.append({"year": year, "seg_a": "ok_a", "seg_b": "bad_b", "result": 1, "model": 0.56, "baseline": 0.81})
                rows.append({"year": year, "seg_a": "good_a", "seg_b": "good_b", "result": 1, "model": 0.82, "baseline": 0.70})
        df = pd.DataFrame(rows)

        got = no_lookahead_multi_segment_fallback_routing(
            df,
            segment_cols=["seg_a", "seg_b"],
            model_col="model",
            baseline_col="baseline",
            min_train_rows=12,
            min_train_years=2,
            min_stable_year_share=1.0,
        )

        self.assertEqual(got["routed_rows"], 16)
        self.assertEqual(got["yearly_routing"][-1]["year"], 2024)
        self.assertIn("bad_a", got["yearly_routing"][-1]["flagged_segments_by_column"]["seg_a"])
        self.assertIn("bad_b", got["yearly_routing"][-1]["flagged_segments_by_column"]["seg_b"])
        self.assertLess(got["routed_metrics"]["log_loss"], got["model_metrics"]["log_loss"])
        self.assertEqual(got["baseline_probability_col"], "baseline")


if __name__ == "__main__":
    unittest.main()
