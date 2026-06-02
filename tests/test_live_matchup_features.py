#!/usr/bin/env python3
"""Tests for live matchup feature construction and routing."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ml.predict.live_features import (  # noqa: E402
    audit_schedule_quality,
    build_matchup_feature_snapshot,
    parse_entry_context,
    route_model_source,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_date": "2026-05-01",
                "tourney_name": "Madrid Masters",
                "surface": "Clay",
                "round": "R64",
                "winner_name": "Player A",
                "loser_name": "Player B",
                "score": "6-4 6-4",
                "minutes": 80,
                "winner_serve_first_won_pct": 75.0,
                "loser_serve_first_won_pct": 62.0,
                "winner_return_points_won_pct": 41.0,
                "loser_return_points_won_pct": 30.0,
                "winner_total_points_won_pct": 55.0,
                "loser_total_points_won_pct": 45.0,
            },
            {
                "match_date": "2026-05-03",
                "tourney_name": "Madrid Masters",
                "surface": "Clay",
                "round": "R32",
                "winner_name": "Player A",
                "loser_name": "Player C",
                "score": "7-6 6-7 6-4",
                "minutes": 155,
                "winner_serve_first_won_pct": 71.0,
                "loser_serve_first_won_pct": 66.0,
                "winner_return_points_won_pct": 38.0,
                "loser_return_points_won_pct": 33.0,
                "winner_total_points_won_pct": 52.0,
                "loser_total_points_won_pct": 48.0,
            },
            {
                "match_date": "2026-05-03",
                "tourney_name": "Madrid Masters",
                "surface": "Clay",
                "round": "R32",
                "winner_name": "Player B",
                "loser_name": "Player D",
                "score": "6-1 6-1",
                "minutes": 58,
                "winner_serve_first_won_pct": 69.0,
                "loser_serve_first_won_pct": 55.0,
                "winner_return_points_won_pct": 44.0,
                "loser_return_points_won_pct": 25.0,
                "winner_total_points_won_pct": 60.0,
                "loser_total_points_won_pct": 40.0,
            },
            {
                "match_date": "2026-05-20",
                "tourney_name": "Roland Garros",
                "surface": "Clay",
                "round": "R128",
                "winner_name": "Player A",
                "loser_name": "Player E",
                "score": "6-3 6-3 6-3",
                "minutes": 120,
                "winner_serve_first_won_pct": 74.0,
                "loser_serve_first_won_pct": 61.0,
                "winner_return_points_won_pct": 40.0,
                "loser_return_points_won_pct": 29.0,
                "winner_total_points_won_pct": 57.0,
                "loser_total_points_won_pct": 43.0,
            },
        ]
    )


class LiveMatchupFeaturesTest(unittest.TestCase):
    def test_feature_snapshot_includes_fatigue_rolling_stats_elo_and_entry_context(self) -> None:
        features = build_matchup_feature_snapshot(
            _history(),
            "Player A",
            "Player B",
            context={
                "match_date": "2026-05-22",
                "tournament": "Roland Garros",
                "surface": "Clay",
                "round": "R64",
                "best_of_5": True,
                "player1_entry": "Q",
                "player2_entry": "WC",
            },
        )

        self.assertGreater(features["elo"]["p1_overall"], 1500.0)
        self.assertGreater(features["elo"]["p1_surface"], features["elo"]["p2_surface"])
        self.assertGreater(features["rolling_stats"]["p1_first_won_pct"], features["rolling_stats"]["p2_first_won_pct"])
        self.assertEqual(features["entry_context"]["p1_qualifier"], 1)
        self.assertEqual(features["entry_context"]["p2_wildcard"], 1)
        self.assertEqual(features["fatigue"]["p1_current_tournament_matches"], 1)
        self.assertEqual(features["fatigue"]["p2_current_tournament_matches"], 0)
        self.assertGreater(features["score_components"]["elo_score"], 0.0)

    def test_schedule_quality_flags_collapsed_tournament_dates(self) -> None:
        df = pd.DataFrame({"match_date": ["2026-05-25"] * 5, "tourney_name": ["Roland Garros"] * 5})
        audit = audit_schedule_quality(df)
        self.assertEqual(audit["collapsed_tournaments"][0]["tournament"], "Roland Garros")
        self.assertIn("Collapsed or low-granularity match dates", audit["caveats"][0])

    def test_routing_prefers_slam_calibrated_market_when_available(self) -> None:
        route = route_model_source({"tournament": "Roland Garros", "best_of_5": True}, has_market=True, has_advanced_report=True)
        self.assertEqual(route["model_source"], "slam_market_calibrated_route_v1")
        self.assertLess(route["fallback_shrink"], 0.2)

    def test_parse_entry_context_understands_entry_codes(self) -> None:
        parsed = parse_entry_context("Q", "LL")
        self.assertEqual(parsed["p1_qualifier"], 1)
        self.assertEqual(parsed["p2_lucky_loser"], 1)
        self.assertEqual(parsed["qualifier_diff"], 1)


if __name__ == "__main__":
    unittest.main()
