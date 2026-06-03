import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.advanced_feature_model_research import add_advanced_side_dataset


def _row(date, tournament, series, round_value, winner, loser, wentry="", lentry=""):
    return {
        "Date": pd.Timestamp(date),
        "Tournament": tournament,
        "Location": tournament,
        "Series": series,
        "Court": "Outdoor",
        "Surface": "Hard",
        "Round": round_value,
        "Best of": 3,
        "Winner": winner,
        "Loser": loser,
        "WRank": 80,
        "LRank": 120,
        "WPts": 800,
        "LPts": 500,
        "W1": 6,
        "L1": 4,
        "W2": 6,
        "L2": 4,
        "W3": None,
        "L3": None,
        "W4": None,
        "L4": None,
        "W5": None,
        "L5": None,
        "Wsets": 2,
        "Lsets": 0,
        "Comment": "Completed",
        "B365W": 1.70,
        "B365L": 2.20,
        "WEntry": wentry,
        "LEntry": lentry,
    }


class MissingContextFeaturesTest(unittest.TestCase):
    def test_entry_codes_are_mapped_to_player_sides_after_random_side_swap(self) -> None:
        df = pd.DataFrame([
            _row("2024-01-01", "Brisbane", "ATP250", "1st Round", "Qualifier Player", "Wildcard Player", wentry="Q", lentry="WC"),
        ])

        features = add_advanced_side_dataset(df, seed=7)
        row = features.iloc[0]

        if row["player1"] == "Qualifier Player":
            self.assertEqual(row["p1_qualifier"], 1)
            self.assertEqual(row["p2_wildcard"], 1)
            self.assertEqual(row["qualifier_diff"], 1)
            self.assertEqual(row["wildcard_diff"], -1)
        else:
            self.assertEqual(row["p2_qualifier"], 1)
            self.assertEqual(row["p1_wildcard"], 1)
            self.assertEqual(row["qualifier_diff"], -1)
            self.assertEqual(row["wildcard_diff"], 1)

    def test_prior_challenger_title_and_form_feed_future_atp_early_round(self) -> None:
        df = pd.DataFrame([
            _row("2024-01-01", "Canberra Challenger", "Challenger", "The Final", "Rising Player", "Other Player"),
            _row("2024-01-15", "Brisbane", "ATP250", "1st Round", "Rising Player", "Tour Player"),
        ])

        features = add_advanced_side_dataset(df, seed=11)
        atp = features[features["tournament"].eq("Brisbane")].iloc[0]

        if atp["player1"] == "Rising Player":
            self.assertGreater(atp["p1_challenger_form"], atp["p2_challenger_form"])
            self.assertEqual(atp["p1_challenger_title_30d"], 1)
            self.assertEqual(atp["challenger_title_30d_diff"], 1)
        else:
            self.assertGreater(atp["p2_challenger_form"], atp["p1_challenger_form"])
            self.assertEqual(atp["p2_challenger_title_30d"], 1)
            self.assertEqual(atp["challenger_title_30d_diff"], -1)


if __name__ == "__main__":
    unittest.main()
