import unittest

import pandas as pd

from tennis_ml.data import DataLoader
from tennis_ml.features import FeatureEngineer


def _match(date, winner, loser):
    return {
        'tourney_id': f'2025-{date}',
        'tourney_name': 'Test Open',
        'surface': 'Hard',
        'draw_size': 32,
        'tourney_level': 'A',
        'tourney_date': pd.Timestamp(date),
        'match_num': 1,
        'winner_name': winner,
        'winner_hand': 'R',
        'winner_ht': 185,
        'winner_age': 25,
        'winner_seed': None,
        'winner_rank': 10,
        'loser_name': loser,
        'loser_hand': 'R',
        'loser_ht': 180,
        'loser_age': 27,
        'loser_seed': None,
        'loser_rank': 20,
    }


class RollingFeatureTests(unittest.TestCase):
    def test_rolling_features_use_only_prior_matches(self):
        matches = pd.DataFrame([
            _match('2025-01-01', 'Player A', 'Player B'),
            _match('2025-01-02', 'Player B', 'Player A'),
        ])
        data_loader = DataLoader()
        feature_engineer = FeatureEngineer()

        features = feature_engineer.create_rolling_features(
            matches,
            data_loader.players,
            data_loader,
            random_state=1
        )

        first_match = features.iloc[0]
        self.assertEqual(first_match['head_to_head_wins_p1'], 0)
        self.assertEqual(first_match['head_to_head_wins_p2'], 0)
        self.assertEqual(first_match['player1_last_5_win_percentage'], 0.5)
        self.assertEqual(first_match['player2_last_5_win_percentage'], 0.5)

        second_match = features.iloc[1]
        expected_p1_h2h = 1 if second_match['player1'] == 'Player A' else 0
        expected_p2_h2h = 1 if second_match['player2'] == 'Player A' else 0
        expected_p1_last_5 = 1 if second_match['player1'] == 'Player A' else 0
        expected_p2_last_5 = 1 if second_match['player2'] == 'Player A' else 0

        self.assertEqual(second_match['head_to_head_wins_p1'], expected_p1_h2h)
        self.assertEqual(second_match['head_to_head_wins_p2'], expected_p2_h2h)
        self.assertEqual(second_match['player1_last_5_win_percentage'], expected_p1_last_5)
        self.assertEqual(second_match['player2_last_5_win_percentage'], expected_p2_last_5)


if __name__ == '__main__':
    unittest.main()
