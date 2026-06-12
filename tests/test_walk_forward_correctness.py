import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from walk_forward import predicted_winner_from_probability, prediction_correctness


def make_predictions_frame() -> pd.DataFrame:
    frame = pd.DataFrame({
        'player1': ['A', 'C', 'E', 'G'],
        'player2': ['B', 'D', 'F', 'H'],
        'winner_name': ['A', 'D', 'F', 'G'],
        'player1_probability': [0.7, 0.6, pd.NA, 0.5],
        'player2_probability': [0.3, 0.4, pd.NA, 0.5],
    })
    frame['predicted_winner'] = frame.apply(predicted_winner_from_probability, axis=1)
    return frame


class PredictedWinnerTests(unittest.TestCase):
    def test_player1_predicted_when_more_probable(self):
        row = pd.Series({'player1': 'A', 'player2': 'B',
                         'player1_probability': 0.7, 'player2_probability': 0.3})
        self.assertEqual(predicted_winner_from_probability(row), 'A')

    def test_player2_predicted_when_more_probable(self):
        row = pd.Series({'player1': 'A', 'player2': 'B',
                         'player1_probability': 0.4, 'player2_probability': 0.6})
        self.assertEqual(predicted_winner_from_probability(row), 'B')

    def test_tie_at_half_predicts_player1(self):
        # Must match the UI recompute convention (player1_probability >= 0.5).
        row = pd.Series({'player1': 'A', 'player2': 'B',
                         'player1_probability': 0.5, 'player2_probability': 0.5})
        self.assertEqual(predicted_winner_from_probability(row), 'A')

    def test_missing_probability_yields_na(self):
        row = pd.Series({'player1': 'A', 'player2': 'B',
                         'player1_probability': pd.NA, 'player2_probability': pd.NA})
        self.assertTrue(pd.isna(predicted_winner_from_probability(row)))


class PredictionCorrectnessTests(unittest.TestCase):
    def test_unscored_rows_are_na_not_false(self):
        # Regression: unscored matches (no market price -> no probability) were
        # stored as prediction_correct=False, deflating accuracy in the CSVs.
        frame = make_predictions_frame()
        correct = prediction_correctness(frame)
        self.assertEqual(str(correct.dtype), 'boolean')
        self.assertTrue(pd.isna(correct.iloc[2]))

    def test_scored_rows_match_probability_recompute(self):
        frame = make_predictions_frame()
        correct = prediction_correctness(frame)
        scored = frame['player1_probability'].notna()
        recomputed = (
            (frame.loc[scored, 'player1_probability'] >= 0.5)
            == (frame.loc[scored, 'winner_name'] == frame.loc[scored, 'player1'])
        )
        self.assertListEqual(list(correct[scored]), list(recomputed))
        # Spot-check: correct player1 pick, wrong player2-favoured pick,
        # tie resolved to player1 who actually won.
        self.assertEqual(list(correct[scored]), [True, False, True])


if __name__ == '__main__':
    unittest.main()
