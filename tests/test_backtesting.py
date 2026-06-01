import tempfile
import unittest

import pandas as pd

from tennis_ml.backtesting import OddsLoader, PnLBacktester
from tennis_ml.backtesting.strategy import add_edge_trust, gate_summary


class BacktestingTests(unittest.TestCase):
    def test_odds_matching_and_flat_stake_pnl(self):
        predictions = pd.DataFrame([
            {
                'tourney_date': pd.Timestamp('2025-01-01'),
                'tourney_name': 'Test Open',
                'round': 'R32',
                'player1': 'Player A',
                'player2': 'Player B',
                'winner_name': 'Player A',
                'loser_name': 'Player B',
                'result': 1,
                'model_type': 'other',
                'player1_probability': 0.60,
                'player2_probability': 0.40,
            },
            {
                'tourney_date': pd.Timestamp('2025-01-02'),
                'tourney_name': 'Test Open',
                'round': 'R16',
                'player1': 'Player C',
                'player2': 'Player D',
                'winner_name': 'Player D',
                'loser_name': 'Player C',
                'result': 2,
                'model_type': 'top_10',
                'player1_probability': 0.58,
                'player2_probability': 0.42,
            },
        ])
        odds = pd.DataFrame([
            {'Date': '01/01/2025', 'Winner': 'Player A', 'Loser': 'Player B', 'AvgW': 2.00, 'AvgL': 1.80},
            {'Date': '02/01/2025', 'Winner': 'Player D', 'Loser': 'Player C', 'AvgW': 1.70, 'AvgL': 2.10},
        ])

        with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
            odds.to_csv(temp_file.name, index=False)
            loader = OddsLoader()
            loaded_odds = loader.load(temp_file.name)
            priced = loader.attach_odds(predictions, loaded_odds)

        self.assertEqual(priced.loc[0, 'player1_odds'], 2.00)
        self.assertEqual(priced.loc[1, 'player1_odds'], 2.10)

        backtester = PnLBacktester(edge_threshold=0.05, stake=1.0)
        bets = backtester.generate_bets(priced)
        summary = backtester.summarize(bets)

        self.assertEqual(len(bets), 2)
        self.assertAlmostEqual(summary['profit'], 0.0)
        self.assertAlmostEqual(summary['roi'], 0.0)
        self.assertEqual(summary['wins'], 1)

    def test_friction_adjusted_pnl_lowers_profit_without_changing_outcomes(self):
        priced = pd.DataFrame([
            {
                'tourney_date': pd.Timestamp('2025-01-01'),
                'match_date': pd.Timestamp('2025-01-01'),
                'round': 'R32',
                'surface': 'Hard',
                'tournament_importance': 3,
                'player1': 'Player A',
                'player2': 'Player B',
                'winner_name': 'Player A',
                'loser_name': 'Player B',
                'result': 1,
                'model_type': 'other',
                'player1_probability': 0.60,
                'player2_probability': 0.40,
                'player1_odds': 2.00,
                'player2_odds': 1.90,
                'player1_market_probability_novig': 0.50,
                'player2_market_probability_novig': 0.50,
            }
        ])

        no_friction = PnLBacktester(edge_threshold=0.05, stake=1.0, friction_bps=0).generate_bets(priced)
        with_friction = PnLBacktester(edge_threshold=0.05, stake=1.0, friction_bps=200).generate_bets(priced)

        self.assertEqual(bool(no_friction.loc[0, 'won']), bool(with_friction.loc[0, 'won']))
        self.assertAlmostEqual(no_friction.loc[0, 'profit'], with_friction.loc[0, 'profit'])
        self.assertLess(with_friction.loc[0, 'friction_adjusted_profit'], no_friction.loc[0, 'profit'])

    def test_clv_calculation_for_favorite_and_underdog(self):
        priced = pd.DataFrame([
            {
                'tourney_date': pd.Timestamp('2025-01-01'),
                'match_date': pd.Timestamp('2025-01-01'),
                'round': 'R32',
                'surface': 'Hard',
                'tournament_importance': 3,
                'player1': 'Favorite',
                'player2': 'Underdog',
                'winner_name': 'Favorite',
                'loser_name': 'Underdog',
                'result': 1,
                'model_type': 'top_10',
                'player1_probability': 0.70,
                'player2_probability': 0.30,
                'player1_odds': 1.60,
                'player2_odds': 3.00,
                'player1_closing_odds': 1.50,
                'player2_closing_odds': 3.20,
                'player1_market_probability_novig': 0.65,
                'player2_market_probability_novig': 0.35,
            },
            {
                'tourney_date': pd.Timestamp('2025-01-02'),
                'match_date': pd.Timestamp('2025-01-02'),
                'round': 'R32',
                'surface': 'Hard',
                'tournament_importance': 3,
                'player1': 'Dog A',
                'player2': 'Fav B',
                'winner_name': 'Dog A',
                'loser_name': 'Fav B',
                'result': 1,
                'model_type': 'other',
                'player1_probability': 0.45,
                'player2_probability': 0.55,
                'player1_odds': 3.00,
                'player2_odds': 1.50,
                'player1_closing_odds': 2.80,
                'player2_closing_odds': 1.55,
                'player1_market_probability_novig': 0.35,
                'player2_market_probability_novig': 0.65,
            },
        ])

        bets = PnLBacktester(edge_threshold=0.03, stake=1.0).generate_bets(priced)

        self.assertAlmostEqual(bets.loc[0, 'clv'], 1.60 / 1.50 - 1)
        self.assertAlmostEqual(bets.loc[1, 'clv'], 3.00 / 2.80 - 1)

    def test_align_match_dates_uses_exact_tennis_data_date(self):
        matches = pd.DataFrame([
            {
                'tourney_date': pd.Timestamp('2025-01-01'),
                'winner_name': 'Player A',
                'loser_name': 'Player B',
                'tourney_id': '2025-test',
                'match_num': 1,
            }
        ])
        odds = pd.DataFrame([
            {'Date': '03/01/2025', 'Winner': 'Player A', 'Loser': 'Player B', 'AvgW': 1.80, 'AvgL': 2.00},
        ])

        with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
            odds.to_csv(temp_file.name, index=False)
            loader = OddsLoader()
            loaded_odds = loader.load(temp_file.name)
            aligned = loader.align_match_dates(matches, loaded_odds)

        self.assertEqual(aligned.loc[0, 'match_date'], pd.Timestamp('2025-01-03'))
        self.assertEqual(aligned.loc[0, 'tourney_start_date'], pd.Timestamp('2025-01-01'))
        self.assertTrue(aligned.loc[0, 'match_date_aligned'])

    def test_edge_trust_uses_prior_years_only(self):
        rows = []
        for year in [2019, 2020, 2021]:
            for i in range(80):
                rows.append({
                    'test_year': year,
                    'tourney_date': pd.Timestamp(f'{year}-01-01'),
                    'match_date': pd.Timestamp(f'{year}-01-{(i % 28) + 1:02d}'),
                    'round': 'R32',
                    'surface': 'Hard',
                    'tournament_importance': 3,
                    'player1': f'Player {year} {i} A',
                    'player2': f'Player {year} {i} B',
                    'winner_name': f'Player {year} {i} A' if i % 2 == 0 else f'Player {year} {i} B',
                    'loser_name': f'Player {year} {i} B' if i % 2 == 0 else f'Player {year} {i} A',
                    'result': 1 if i % 2 == 0 else 2,
                    'model_type': 'top_10',
                    'player1_probability': 0.60,
                    'player2_probability': 0.40,
                    'player1_odds': 2.10,
                    'player2_odds': 1.80,
                    'player1_closing_odds': 2.00,
                    'player2_closing_odds': 1.85,
                    'player1_market_probability_novig': 0.50,
                    'player2_market_probability_novig': 0.50,
                })
        predictions = pd.DataFrame(rows)

        _, report = add_edge_trust(predictions, edge_threshold=0.03, min_prior_bets=50)

        self.assertTrue((report['max_prior_year'] < report['test_year']).all())
        self.assertIn(2020, set(report['test_year']))

    def test_gate_summary_contains_required_gates(self):
        bets = pd.DataFrame([
            {
                'tourney_date': pd.Timestamp('2025-01-01'),
                'match_date': pd.Timestamp('2025-01-01'),
                'won': True,
                'stake': 1.0,
                'profit': 1.0,
                'friction_adjusted_profit': 0.98,
                'odds': 2.0,
                'model_probability': 0.6,
                'edge': 0.1,
                'clv': 0.03,
            }
            for _ in range(300)
        ])

        gates = gate_summary(bets, calibration=pd.DataFrame({'ece_component': [0.001]}), min_bets=300)
        self.assertTrue({'minimum_bets', 'friction_adjusted_roi', 'average_clv', 'positive_clv_rate', 'overall'}.issubset(set(gates['gate'])))


if __name__ == '__main__':
    unittest.main()
