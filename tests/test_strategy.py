import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tennis_ml.live_stats.database import fetch_picks, read_table
from tennis_ml.strategy import (
    StrategyConfig,
    evaluate_pick,
    generate_picks,
    kalshi_fee,
    persist_picks,
    settle_picks,
    size_stake,
)


class EvaluatePickTests(unittest.TestCase):
    def test_positive_ev_on_yes_when_fair_prob_clears_price(self):
        result = evaluate_pick(0.65, 0.55)
        self.assertEqual(result['recommended_side'], 'yes')
        self.assertGreater(result['ev_yes'], 0)
        self.assertGreater(result['net_ev'], 0)
        self.assertLess(result['ev_no'], 0)

    def test_fair_price_is_negative_after_fees(self):
        result = evaluate_pick(0.5, 0.5)
        self.assertLess(result['ev_yes'], 0)
        self.assertLess(result['ev_no'], 0)
        self.assertLess(result['net_ev'], 0)

    def test_ev_math_matches_fee_formula(self):
        fair, price = 0.65, 0.55
        fee = 0.07 * price * (1 - price)
        expected_per_contract = fair * (1 - price) - (1 - fair) * price - fee
        result = evaluate_pick(fair, price)
        self.assertAlmostEqual(result['ev_yes_per_contract'], expected_per_contract, places=12)
        self.assertAlmostEqual(result['ev_yes'], expected_per_contract / price, places=12)
        self.assertAlmostEqual(result['fee_per_contract'], kalshi_fee(price), places=12)

    def test_no_side_recommended_when_yes_overpriced(self):
        result = evaluate_pick(0.30, 0.50)
        self.assertEqual(result['recommended_side'], 'no')
        self.assertGreater(result['net_ev'], 0)

    def test_invalid_price_raises(self):
        with self.assertRaises(ValueError):
            evaluate_pick(0.6, 0.0)
        with self.assertRaises(ValueError):
            evaluate_pick(0.6, 1.0)


class SizeStakeTests(unittest.TestCase):
    def test_kelly_cap_limits_stake(self):
        # Huge edge: full Kelly would exceed the 2% cap.
        stake = size_stake(0.9, 0.4, bankroll=1000, fraction=0.25, cap=0.02)
        self.assertAlmostEqual(stake, 20.0)

    def test_fractional_kelly_below_cap(self):
        fair, price = 0.62, 0.58
        b = (1 - price) / price
        full = (fair * b - (1 - fair)) / b
        expected = min(full * 0.25, 0.02) * 1000
        stake = size_stake(fair, price, bankroll=1000)
        self.assertAlmostEqual(stake, expected)
        self.assertGreater(stake, 0)

    def test_negative_edge_yields_zero_stake(self):
        self.assertEqual(size_stake(0.4, 0.5, bankroll=1000), 0.0)


class GeneratePicksTests(unittest.TestCase):
    def setUp(self):
        self.config = StrategyConfig(bankroll=1000.0)
        self.match = {
            'player1': 'Jannik Sinner',
            'player2': 'Some Qualifier',
            'match_date': '2026-06-11',
            'tournament': 'Test Open',
            'kalshi_ticker': 'TENNIS-TEST-SIN',
        }

    def test_value_bet_tier_when_ev_and_band_pass(self):
        picks = generate_picks(
            [{'match': self.match, 'fair_prob': 0.70, 'kalshi_price': 0.58}],
            self.config,
        )
        pick = picks[0]
        self.assertEqual(pick['confidence_tier'], 'value_bet')
        self.assertEqual(pick['side'], 'yes')
        self.assertEqual(pick['player'], 'Jannik Sinner')
        self.assertGreaterEqual(pick['net_ev'], self.config.ev_threshold)
        self.assertGreater(pick['stake_suggested'], 0)
        self.assertLessEqual(pick['stake_suggested'], self.config.kelly_cap * self.config.bankroll)

    def test_odds_band_gate_blocks_heavy_favorite(self):
        # Price 0.90 -> implied odds 1.11, below the 1.2 floor even with EV edge.
        picks = generate_picks(
            [{'match': self.match, 'fair_prob': 0.99, 'kalshi_price': 0.90}],
            self.config,
        )
        pick = picks[0]
        self.assertNotEqual(pick['confidence_tier'], 'value_bet')
        self.assertEqual(pick['confidence_tier'], 'confident_pick')
        self.assertIn('FAIL', pick['reasons'])

    def test_ev_gate_blocks_thin_edge(self):
        picks = generate_picks(
            [{'match': self.match, 'fair_prob': 0.62, 'kalshi_price': 0.60}],
            self.config,
        )
        self.assertNotEqual(picks[0]['confidence_tier'], 'value_bet')

    def test_confident_pick_without_price(self):
        picks = generate_picks(
            [{'match': self.match, 'fair_prob': 0.66, 'kalshi_price': None}],
            self.config,
        )
        self.assertEqual(picks[0]['confidence_tier'], 'confident_pick')
        self.assertIsNone(picks[0]['net_ev'])

    def test_lean_and_no_bet_tiers(self):
        picks = generate_picks(
            [
                {'match': self.match, 'fair_prob': 0.57, 'kalshi_price': None},
                {'match': self.match, 'fair_prob': 0.52, 'kalshi_price': None},
            ],
            self.config,
        )
        self.assertEqual(picks[0]['confidence_tier'], 'lean')
        self.assertEqual(picks[1]['confidence_tier'], 'no_bet')

    def test_underdog_side_flips_to_player2(self):
        picks = generate_picks(
            [{'match': self.match, 'fair_prob': 0.35, 'kalshi_price': None}],
            self.config,
        )
        pick = picks[0]
        self.assertEqual(pick['side'], 'no')
        self.assertEqual(pick['player'], 'Some Qualifier')
        self.assertAlmostEqual(pick['fair_prob'], 0.65)


class SettlementTests(unittest.TestCase):
    def test_settle_picks_writes_paper_trades_with_kalshi_pnl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / 'live.db'
            config = StrategyConfig(bankroll=1000.0)
            match = {
                'player1': 'Jannik Sinner',
                'player2': 'Some Qualifier',
                'match_date': '2026-06-10',
                'tournament': 'Test Open',
                'kalshi_ticker': 'TENNIS-TEST-SIN',
            }
            picks = generate_picks(
                [{'match': match, 'fair_prob': 0.70, 'kalshi_price': 0.58}],
                config,
            )
            persist_picks(picks, db_path=db_path)
            self.assertEqual(len(fetch_picks(status='open', db_path=db_path)), 1)

            completed = pd.DataFrame([{
                'player1': 'Jannik Sinner',
                'player2': 'Some Qualifier',
                'winner': 'Jannik Sinner',
                'match_date': '2026-06-10',
            }])
            summary = settle_picks(db_path, completed)
            self.assertEqual(summary['settled'], 1)
            self.assertEqual(summary['won'], 1)
            self.assertEqual(summary['paper_trades'], 1)

            settled = fetch_picks(status='won', db_path=db_path)
            self.assertEqual(len(settled), 1)
            trades = read_table('paper_trades', db_path=db_path)
            self.assertEqual(len(trades), 1)
            trade = trades.iloc[0]
            price = picks[0]['kalshi_price']
            stake = picks[0]['stake_suggested']
            contracts = stake / price
            expected_profit = contracts * ((1 - price) - kalshi_fee(price))
            self.assertAlmostEqual(trade['profit'], expected_profit, places=9)
            self.assertEqual(int(trade['won']), 1)

    def test_losing_pick_loses_price_per_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / 'live.db'
            config = StrategyConfig(bankroll=1000.0)
            match = {
                'player1': 'Player A',
                'player2': 'Player B',
                'match_date': '2026-06-10',
                'tournament': 'Test Open',
                'kalshi_ticker': 'TENNIS-TEST-A',
            }
            picks = generate_picks(
                [{'match': match, 'fair_prob': 0.70, 'kalshi_price': 0.58}],
                config,
            )
            persist_picks(picks, db_path=db_path)
            completed = pd.DataFrame([{
                'player1': 'Player A',
                'player2': 'Player B',
                'winner': 'Player B',
                'match_date': '2026-06-10',
            }])
            summary = settle_picks(db_path, completed)
            self.assertEqual(summary['lost'], 1)
            trades = read_table('paper_trades', db_path=db_path)
            price = picks[0]['kalshi_price']
            stake = picks[0]['stake_suggested']
            contracts = stake / price
            self.assertAlmostEqual(trades.iloc[0]['profit'], contracts * -price, places=9)


if __name__ == '__main__':
    unittest.main()
