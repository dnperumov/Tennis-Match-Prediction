import unittest

import pandas as pd

from tennis_ml.backtesting.trade_mining import mine_trades
from tennis_ml.markets.kalshi import KalshiClient, normalize_kalshi_events


class TradeMiningAndKalshiTests(unittest.TestCase):
    def test_trade_mining_finds_profitable_filter(self):
        bets = pd.DataFrame([
            {
                'model_type': 'top_10',
                'surface': 'Grass',
                'tournament_importance': 4,
                'round': 'QF',
                'model_probability': 0.65,
                'market_probability': 0.55,
                'odds': 2.0,
                'edge': 0.10,
                'ev': 0.30,
                'clv': 0.02,
                'market_overround': 1.04,
                'friction_adjusted_ev': 0.25,
                'fractional_kelly_25': 0.05,
                'won': i % 2 == 0,
                'stake': 1.0,
                'profit': 1.0 if i % 2 == 0 else -1.0,
                'friction_adjusted_profit': 0.98 if i % 2 == 0 else -1.0,
            }
            for i in range(40)
        ])
        reports = mine_trades(bets, min_bets=10, clusters=2)

        self.assertIn('filter_profit_report', reports)
        self.assertIn('cluster_report', reports)
        self.assertFalse(reports['filter_profit_report'].empty)

    def test_kalshi_normalization_converts_prices_to_decimal_odds(self):
        events = [{
            'event_ticker': 'TEST',
            'series_ticker': 'KXTENNIS',
            'title': 'Tennis test',
            'markets': [{
                'ticker': 'TEST-MKT',
                'title': 'Will Player A win?',
                'yes_sub_title': 'Player A',
                'no_sub_title': 'Player B',
                'yes_ask_dollars': '0.4000',
                'yes_bid_dollars': '0.3900',
                'no_ask_dollars': '0.6200',
                'no_bid_dollars': '0.6100',
                'liquidity_dollars': '100.00',
                'volume_fp': '10.00',
            }],
        }]

        markets = normalize_kalshi_events(events)

        self.assertEqual(markets.loc[0, 'market_ticker'], 'TEST-MKT')
        self.assertAlmostEqual(markets.loc[0, 'yes_decimal_odds'], 2.5)
        self.assertAlmostEqual(markets.loc[0, 'no_decimal_odds'], 1 / 0.62)

    def test_kalshi_authenticated_requests_require_env_credentials(self):
        client = KalshiClient(api_key_id=None, private_key_pem=None)
        with self.assertRaises(ValueError):
            client.auth_headers('GET', '/markets/TEST/orderbook')


if __name__ == '__main__':
    unittest.main()
