"""
PnL backtesting for model probabilities against market odds.
"""

import pandas as pd


class PnLBacktester:
    """Runs flat-stake betting simulations from model probabilities and odds."""

    def __init__(
        self,
        edge_threshold: float = 0.03,
        stake: float = 1.0,
        friction_bps: float = 0.0,
        execution_odds_source: str = 'selected',
        closing_odds_source: str = 'average_proxy',
    ):
        self.edge_threshold = edge_threshold
        self.stake = stake
        self.friction_bps = friction_bps
        self.execution_odds_source = execution_odds_source
        self.closing_odds_source = closing_odds_source

    def generate_bets(
        self,
        predictions: pd.DataFrame,
        min_odds: float = None,
        max_odds: float = None,
        model_type: str = None,
        exclude_slams: bool = False,
        surface: str = None
    ) -> pd.DataFrame:
        """Create one bet per match when either side clears the EV threshold."""
        bets = []
        priced = predictions.dropna(subset=['player1_odds', 'player2_odds']).copy()
        if model_type is not None:
            priced = priced[priced['model_type'] == model_type]
        if exclude_slams:
            priced = priced[priced['tournament_importance'] != 5]
        if surface is not None:
            priced = priced[priced['surface'] == surface]

        for _, row in priced.iterrows():
            candidates = [
                self._candidate(row, side='player1', min_odds=min_odds, max_odds=max_odds),
                self._candidate(row, side='player2', min_odds=min_odds, max_odds=max_odds),
            ]
            candidates = [candidate for candidate in candidates if candidate is not None]
            candidates = [candidate for candidate in candidates if candidate['ev'] >= self.edge_threshold]
            if not candidates:
                continue
            bet = max(candidates, key=lambda candidate: candidate['ev'])
            bets.append(bet)

        return pd.DataFrame(bets)

    def summarize(self, bets: pd.DataFrame) -> dict:
        """Summarize PnL, ROI, hit rate, and drawdown."""
        if bets.empty:
            return {
                'bets': 0,
                'wins': 0,
                'win_rate': 0.0,
                'staked': 0.0,
                'profit': 0.0,
                'roi': 0.0,
                'max_drawdown': 0.0,
                'average_odds': 0.0,
                'average_model_probability': 0.0,
                'average_edge': 0.0,
                'friction_adjusted_profit': 0.0,
                'friction_adjusted_roi': 0.0,
                'average_clv': 0.0,
                'clv_positive_rate': 0.0,
                'average_fractional_kelly': 0.0,
            }

        equity = bets['profit'].cumsum()
        running_peak = equity.cummax().clip(lower=0)
        drawdown = equity - running_peak
        staked = len(bets) * self.stake
        profit = bets['profit'].sum()
        friction_adjusted_profit = (
            bets['friction_adjusted_profit'].sum()
            if 'friction_adjusted_profit' in bets.columns
            else profit
        )

        return {
            'bets': int(len(bets)),
            'wins': int(bets['won'].sum()),
            'win_rate': float(bets['won'].mean()),
            'staked': float(staked),
            'profit': float(profit),
            'roi': float(profit / staked) if staked else 0.0,
            'friction_adjusted_profit': float(friction_adjusted_profit),
            'friction_adjusted_roi': float(friction_adjusted_profit / staked) if staked else 0.0,
            'max_drawdown': float(drawdown.min()),
            'average_odds': float(bets['odds'].mean()),
            'average_model_probability': float(bets['model_probability'].mean()),
            'average_edge': float(bets['edge'].mean()),
            'average_clv': float(bets['clv'].mean()) if 'clv' in bets.columns else 0.0,
            'clv_positive_rate': float((bets['clv'] > 0).mean()) if 'clv' in bets.columns else 0.0,
            'average_fractional_kelly': float(bets['fractional_kelly_25'].mean()) if 'fractional_kelly_25' in bets.columns else 0.0,
        }

    def _candidate(self, row: pd.Series, side: str, min_odds: float = None, max_odds: float = None) -> dict:
        player = row[side]
        probability = row[f'{side}_probability']
        odds = row[f'{side}_odds']
        if min_odds is not None and odds < min_odds:
            return None
        if max_odds is not None and odds > max_odds:
            return None
        won = row['result'] == (1 if side == 'player1' else 2)
        market_probability = row.get(f'{side}_market_probability_novig', 1 / odds)
        closing_odds = row.get(f'{side}_closing_odds')
        friction_adjusted_odds = self._apply_friction(odds)
        clv = None
        if pd.notna(closing_odds) and closing_odds > 0:
            clv = (odds / closing_odds) - 1
        edge = probability - market_probability
        ev = probability * (odds - 1) - (1 - probability)
        profit = self.stake * (odds - 1) if won else -self.stake
        friction_adjusted_profit = self.stake * (friction_adjusted_odds - 1) if won else -self.stake
        friction_adjusted_ev = probability * (friction_adjusted_odds - 1) - (1 - probability)
        fractional_kelly_25 = self._fractional_kelly(probability, odds, fraction=0.25)
        market_overround = row.get('market_overround')

        return {
            'tourney_date': row['tourney_date'],
            'match_date': row.get('match_date', row['tourney_date']),
            'tourney_name': row.get('tourney_name'),
            'round': row.get('round'),
            'surface': row.get('surface'),
            'tournament_importance': row.get('tournament_importance'),
            'player': player,
            'opponent': row['player2'] if side == 'player1' else row['player1'],
            'side': side,
            'model_type': row['model_type'],
            'model_probability': probability,
            'odds': odds,
            'closing_odds': closing_odds,
            'execution_odds_source': row.get('execution_odds_source', self.execution_odds_source),
            'closing_odds_source': row.get('closing_odds_source', self.closing_odds_source),
            'friction_bps': self.friction_bps,
            'friction_adjusted_odds': friction_adjusted_odds,
            'clv': clv,
            'market_probability': market_probability,
            'market_overround': market_overround,
            'edge': edge,
            'ev': ev,
            'friction_adjusted_ev': friction_adjusted_ev,
            'fractional_kelly_25': fractional_kelly_25,
            'edge_trust_probability': row.get(f'{side}_edge_trust_probability', row.get('edge_trust_probability')),
            'accepted_by_strategy': row.get('accepted_by_strategy', True),
            'won': bool(won),
            'stake': self.stake,
            'profit': profit,
            'friction_adjusted_profit': friction_adjusted_profit,
        }

    def _apply_friction(self, odds: float) -> float:
        friction = max(0.0, self.friction_bps) / 10000
        return max(1.01, 1 + (odds - 1) * (1 - friction))

    @staticmethod
    def _fractional_kelly(probability: float, odds: float, fraction: float = 0.25) -> float:
        b = odds - 1
        if b <= 0:
            return 0.0
        full_kelly = (probability * b - (1 - probability)) / b
        return float(max(0.0, full_kelly * fraction))
