"""
Named paper-trading strategy rules.

These rules are candidates for forward paper trading, not a guarantee of live
profitability. Promote a rule only after it passes gate checks on future data.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_STRATEGY = 'favorite_band_no_slams_v1'


STRATEGY_DESCRIPTIONS = {
    'favorite_band_no_slams_v1': (
        'Bet model-selected sides priced from 1.50 to 2.00 decimal odds, '
        'excluding Grand Slams. This was the most stable mined pocket in the '
        '2019-2026 B365 walk-forward ledger.'
    ),
    'qf_edge_no_slams_research_v1': (
        'Research-only quarterfinal edge pocket: QF, edge >= 10%, no Slams. '
        'High historical ROI, but smaller sample and less suitable as default.'
    ),
}


def apply_strategy(bets: pd.DataFrame, strategy: str = DEFAULT_STRATEGY) -> pd.DataFrame:
    if strategy == 'favorite_band_no_slams_v1':
        mask = (
            bets['odds'].ge(1.5) &
            bets['odds'].lt(2.0) &
            bets['tournament_importance'].ne(5)
        )
    elif strategy == 'qf_edge_no_slams_research_v1':
        mask = (
            bets['round'].eq('QF') &
            bets['edge'].ge(0.10) &
            bets['tournament_importance'].ne(5)
        )
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    filtered = bets[mask].copy()
    filtered['strategy_name'] = strategy
    filtered['accepted_by_strategy'] = True
    return filtered


def available_strategies() -> pd.DataFrame:
    return pd.DataFrame([
        {'strategy_name': name, 'description': description}
        for name, description in STRATEGY_DESCRIPTIONS.items()
    ])
