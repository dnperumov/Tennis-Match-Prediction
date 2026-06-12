"""Strategy layer: execution price vs fair value picks, sizing, and settlement."""

from .picks import (
    STRATEGY_VERSION,
    StrategyConfig,
    evaluate_pick,
    generate_picks,
    kalshi_fee,
    match_kalshi_markets_to_predictions,
    persist_picks,
    settle_picks,
    size_stake,
)

__all__ = [
    'STRATEGY_VERSION',
    'StrategyConfig',
    'evaluate_pick',
    'generate_picks',
    'kalshi_fee',
    'match_kalshi_markets_to_predictions',
    'persist_picks',
    'settle_picks',
    'size_stake',
]
