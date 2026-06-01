#!/usr/bin/env python3
"""
Evaluate a named paper-trading strategy against a historical bet ledger.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.backtesting import (
    apply_friction_to_bets,
    apply_strategy,
    available_strategies,
    gate_summary,
    segment_stability_report,
)
from tennis_ml.backtesting.pnl_backtester import PnLBacktester


def main():
    parser = argparse.ArgumentParser(description='Evaluate named tennis strategy')
    parser.add_argument('--bets', default='walk_forward_results_current/bets.csv')
    parser.add_argument('--strategy', default='favorite_band_no_slams_v1')
    parser.add_argument('--output-dir', default='strategy_eval_current')
    parser.add_argument('--friction-bps', type=float, default=200)
    parser.add_argument('--stake', type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bets = apply_friction_to_bets(pd.read_csv(args.bets), friction_bps=args.friction_bps, stake=args.stake)
    strategy_bets = apply_strategy(bets, args.strategy)
    strategy_bets.to_csv(output_dir / 'strategy_bets.csv', index=False)
    pd.DataFrame([PnLBacktester(stake=args.stake).summarize(strategy_bets)]).to_csv(output_dir / 'summary.csv', index=False)
    gate_summary(strategy_bets, stake=args.stake, min_bets=100).to_csv(output_dir / 'gate_summary.csv', index=False)
    segment_stability_report(strategy_bets, stake=args.stake).to_csv(output_dir / 'segment_stability.csv', index=False)
    available_strategies().to_csv(output_dir / 'available_strategies.csv', index=False)
    print(f'Wrote strategy evaluation to: {output_dir}')


if __name__ == '__main__':
    main()
