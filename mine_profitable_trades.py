#!/usr/bin/env python3
"""
Find historically profitable bet filters and clusters.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.backtesting import apply_friction_to_bets, mine_trades


def main():
    parser = argparse.ArgumentParser(description='Mine profitable trade pockets from a bet ledger')
    parser.add_argument('--bets', required=True)
    parser.add_argument('--output-dir', default='trade_mining_results')
    parser.add_argument('--stake', type=float, default=1.0)
    parser.add_argument('--friction-bps', type=float, default=200)
    parser.add_argument('--min-bets', type=int, default=30)
    parser.add_argument('--clusters', type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bets = apply_friction_to_bets(
        pd.read_csv(args.bets),
        friction_bps=args.friction_bps,
        stake=args.stake,
    )
    reports = mine_trades(
        bets,
        stake=args.stake,
        min_bets=args.min_bets,
        clusters=args.clusters,
    )
    for name, frame in reports.items():
        frame.to_csv(output_dir / f'{name}.csv', index=False)

    print(f'Wrote trade-mining reports to: {output_dir}')


if __name__ == '__main__':
    main()
