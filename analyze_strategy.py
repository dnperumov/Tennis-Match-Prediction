#!/usr/bin/env python3
"""
Analyze walk-forward predictions and bets against profitability gates.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.backtesting import (
    PnLBacktester,
    add_edge_trust,
    apply_friction_to_bets,
    gate_summary,
    segment_stability_report,
    strategy_candidate_report,
)


def main():
    parser = argparse.ArgumentParser(description='Analyze tennis betting strategy outputs')
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--bets')
    parser.add_argument('--calibration')
    parser.add_argument('--output-dir', default='strategy_analysis')
    parser.add_argument('--edge-threshold', type=float, default=0.03)
    parser.add_argument('--threshold-grid', type=float, nargs='*',
                        default=[0.01, 0.02, 0.03, 0.05, 0.075, 0.10])
    parser.add_argument('--stake', type=float, default=1.0)
    parser.add_argument('--friction-bps', type=float, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    predictions, trust_summary = add_edge_trust(
        predictions,
        edge_threshold=args.edge_threshold,
        stake=args.stake,
        friction_bps=args.friction_bps,
    )
    trust_summary.to_csv(output_dir / 'edge_trust_summary.csv', index=False)

    if args.bets:
        bets = apply_friction_to_bets(pd.read_csv(args.bets), friction_bps=args.friction_bps, stake=args.stake)
    else:
        bets = PnLBacktester(
            edge_threshold=args.edge_threshold,
            stake=args.stake,
            friction_bps=args.friction_bps,
        ).generate_bets(predictions)

    calibration = pd.read_csv(args.calibration) if args.calibration else None
    strategy_candidate_report(
        predictions,
        thresholds=args.threshold_grid,
        stake=args.stake,
        friction_bps=args.friction_bps,
    ).to_csv(output_dir / 'strategy_candidates.csv', index=False)
    segment_stability_report(bets, stake=args.stake).to_csv(output_dir / 'segment_stability.csv', index=False)
    gate_summary(bets, calibration=calibration, stake=args.stake).to_csv(output_dir / 'gate_summary.csv', index=False)
    bets.to_csv(output_dir / 'bets.csv', index=False)
    predictions.to_csv(output_dir / 'predictions_with_edge_trust.csv', index=False)
    print(f'Wrote strategy analysis reports to: {output_dir}')


if __name__ == '__main__':
    main()
