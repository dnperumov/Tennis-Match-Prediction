#!/usr/bin/env python3
"""
Record and settle forward paper-trading opportunities.

This v1 command intentionally does not scrape live odds. It normalizes an
operator-supplied odds/prediction CSV into an append-only paper-trade ledger
and can settle that ledger once results and closing odds are supplied.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.backtesting import PnLBacktester, apply_friction_to_bets, gate_summary, segment_stability_report


LEDGER_COLUMNS = [
    'snapshot_time', 'match_date', 'tourney_name', 'round', 'surface',
    'player', 'opponent', 'side', 'model_probability', 'market_probability',
    'odds', 'closing_odds', 'clv', 'stake', 'won', 'profit',
    'friction_adjusted_profit', 'execution_odds_source', 'closing_odds_source',
]


def main():
    parser = argparse.ArgumentParser(description='Create or settle tennis paper trades')
    parser.add_argument('--odds-input', required=True,
                        help='CSV containing upcoming scored opportunities or settled paper trades')
    parser.add_argument('--output-dir', default='paper_trading')
    parser.add_argument('--edge-threshold', type=float, default=0.03)
    parser.add_argument('--stake', type=float, default=1.0)
    parser.add_argument('--friction-bps', type=float, default=200)
    parser.add_argument('--append', action='store_true',
                        help='Append to an existing paper_trades.csv ledger')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_frame = pd.read_csv(args.odds_input)
    paper_trades = normalize_input(input_frame, args.edge_threshold, args.stake, args.friction_bps)

    ledger_path = output_dir / 'paper_trades.csv'
    if args.append and ledger_path.exists():
        existing = pd.read_csv(ledger_path)
        paper_trades = pd.concat([existing, paper_trades], ignore_index=True)

    for column in LEDGER_COLUMNS:
        if column not in paper_trades.columns:
            paper_trades[column] = pd.NA
    paper_trades[LEDGER_COLUMNS].to_csv(ledger_path, index=False)

    settled = paper_trades.dropna(subset=['won'])
    if not settled.empty:
        settled = apply_friction_to_bets(settled, friction_bps=args.friction_bps, stake=args.stake)
        segment_stability_report(settled, stake=args.stake).to_csv(output_dir / 'segment_stability.csv', index=False)
        gate_summary(settled, stake=args.stake, min_bets=300).to_csv(output_dir / 'gate_summary.csv', index=False)

    print(f'Wrote paper-trade ledger to: {ledger_path}')


def normalize_input(frame: pd.DataFrame, edge_threshold: float, stake: float, friction_bps: float) -> pd.DataFrame:
    if {'player1', 'player2', 'player1_probability', 'player2_probability', 'player1_odds', 'player2_odds'}.issubset(frame.columns):
        return PnLBacktester(
            edge_threshold=edge_threshold,
            stake=stake,
            friction_bps=friction_bps,
        ).generate_bets(frame)

    output = frame.copy()
    if 'snapshot_time' not in output.columns:
        output['snapshot_time'] = pd.Timestamp.utcnow().isoformat()
    if 'stake' not in output.columns:
        output['stake'] = stake
    if 'profit' not in output.columns and {'won', 'odds'}.issubset(output.columns):
        output['profit'] = output.apply(
            lambda row: stake * (row['odds'] - 1) if bool(row['won']) else -stake,
            axis=1,
        )
    if 'friction_adjusted_profit' not in output.columns and {'won', 'odds'}.issubset(output.columns):
        adjusted_odds = 1 + (output['odds'] - 1) * (1 - max(0, friction_bps) / 10000)
        output['friction_adjusted_profit'] = [
            stake * (odds - 1) if bool(won) else -stake
            for odds, won in zip(adjusted_odds, output['won'])
        ]
    if 'clv' not in output.columns and {'odds', 'closing_odds'}.issubset(output.columns):
        output['clv'] = (output['odds'] / output['closing_odds']) - 1
    return output


if __name__ == '__main__':
    main()
