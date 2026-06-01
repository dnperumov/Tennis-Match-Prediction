#!/usr/bin/env python3
"""
Generate a lightweight HTML dashboard for tennis ML trading research.
"""

import argparse
import html
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Build tennis trading dashboard HTML')
    parser.add_argument('--walk-forward-dir', default='walk_forward_results_current')
    parser.add_argument('--strategy-dir', default='strategy_eval_current')
    parser.add_argument('--mining-dir', default='trade_mining_current')
    parser.add_argument('--kalshi-file', default='data/kalshi/tennis_markets.csv')
    parser.add_argument('--output', default='dashboard/tennis_trading_dashboard.html')
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    html_text = render_dashboard(args)
    output.write_text(html_text, encoding='utf-8')
    print(f'Wrote dashboard to: {output}')


def render_dashboard(args) -> str:
    wf = Path(args.walk_forward_dir)
    strategy = Path(args.strategy_dir)
    mining = Path(args.mining_dir)
    kalshi = Path(args.kalshi_file)

    sections = [
        hero_section(),
        metrics_section('Full Walk-Forward', read_csv(wf / 'summary.csv')),
        table_section('Promotion Gates', read_csv(wf / 'gate_summary.csv')),
        metrics_section('Default Strategy: favorite_band_no_slams_v1', read_csv(strategy / 'summary.csv')),
        table_section('Default Strategy Gates', read_csv(strategy / 'gate_summary.csv')),
        table_section('Default Strategy Stability', read_csv(strategy / 'segment_stability.csv'), limit=16),
        table_section('Top Mined Single Filters', read_csv(mining / 'filter_profit_report.csv'), limit=12),
        table_section('Top Mined Multi-Filter Rules', read_csv(mining / 'profitable_filter_rules.csv'), limit=12),
        table_section('Trade Clusters', read_csv(mining / 'cluster_report.csv'), limit=12),
        table_section('Kalshi Tennis Markets', read_csv(kalshi), limit=20),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tennis Trading Research Dashboard</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f8fa; color: #17202a; }}
    header {{ padding: 28px 36px; background: #0f1720; color: white; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    main {{ padding: 24px 36px 48px; }}
    section {{ margin-bottom: 24px; background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #e2e6ee; border-radius: 6px; padding: 12px; background: #fbfcfd; }}
    .label {{ color: #667085; font-size: 12px; text-transform: uppercase; letter-spacing: .02em; }}
    .value {{ font-size: 22px; font-weight: 650; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf0f5; text-align: left; vertical-align: top; }}
    th {{ background: #f3f5f8; color: #334155; position: sticky; top: 0; }}
    .table-wrap {{ overflow-x: auto; }}
    .pass {{ color: #057a55; font-weight: 650; }}
    .fail {{ color: #b42318; font-weight: 650; }}
    .note {{ color: #667085; max-width: 980px; }}
  </style>
</head>
<body>
{''.join(sections)}
</body>
</html>
"""


def hero_section() -> str:
    return """
<header>
  <h1>Tennis Trading Research Dashboard</h1>
  <div class="note">Research-grade model tracking for walk-forward PnL, strategy gates, mined trade pockets, and Kalshi market discovery. Do not use for live staking until paper-trade gates pass.</div>
</header>
<main>
"""


def metrics_section(title: str, frame: pd.DataFrame) -> str:
    if frame.empty:
        return f'<section><h2>{html.escape(title)}</h2><p class="note">No data found.</p></section>'
    row = frame.iloc[0].to_dict()
    keys = [
        'bets', 'wins', 'win_rate', 'profit', 'roi',
        'friction_adjusted_profit', 'friction_adjusted_roi',
        'average_clv', 'clv_positive_rate', 'max_drawdown',
    ]
    cards = []
    for key in keys:
        if key in row:
            cards.append(f'<div class="metric"><div class="label">{html.escape(key)}</div><div class="value">{format_value(row[key])}</div></div>')
    return f'<section><h2>{html.escape(title)}</h2><div class="grid">{"".join(cards)}</div></section>'


def table_section(title: str, frame: pd.DataFrame, limit: int | None = None) -> str:
    if frame.empty:
        return f'<section><h2>{html.escape(title)}</h2><p class="note">No data found.</p></section>'
    if limit is not None:
        frame = frame.head(limit)
    columns = list(frame.columns)
    header = ''.join(f'<th>{html.escape(str(column))}</th>' for column in columns)
    rows = []
    for _, row in frame.iterrows():
        cells = ''.join(f'<td>{format_cell(column, row[column])}</td>' for column in columns)
        rows.append(f'<tr>{cells}</tr>')
    return f'<section><h2>{html.escape(title)}</h2><div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>'


def format_cell(column, value) -> str:
    text = format_value(value)
    if column == 'passed':
        css = 'pass' if str(value).lower() == 'true' else 'fail'
        return f'<span class="{css}">{html.escape(text)}</span>'
    return html.escape(text)


def format_value(value) -> str:
    if pd.isna(value):
        return ''
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


if __name__ == '__main__':
    main()
