#!/usr/bin/env python3
"""Run the daily tennis ML update pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.daily import run_daily_update


def main() -> None:
    parser = argparse.ArgumentParser(description='Daily scrape, odds snapshot, retrain, and dashboard refresh.')
    parser.add_argument('--date', required=True, help='Pipeline date, e.g. 2026-05-31')
    parser.add_argument('--db-path', default='data/live/tennis_live.db')
    parser.add_argument('--source', help='Optional Tennis Abstract CSV/HTML file or URL.')
    parser.add_argument('--table-index', type=int)
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--refresh-dashboard', action='store_true')
    parser.add_argument('--no-kalshi', action='store_true', help='Skip Kalshi market snapshot.')
    args = parser.parse_args()

    summary = run_daily_update(
        args.date,
        db_path=args.db_path,
        source=args.source,
        table_index=args.table_index,
        retrain=args.retrain,
        fetch_kalshi=not args.no_kalshi,
        refresh_dashboard=args.refresh_dashboard,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == '__main__':
    main()
