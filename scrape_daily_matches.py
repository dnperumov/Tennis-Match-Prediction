#!/usr/bin/env python3
"""Scrape and store daily tennis match results/stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.daily import scrape_daily_matches


def main() -> None:
    parser = argparse.ArgumentParser(description='Scrape daily ATP match stats into SQLite.')
    parser.add_argument('--date', required=True, help='Match date, e.g. 2026-05-31')
    parser.add_argument('--db-path', default='data/live/tennis_live.db')
    parser.add_argument('--source', help='Optional Tennis Abstract CSV/HTML file or URL.')
    parser.add_argument('--table-index', type=int, help='Optional HTML table index.')
    args = parser.parse_args()

    rows = scrape_daily_matches(args.date, db_path=args.db_path, source=args.source, table_index=args.table_index)
    print(f'Ingested {len(rows)} normalized rows into {args.db_path}')


if __name__ == '__main__':
    main()
