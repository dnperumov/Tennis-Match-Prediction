#!/usr/bin/env python3
"""
Scrape new tennis match stats and upsert them into a local CSV database.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.live_stats import scrape_match_stats, upsert_match_stats


def main():
    parser = argparse.ArgumentParser(description='Scrape tennis match stats into a local database')
    parser.add_argument('--source', required=True,
                        help='CSV/HTML file path or URL containing match stats table(s)')
    parser.add_argument('--source-name', default='manual')
    parser.add_argument('--table-index', type=int)
    parser.add_argument('--database', default='data/live/match_stats.csv')
    parser.add_argument('--output')
    args = parser.parse_args()

    rows = scrape_match_stats(args.source, table_index=args.table_index, source_name=args.source_name)
    database = upsert_match_stats(rows, args.database)
    if args.output:
        rows.to_csv(args.output, index=False)
    print(f'Scraped {len(rows)} rows; database now has {len(database)} rows: {args.database}')


if __name__ == '__main__':
    main()
