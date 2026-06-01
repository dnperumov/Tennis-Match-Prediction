#!/usr/bin/env python3
"""
Fetch open Kalshi tennis markets into a normalized CSV.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.markets import KalshiClient, normalize_kalshi_events


def main():
    parser = argparse.ArgumentParser(description='Fetch open Kalshi tennis markets')
    parser.add_argument('--output', default='data/kalshi/tennis_markets.csv')
    parser.add_argument('--base-url', default='https://external-api.kalshi.com/trade-api/v2')
    parser.add_argument('--max-pages', type=int, default=10)
    args = parser.parse_args()

    client = KalshiClient(base_url=args.base_url)
    events = client.get_open_tennis_events(max_pages=args.max_pages)
    markets = normalize_kalshi_events(events)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    markets.to_csv(output, index=False)
    print(f'Wrote {len(markets)} Kalshi tennis markets to: {output}')


if __name__ == '__main__':
    main()
