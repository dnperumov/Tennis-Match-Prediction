#!/usr/bin/env python3
"""Score a future tennis match with the latest daily model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.daily import MatchPredictionRequest, TennisPredictionService


def main() -> None:
    parser = argparse.ArgumentParser(description='Predict a tennis match and report fair odds / bet decision.')
    parser.add_argument('--player1', required=True)
    parser.add_argument('--player2', required=True)
    parser.add_argument('--surface', default='Hard')
    parser.add_argument('--tournament-level', default='A', choices=['G', 'M', 'A', 'C', 'F', 'D'])
    parser.add_argument('--round', default='R32')
    parser.add_argument('--date')
    parser.add_argument('--player1-odds', type=float)
    parser.add_argument('--player2-odds', type=float)
    parser.add_argument('--model-dir')
    parser.add_argument('--db-path', default='data/live/tennis_live.db')
    parser.add_argument('--no-persist', action='store_true')
    args = parser.parse_args()

    request = MatchPredictionRequest(
        player1=args.player1,
        player2=args.player2,
        surface=args.surface,
        tournament_level=args.tournament_level,
        round=args.round,
        match_date=args.date,
        player1_odds=args.player1_odds,
        player2_odds=args.player2_odds,
        model_dir=args.model_dir,
    )
    service = TennisPredictionService(db_path=args.db_path, model_dir=args.model_dir)
    result = service.predict(request, persist=not args.no_persist)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == '__main__':
    main()
