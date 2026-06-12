"""Today's matchup board: Kalshi head-to-head markets x model evaluation.

Builds one row per scheduled match from Kalshi's per-match series
(KXATPMATCH), scores each matchup with the production stacked model, applies
the strategy gates, and persists snapshots/predictions/picks so the rest of
the pipeline (paper trading, settlement) sees the same data.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tennis_ml.live_stats.database import DEFAULT_DB_PATH, init_live_db, upsert_odds_snapshots
from tennis_ml.markets.kalshi import (
    ATP_MATCH_SERIES,
    KalshiClient,
    normalize_kalshi_events,
    parse_match_event,
)
from tennis_ml.strategy import StrategyConfig, generate_picks, persist_picks

from .prediction import MatchPredictionRequest, TennisPredictionService


def infer_surface(board_date: date) -> str:
    """Calendar heuristic for the dominant tour surface; UI allows override."""
    month, day = board_date.month, board_date.day
    if 4 <= month <= 5:
        return 'Clay'
    if month == 6 or (month == 7 and day <= 14):
        return 'Grass'
    return 'Hard'


def fetch_matchups(
    client: KalshiClient | None = None,
    series_ticker: str = ATP_MATCH_SERIES,
    board_date: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict]]:
    """Fetch and parse match events; returns (matchups, raw events kept)."""
    client = client or KalshiClient.from_env()
    events = client.get_match_events(series_ticker=series_ticker)
    matchups = []
    kept_events = []
    for event in events:
        parsed = parse_match_event(event)
        if parsed is None:
            continue
        if board_date and parsed.get('match_date') != board_date:
            continue
        matchups.append(parsed)
        kept_events.append(event)
    return matchups, kept_events


def build_matchup_board(
    db_path: str | Path = DEFAULT_DB_PATH,
    model_dir: str | Path | None = None,
    board_date: str | None = None,
    surface: str | None = None,
    config: StrategyConfig | None = None,
    series_ticker: str = ATP_MATCH_SERIES,
    client: KalshiClient | None = None,
    service: TennisPredictionService | None = None,
) -> pd.DataFrame:
    """One row per Kalshi matchup on board_date, scored and strategy-evaluated.

    Side effects: snapshots markets into odds_snapshots, persists predictions,
    and persists generated picks (idempotent upserts).
    """
    board_date = board_date or datetime.now(timezone.utc).date().isoformat()
    surface = surface or infer_surface(date.fromisoformat(board_date))
    config = config or StrategyConfig()

    init_live_db(db_path)
    matchups, kept_events = fetch_matchups(client, series_ticker=series_ticker, board_date=board_date)
    if kept_events:
        upsert_odds_snapshots(normalize_kalshi_events(kept_events), db_path=db_path)
    if not matchups:
        return pd.DataFrame()

    service = service or TennisPredictionService(db_path=db_path, model_dir=model_dir)

    rows = []
    candidates = []
    for matchup in matchups:
        price = matchup.get('player1_yes_ask')
        if price is None:
            price = matchup.get('player1_last_price')
        row = dict(matchup)
        row['surface'] = surface
        try:
            result = service.predict(MatchPredictionRequest(
                player1=matchup['player1'],
                player2=matchup['player2'],
                surface=surface,
                match_date=matchup.get('match_date'),
                kalshi_yes_price=float(price) if price is not None else None,
            ))
        except Exception as exc:  # one bad matchup must not kill the board
            row['error'] = str(exc)
            rows.append(row)
            continue

        row['player1_probability'] = result['player1_probability']
        row['player2_probability'] = result['player2_probability']
        row['model_confidence'] = result.get('confidence')
        row['prediction_id'] = result.get('prediction_id')
        row['model_kind'] = result.get('model_kind')
        rows.append(row)

        candidates.append({
            'match': {
                'player1': matchup['player1'],
                'player2': matchup['player2'],
                'match_date': matchup.get('match_date'),
                'tournament': matchup.get('title'),
                'kalshi_ticker': matchup.get('player1_market_ticker'),
                'prediction_id': result.get('prediction_id'),
            },
            'fair_prob': result['player1_probability'],
            'kalshi_price': float(price) if price is not None else None,
            'confidence': result.get('confidence'),
        })

    picks = generate_picks(candidates, config) if candidates else []
    if picks:
        persist_picks(picks, db_path=db_path)

    board = pd.DataFrame(rows)
    if picks:
        pick_frame = pd.DataFrame([{
            'prediction_id': pick['prediction_id'],
            'pick_player': pick['player'],
            'pick_side': pick['side'],
            'confidence_tier': pick['confidence_tier'],
            'net_ev': pick['net_ev'],
            'stake_suggested': pick['stake_suggested'],
            'pick_reasons': pick['reasons'],
        } for pick in picks])
        board = board.merge(pick_frame, on='prediction_id', how='left')
    return board
