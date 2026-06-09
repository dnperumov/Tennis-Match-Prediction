"""Daily pipeline orchestration for scraping, odds snapshots, and retraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tennis_ml.live_stats.database import (
    DEFAULT_DB_PATH,
    init_live_db,
    read_table,
    upsert_normalized_matches,
    upsert_odds_snapshots,
)
from tennis_ml.live_stats.tennis_abstract import scrape_tennis_abstract_daily
from tennis_ml.markets.kalshi import KalshiClient, normalize_kalshi_events

from .prediction import train_daily_model


def scrape_daily_matches(
    match_date: str | pd.Timestamp,
    db_path: str | Path = DEFAULT_DB_PATH,
    source: str | None = None,
    table_index: int | None = None,
) -> pd.DataFrame:
    rows = scrape_tennis_abstract_daily(match_date, source=source, table_index=table_index)
    upsert_normalized_matches(rows, db_path=db_path)
    return rows


def snapshot_kalshi_tennis(db_path: str | Path = DEFAULT_DB_PATH, max_pages: int = 10) -> pd.DataFrame:
    client = KalshiClient.from_env()
    events = client.get_open_tennis_events(max_pages=max_pages)
    markets = normalize_kalshi_events(events)
    upsert_odds_snapshots(markets, db_path=db_path)
    return markets


def run_daily_update(
    match_date: str | pd.Timestamp,
    db_path: str | Path = DEFAULT_DB_PATH,
    source: str | None = None,
    table_index: int | None = None,
    retrain: bool = False,
    training_start_year: int = 2000,
    fetch_kalshi: bool = True,
    refresh_dashboard: bool = False,
) -> dict[str, Any]:
    init_live_db(db_path)
    matches = scrape_daily_matches(match_date, db_path=db_path, source=source, table_index=table_index)
    odds = pd.DataFrame()
    odds_error = None
    if fetch_kalshi:
        try:
            odds = snapshot_kalshi_tennis(db_path=db_path)
        except Exception as exc:  # network/auth should not block local retraining
            odds_error = str(exc)

    model_run = None
    if retrain:
        model_run = train_daily_model(
            as_of_date=match_date,
            db_path=db_path,
            start_year=training_start_year,
        )

    dashboard_path = None
    if refresh_dashboard:
        dashboard_path = refresh_static_dashboard()

    return {
        'match_date': str(pd.to_datetime(match_date).date()),
        'db_path': str(db_path),
        'matches_ingested': int(len(matches)),
        'odds_snapshots': int(len(odds)),
        'odds_error': odds_error,
        'model_run': model_run,
        'dashboard_path': dashboard_path,
    }


def refresh_static_dashboard() -> str | None:
    try:
        from argparse import Namespace

        from dashboard_app import render_dashboard
    except ImportError:
        return None
    output = Path('dashboard/tennis_trading_dashboard.html')
    args = Namespace(
        walk_forward_dir='walk_forward_results_current',
        strategy_dir='strategy_eval_current',
        mining_dir='trade_mining_current',
        kalshi_file='data/kalshi/tennis_markets.csv',
        output=str(output),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_dashboard(args), encoding='utf-8')
    return str(output)


def daily_health(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, int]:
    return {
        'matches': len(read_table('matches', db_path=db_path)),
        'odds_snapshots': len(read_table('odds_snapshots', db_path=db_path)),
        'predictions': len(read_table('predictions', db_path=db_path)),
        'model_runs': len(read_table('model_runs', db_path=db_path)),
    }
