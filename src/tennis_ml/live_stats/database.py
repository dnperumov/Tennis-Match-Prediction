"""SQLite storage for live tennis data, odds snapshots, and predictions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DB_PATH = Path('data/live/tennis_live.db')


SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    source_match_id TEXT PRIMARY KEY,
    source TEXT,
    scraped_at TEXT,
    match_date TEXT,
    tournament TEXT,
    round TEXT,
    surface TEXT,
    player1 TEXT,
    player2 TEXT,
    winner TEXT,
    score TEXT,
    minutes REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS match_stats (
    source_match_id TEXT PRIMARY KEY,
    player1_serve_points_won_pct REAL,
    player1_return_points_won_pct REAL,
    player2_serve_points_won_pct REAL,
    player2_return_points_won_pct REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_match_id) REFERENCES matches(source_match_id)
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    snapshot_time TEXT,
    source TEXT,
    event_ticker TEXT,
    market_ticker TEXT,
    event_title TEXT,
    market_title TEXT,
    yes_sub_title TEXT,
    no_sub_title TEXT,
    occurrence_datetime TEXT,
    close_time TEXT,
    yes_bid REAL,
    yes_ask REAL,
    no_bid REAL,
    no_ask REAL,
    yes_decimal_odds REAL,
    no_decimal_odds REAL,
    liquidity_dollars REAL,
    volume REAL,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    prediction_time TEXT,
    model_run_id TEXT,
    match_date TEXT,
    player1 TEXT,
    player2 TEXT,
    surface TEXT,
    tournament_level TEXT,
    round TEXT,
    player1_probability REAL,
    player2_probability REAL,
    fair_odds_player1 REAL,
    fair_odds_player2 REAL,
    market_odds_player1 REAL,
    market_odds_player2 REAL,
    recommended_side TEXT,
    strategy_decision TEXT,
    edge REAL,
    ev REAL,
    confidence TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS paper_trades (
    paper_trade_id TEXT PRIMARY KEY,
    prediction_id TEXT,
    created_at TEXT,
    settled_at TEXT,
    player TEXT,
    opponent TEXT,
    odds REAL,
    stake REAL,
    won INTEGER,
    profit REAL,
    friction_adjusted_profit REAL,
    clv REAL,
    status TEXT
);

CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id TEXT PRIMARY KEY,
    trained_at TEXT,
    as_of_date TEXT,
    artifact_dir TEXT,
    training_rows INTEGER,
    validation_rows INTEGER,
    metrics_json TEXT,
    notes TEXT
);
"""


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_live_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    with connect(db_path) as connection:
        connection.executescript(SCHEMA)


def upsert_normalized_matches(rows: pd.DataFrame, db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Upsert normalized scraper rows into matches and match_stats."""
    if rows is None or rows.empty:
        init_live_db(db_path)
        return 0

    init_live_db(db_path)
    rows = rows.copy()
    rows['match_date'] = pd.to_datetime(rows.get('date'), errors='coerce').dt.strftime('%Y-%m-%d')
    with connect(db_path) as connection:
        for _, row in rows.iterrows():
            source_match_id = str(row.get('source_match_id') or '').strip()
            if not source_match_id:
                continue
            connection.execute(
                """
                INSERT INTO matches (
                    source_match_id, source, scraped_at, match_date, tournament, round,
                    surface, player1, player2, winner, score, minutes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(source_match_id) DO UPDATE SET
                    source=excluded.source,
                    scraped_at=excluded.scraped_at,
                    match_date=excluded.match_date,
                    tournament=excluded.tournament,
                    round=excluded.round,
                    surface=excluded.surface,
                    player1=excluded.player1,
                    player2=excluded.player2,
                    winner=excluded.winner,
                    score=excluded.score,
                    minutes=excluded.minutes,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    source_match_id,
                    _clean(row.get('source')),
                    _clean(row.get('scraped_at')),
                    _clean(row.get('match_date')),
                    _clean(row.get('tournament')),
                    _clean(row.get('round')),
                    _clean(row.get('surface')),
                    _clean(row.get('player1')),
                    _clean(row.get('player2')),
                    _clean(row.get('winner')),
                    _clean(row.get('score')),
                    _float(row.get('minutes')),
                ),
            )
            connection.execute(
                """
                INSERT INTO match_stats (
                    source_match_id,
                    player1_serve_points_won_pct,
                    player1_return_points_won_pct,
                    player2_serve_points_won_pct,
                    player2_return_points_won_pct,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(source_match_id) DO UPDATE SET
                    player1_serve_points_won_pct=excluded.player1_serve_points_won_pct,
                    player1_return_points_won_pct=excluded.player1_return_points_won_pct,
                    player2_serve_points_won_pct=excluded.player2_serve_points_won_pct,
                    player2_return_points_won_pct=excluded.player2_return_points_won_pct,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    source_match_id,
                    _float(row.get('player1_serve_points_won_pct')),
                    _float(row.get('player1_return_points_won_pct')),
                    _float(row.get('player2_serve_points_won_pct')),
                    _float(row.get('player2_return_points_won_pct')),
                ),
            )
    return int(len(rows))


def upsert_odds_snapshots(rows: pd.DataFrame, db_path: str | Path = DEFAULT_DB_PATH) -> int:
    if rows is None or rows.empty:
        init_live_db(db_path)
        return 0

    init_live_db(db_path)
    with connect(db_path) as connection:
        for _, row in rows.iterrows():
            snapshot_id = make_snapshot_id(row)
            connection.execute(
                """
                INSERT INTO odds_snapshots (
                    snapshot_id, snapshot_time, source, event_ticker, market_ticker,
                    event_title, market_title, yes_sub_title, no_sub_title,
                    occurrence_datetime, close_time, yes_bid, yes_ask, no_bid, no_ask,
                    yes_decimal_odds, no_decimal_odds, liquidity_dollars, volume, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    yes_bid=excluded.yes_bid,
                    yes_ask=excluded.yes_ask,
                    no_bid=excluded.no_bid,
                    no_ask=excluded.no_ask,
                    yes_decimal_odds=excluded.yes_decimal_odds,
                    no_decimal_odds=excluded.no_decimal_odds,
                    liquidity_dollars=excluded.liquidity_dollars,
                    volume=excluded.volume,
                    raw_json=excluded.raw_json
                """,
                (
                    snapshot_id,
                    _clean(row.get('snapshot_time')),
                    _clean(row.get('source', 'kalshi')),
                    _clean(row.get('event_ticker')),
                    _clean(row.get('market_ticker')),
                    _clean(row.get('event_title')),
                    _clean(row.get('market_title')),
                    _clean(row.get('yes_sub_title')),
                    _clean(row.get('no_sub_title')),
                    _clean(row.get('occurrence_datetime')),
                    _clean(row.get('close_time')),
                    _float(row.get('yes_bid')),
                    _float(row.get('yes_ask')),
                    _float(row.get('no_bid')),
                    _float(row.get('no_ask')),
                    _float(row.get('yes_decimal_odds')),
                    _float(row.get('no_decimal_odds')),
                    _float(row.get('liquidity_dollars')),
                    _float(row.get('volume')),
                    json.dumps(_public_row(row), sort_keys=True),
                ),
            )
    return int(len(rows))


def insert_prediction(prediction: dict[str, Any], db_path: str | Path = DEFAULT_DB_PATH) -> None:
    init_live_db(db_path)
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                prediction_id, prediction_time, model_run_id, match_date, player1,
                player2, surface, tournament_level, round, player1_probability,
                player2_probability, fair_odds_player1, fair_odds_player2,
                market_odds_player1, market_odds_player2, recommended_side,
                strategy_decision, edge, ev, confidence, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prediction_id) DO UPDATE SET
                prediction_time=excluded.prediction_time,
                model_run_id=excluded.model_run_id,
                player1_probability=excluded.player1_probability,
                player2_probability=excluded.player2_probability,
                fair_odds_player1=excluded.fair_odds_player1,
                fair_odds_player2=excluded.fair_odds_player2,
                market_odds_player1=excluded.market_odds_player1,
                market_odds_player2=excluded.market_odds_player2,
                recommended_side=excluded.recommended_side,
                strategy_decision=excluded.strategy_decision,
                edge=excluded.edge,
                ev=excluded.ev,
                confidence=excluded.confidence,
                notes=excluded.notes
            """,
            tuple(prediction.get(key) for key in [
                'prediction_id', 'prediction_time', 'model_run_id', 'match_date',
                'player1', 'player2', 'surface', 'tournament_level', 'round',
                'player1_probability', 'player2_probability', 'fair_odds_player1',
                'fair_odds_player2', 'market_odds_player1', 'market_odds_player2',
                'recommended_side', 'strategy_decision', 'edge', 'ev',
                'confidence', 'notes',
            ]),
        )


def insert_model_run(run: dict[str, Any], db_path: str | Path = DEFAULT_DB_PATH) -> None:
    init_live_db(db_path)
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO model_runs (
                model_run_id, trained_at, as_of_date, artifact_dir, training_rows,
                validation_rows, metrics_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_run_id) DO UPDATE SET
                trained_at=excluded.trained_at,
                artifact_dir=excluded.artifact_dir,
                training_rows=excluded.training_rows,
                validation_rows=excluded.validation_rows,
                metrics_json=excluded.metrics_json,
                notes=excluded.notes
            """,
            (
                run.get('model_run_id'),
                run.get('trained_at'),
                run.get('as_of_date'),
                run.get('artifact_dir'),
                run.get('training_rows'),
                run.get('validation_rows'),
                json.dumps(run.get('metrics', {}), sort_keys=True),
                run.get('notes'),
            ),
        )


def read_table(table_name: str, db_path: str | Path = DEFAULT_DB_PATH, limit: int | None = None) -> pd.DataFrame:
    init_live_db(db_path)
    query = f'SELECT * FROM {table_name}'
    if limit is not None:
        query += f' LIMIT {int(limit)}'
    with connect(db_path) as connection:
        return pd.read_sql_query(query, connection)


def latest_model_run(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    init_live_db(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            'SELECT * FROM model_runs ORDER BY trained_at DESC LIMIT 1'
        ).fetchone()
    return dict(row) if row else None


def make_snapshot_id(row: pd.Series) -> str:
    parts = [
        row.get('source', 'kalshi'),
        row.get('snapshot_time'),
        row.get('market_ticker'),
        row.get('yes_ask'),
        row.get('no_ask'),
    ]
    return '|'.join(str(part or '').strip() for part in parts)


def _clean(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _public_row(row: pd.Series) -> dict[str, Any]:
    blocked = {'api_key', 'private_key', 'signature', 'authorization'}
    result = {}
    for key, value in row.to_dict().items():
        if any(token in str(key).lower() for token in blocked):
            continue
        result[key] = None if pd.isna(value) else value
    return result
