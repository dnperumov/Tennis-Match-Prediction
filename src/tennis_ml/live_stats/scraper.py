"""
Generic tennis match stats scraper/ingester.

The first live-data requirement is source flexibility: Tennis Abstract,
Tennis-Data exports, ATP pages, or hand-saved HTML/CSV files can expose
slightly different column names. This module normalizes common columns and
upserts them into a local CSV database that can later be replaced by SQLite.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


COLUMN_ALIASES = {
    'date': ['date', 'match_date', 'Date'],
    'tournament': ['tournament', 'tourney_name', 'Tournament'],
    'round': ['round', 'Round'],
    'surface': ['surface', 'Surface'],
    'player1': ['player1', 'player_1', 'winner', 'Winner', 'Player 1'],
    'player2': ['player2', 'player_2', 'loser', 'Loser', 'Player 2'],
    'winner': ['winner', 'Winner'],
    'score': ['score', 'Score'],
    'minutes': ['minutes', 'mins', 'Time', 'duration'],
    'player1_serve_points_won_pct': ['player1_serve_points_won_pct', 'w_svpt_won_pct', 'serve_points_won_pct'],
    'player1_return_points_won_pct': ['player1_return_points_won_pct', 'w_return_points_won_pct', 'return_points_won_pct'],
    'player2_serve_points_won_pct': ['player2_serve_points_won_pct', 'l_svpt_won_pct'],
    'player2_return_points_won_pct': ['player2_return_points_won_pct', 'l_return_points_won_pct'],
}

CANONICAL_COLUMNS = [
    'source',
    'source_match_id',
    'scraped_at',
    'date',
    'tournament',
    'round',
    'surface',
    'player1',
    'player2',
    'winner',
    'score',
    'minutes',
    'player1_serve_points_won_pct',
    'player1_return_points_won_pct',
    'player2_serve_points_won_pct',
    'player2_return_points_won_pct',
]


def scrape_match_stats(source: str, table_index: int | None = None, source_name: str = 'manual') -> pd.DataFrame:
    """Scrape CSV or HTML-table match stats from a local path or URL."""
    if source.lower().endswith('.csv'):
        raw = pd.read_csv(source)
    else:
        tables = pd.read_html(source)
        if not tables:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        raw = select_table(tables, table_index=table_index)
    return normalize_match_stats(raw, source=source_name)


def select_table(tables: list[pd.DataFrame], table_index: int | None = None) -> pd.DataFrame:
    if table_index is not None:
        return tables[table_index]
    scored = []
    wanted = {'winner', 'loser', 'player1', 'player2', 'date', 'round', 'surface'}
    for idx, table in enumerate(tables):
        columns = {str(column).strip().lower() for column in table.columns}
        score = len(columns & wanted)
        scored.append((score, idx, table))
    scored.sort(reverse=True, key=lambda item: (item[0], len(item[2])))
    return scored[0][2]


def normalize_match_stats(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = pd.DataFrame()
    for canonical, aliases in COLUMN_ALIASES.items():
        column = find_column(raw, aliases)
        frame[canonical] = raw[column] if column is not None else pd.NA

    frame['source'] = source
    frame['scraped_at'] = pd.Timestamp.utcnow().isoformat()
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce').dt.date
    frame['source_match_id'] = frame.apply(make_source_match_id, axis=1)
    frame = frame[CANONICAL_COLUMNS]
    return frame.dropna(subset=['player1', 'player2'], how='all').reset_index(drop=True)


def find_column(df: pd.DataFrame, aliases: list[str]):
    normalized = {str(column).strip().lower(): column for column in df.columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def make_source_match_id(row: pd.Series) -> str:
    parts = [
        row.get('source', ''),
        row.get('date', ''),
        row.get('tournament', ''),
        row.get('round', ''),
        row.get('player1', ''),
        row.get('player2', ''),
    ]
    return '|'.join(str(part).strip().lower() for part in parts)


def upsert_match_stats(new_rows: pd.DataFrame, database_path: str | Path) -> pd.DataFrame:
    """Append/update a CSV match-stats database by source_match_id."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
    else:
        existing = pd.DataFrame(columns=CANONICAL_COLUMNS)
    combined = pd.concat([existing, new_rows], ignore_index=True)
    if 'source_match_id' not in combined.columns:
        combined['source_match_id'] = combined.apply(make_source_match_id, axis=1)
    combined = combined.drop_duplicates(subset=['source_match_id'], keep='last')
    combined = combined.reindex(columns=CANONICAL_COLUMNS)
    combined.to_csv(path, index=False)
    return combined
