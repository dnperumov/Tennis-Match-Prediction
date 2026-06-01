"""Tennis Abstract-first daily ingestion helpers.

Tennis Abstract pages and locally exported tables are not a single stable API.
This module provides a source-configurable v1: pass a Tennis Abstract style
CSV/HTML URL or file when available, otherwise it falls back to known local
Tennis Abstract exports in this repository.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .scraper import CANONICAL_COLUMNS, scrape_match_stats


LOCAL_TA_EXPORTS = [
    Path('ta_tourney_scrape/data/atp_matches_2025_COMPLETE.csv'),
    Path('ta_tourney_scrape/data/atp_matches_2025_ULTIMATE_COMPLETE.csv'),
    Path('ta_tourney_scrape/data/atp_matches_2025_from_tennisabstract.csv'),
]


def scrape_tennis_abstract_daily(
    match_date: str | pd.Timestamp,
    source: str | None = None,
    table_index: int | None = None,
) -> pd.DataFrame:
    """Return normalized Tennis Abstract rows for one date."""
    target = pd.to_datetime(match_date).date()
    if source:
        rows = scrape_match_stats(source, table_index=table_index, source_name='tennis_abstract')
        return _filter_date(rows, target)

    for path in LOCAL_TA_EXPORTS:
        if path.exists():
            rows = scrape_match_stats(str(path), source_name='tennis_abstract_local')
            filtered = _filter_date(rows, target)
            if not filtered.empty:
                return filtered
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def _filter_date(rows: pd.DataFrame, target) -> pd.DataFrame:
    if rows.empty or 'date' not in rows.columns:
        return rows
    dates = pd.to_datetime(rows['date'], errors='coerce').dt.date
    return rows[dates == target].reset_index(drop=True)
