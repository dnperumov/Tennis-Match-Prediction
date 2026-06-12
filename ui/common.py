"""Shared helpers for the Streamlit control center: settings, loaders, formatting."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

SETTINGS_PATH = Path('data/live/app_settings.json')
DEFAULT_BANKROLL = 1000.0

RESULTS_DIR_CANDIDATES = [
    'walk_forward_results_stacked_avg',
    'walk_forward_results_stacked',
    'walk_forward_results',
]

TIER_BADGES = {
    'value_bet': ':green-badge[VALUE BET]',
    'confident_pick': ':blue-badge[CONFIDENT]',
    'lean': ':orange-badge[LEAN]',
    'no_bet': ':gray-badge[NO BET]',
}

TIER_ORDER = {'value_bet': 0, 'confident_pick': 1, 'lean': 2, 'no_bet': 3}


# ---------------------------------------------------------------- settings

def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_settings(settings: dict) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    except Exception:
        pass  # settings persistence is best-effort


def get_bankroll() -> float:
    if 'bankroll' not in st.session_state:
        st.session_state['bankroll'] = float(load_settings().get('bankroll', DEFAULT_BANKROLL))
    return float(st.session_state['bankroll'])


def set_bankroll(value: float) -> None:
    st.session_state['bankroll'] = float(value)
    settings = load_settings()
    settings['bankroll'] = float(value)
    save_settings(settings)


# ---------------------------------------------------------------- formatting

def pct(value, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return '—'
    return f'{value:.{digits}%}'


def money(value) -> str:
    if value is None or pd.isna(value):
        return '—'
    return f'${value:,.2f}'


def cents(price) -> str:
    """Kalshi price (0-1 dollars) shown as cents."""
    if price is None or pd.isna(price):
        return '—'
    return f'{price * 100:.0f}¢'


def tier_badge(tier: str) -> str:
    return TIER_BADGES.get(str(tier), f':gray-badge[{tier}]')


# ---------------------------------------------------------------- DB access

def safe_read_table(table: str, db_path: str) -> pd.DataFrame:
    """Read a live-DB table; empty frame (never a traceback) on any failure."""
    try:
        from tennis_ml.live_stats.database import read_table

        return read_table(table, db_path=db_path)
    except Exception:
        return pd.DataFrame()


def safe_fetch_picks(db_path: str, status: str | None = None) -> pd.DataFrame:
    try:
        from tennis_ml.live_stats.database import fetch_picks

        return fetch_picks(status=status, db_path=db_path)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------- walk-forward results

def existing_results_dirs() -> list[str]:
    dirs = [d for d in RESULTS_DIR_CANDIDATES if (Path(d) / 'predictions.csv').exists()]
    # include any other walk_forward_results_* dirs with predictions.csv
    for path in sorted(Path('.').glob('walk_forward_results*')):
        if path.is_dir() and (path / 'predictions.csv').exists() and str(path) not in dirs:
            dirs.append(str(path))
    return dirs


PREDICTION_COLUMNS = [
    'player1', 'player2', 'winner_name', 'tourney_name', 'surface', 'round',
    'player1_probability', 'player1_market_probability_novig',
    'prediction_correct', 'test_year', 'model_type',
]


@st.cache_data(show_spinner='Loading walk-forward predictions...')
def load_predictions(results_dir: str, mtime: float) -> pd.DataFrame:
    path = Path(results_dir) / 'predictions.csv'
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in PREDICTION_COLUMNS if c in header]
    df = pd.read_csv(path, usecols=usecols)
    # Some walk-forward rows are never scored (no fair probability) — drop them.
    df = df[df['player1_probability'].notna()].reset_index(drop=True)
    # Recompute correctness consistently from the fair probability.
    df['p1_won'] = df['winner_name'] == df['player1']
    df['model_correct'] = (df['player1_probability'] >= 0.5) == df['p1_won']
    df['confidence'] = df['player1_probability'].where(
        df['player1_probability'] >= 0.5, 1 - df['player1_probability']
    )
    if 'player1_market_probability_novig' in df.columns:
        market = df['player1_market_probability_novig']
        market_correct = ((market >= 0.5) == df['p1_won']).astype('boolean')
        market_correct[market.isna()] = pd.NA
        df['market_correct'] = market_correct
    else:
        df['market_correct'] = pd.Series(pd.NA, index=df.index, dtype='boolean')
    return df


@st.cache_data(show_spinner=False)
def load_results_csv(results_dir: str, name: str, mtime: float) -> pd.DataFrame:
    path = Path(results_dir) / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def file_mtime(path: str | Path) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0
