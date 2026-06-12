#!/usr/bin/env python
"""Refresh ATP training data.

Two-stage, idempotent data-ingestion bridge:

1. Download the current year's (and, in January, the previous year's)
   ``atp_matches_YYYY.csv`` from Jeff Sackmann's tennis_atp GitHub repo and
   overwrite the local copy in ``tennis_datav2/`` -- but only when the
   download is valid CSV with the expected columns and has at least as many
   rows as the existing local file (protects against upstream truncation).

2. Read completed singles matches from the Hermes agent database
   (``data/tennis_matches.sqlite``) that fall AFTER the Sackmann coverage
   cutoff, normalize them to the Sackmann column layout, dedup them against
   the Sackmann file, and write ``tennis_datav2/atp_matches_{YEAR}_supplement.csv``
   (overwritten wholesale each run -- it is derived data).

When Sackmann publishes new data, the cutoff moves forward and supplement
rows for now-covered dates drop out automatically on the next run.

Usage:
    .venv/bin/python scripts/refresh_match_data.py
    .venv/bin/python scripts/refresh_match_data.py --skip-download
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sqlite3
import sys
import unicodedata
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, 'tennis_datav2')
DEFAULT_DB_PATH = os.path.join(REPO_ROOT, 'data', 'tennis_matches.sqlite')
SACKMANN_URL = 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv'

# Canonical Sackmann atp_matches column layout.
SACKMANN_COLUMNS = [
    'tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level',
    'tourney_date', 'match_num', 'winner_id', 'winner_seed', 'winner_entry',
    'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
    'loser_id', 'loser_seed', 'loser_entry', 'loser_name', 'loser_hand',
    'loser_ht', 'loser_ioc', 'loser_age', 'score', 'best_of', 'round',
    'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
    'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn',
    'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
    'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points',
]

# Columns that must be present for a downloaded file to be considered valid.
REQUIRED_COLUMNS = {
    'tourney_id', 'tourney_name', 'surface', 'tourney_date',
    'winner_name', 'loser_name', 'score', 'round', 'best_of',
}

GRAND_SLAMS = {'australian open', 'roland garros', 'wimbledon', 'us open'}
MASTERS = {
    'indian wells masters', 'miami masters', 'monte carlo masters',
    'madrid masters', 'rome masters', 'canada masters', 'cincinnati masters',
    'shanghai masters', 'paris masters', 'indian wells', 'miami',
    'monte carlo', 'madrid', 'rome', 'montreal', 'toronto', 'cincinnati',
    'shanghai',
}

DEDUP_WINDOW_DAYS = 14


# ---------------------------------------------------------------------------
# Step (a): refresh Sackmann files
# ---------------------------------------------------------------------------

def years_to_refresh(today: date) -> list[int]:
    years = [today.year]
    if today.month == 1:
        years.insert(0, today.year - 1)
    return years


def refresh_sackmann_file(year: int, data_dir: str) -> bool:
    """Download and safely overwrite one Sackmann yearly file.

    Returns True if the local file was updated.
    """
    local_path = os.path.join(data_dir, f'atp_matches_{year}.csv')
    local_rows = None
    if os.path.exists(local_path):
        try:
            local_rows = len(pd.read_csv(local_path, low_memory=False))
        except Exception as exc:
            print(f'[sackmann {year}] WARNING: local file unreadable ({exc}); will replace if download is valid')
    print(f'[sackmann {year}] local rows before: {local_rows if local_rows is not None else "none"}')

    url = SACKMANN_URL.format(year=year)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read().decode('utf-8')
    except Exception as exc:
        print(f'[sackmann {year}] download failed ({exc}); keeping local file')
        return False

    try:
        remote = pd.read_csv(io.StringIO(raw), low_memory=False)
    except Exception as exc:
        print(f'[sackmann {year}] downloaded file is not valid CSV ({exc}); keeping local file')
        return False

    missing = REQUIRED_COLUMNS - set(remote.columns)
    if missing:
        print(f'[sackmann {year}] downloaded file missing columns {sorted(missing)}; keeping local file')
        return False

    if local_rows is not None and len(remote) < local_rows:
        print(f'[sackmann {year}] downloaded file has {len(remote)} rows < local {local_rows}; '
              'refusing to overwrite (possible upstream truncation)')
        return False

    os.makedirs(data_dir, exist_ok=True)
    with open(local_path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(raw)
    print(f'[sackmann {year}] local rows after: {len(remote)} (updated)')
    return True


# ---------------------------------------------------------------------------
# Step (b): coverage cutoff
# ---------------------------------------------------------------------------

def compute_cutoff(years: list[int], data_dir: str) -> pd.Timestamp | None:
    """Max date covered by the local Sackmann files for the given years.

    Sackmann only carries ``tourney_date`` (tournament start), so the cutoff
    is max(tourney_date). Supplement rows may overlap the cutoff tournament;
    dedup handles the overlap.
    """
    max_date = None
    for year in years:
        path = os.path.join(data_dir, f'atp_matches_{year}.csv')
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=['tourney_date'], low_memory=False)
        except Exception as exc:
            print(f'[cutoff] could not read {path}: {exc}')
            continue
        dates = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
        candidate = dates.max()
        if pd.notna(candidate) and (max_date is None or candidate > max_date):
            max_date = candidate
    return max_date


# ---------------------------------------------------------------------------
# Step (c): read + normalize agent (Hermes) matches
# ---------------------------------------------------------------------------

def load_agent_matches(db_path: str) -> pd.DataFrame:
    """Load completed ATP singles matches from the agent sqlite.

    Effective date = match_date when populated, else date(first_seen_utc)
    (the day the agent first saw the completed result).
    """
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT m.match_key, m.tournament, m.match_date, m.round,
                   m.winner_name, m.loser_name, m.score, m.best_of,
                   m.surface, m.first_seen_utc,
                   w.aces AS w_ace, w.double_faults AS w_df,
                   w.service_points_total AS w_svpt,
                   w.first_serve_in AS w_1stIn,
                   w.first_serve_points_won AS w_1stWon,
                   w.second_serve_points_won AS w_2ndWon,
                   w.break_points_saved AS w_bpSaved,
                   w.break_points_faced AS w_bpFaced,
                   l.aces AS l_ace, l.double_faults AS l_df,
                   l.service_points_total AS l_svpt,
                   l.first_serve_in AS l_1stIn,
                   l.first_serve_points_won AS l_1stWon,
                   l.second_serve_points_won AS l_2ndWon,
                   l.break_points_saved AS l_bpSaved,
                   l.break_points_faced AS l_bpFaced
            FROM matches m
            LEFT JOIN match_side_stats w
                   ON w.match_key = m.match_key AND w.side = 'winner'
            LEFT JOIN match_side_stats l
                   ON l.match_key = m.match_key AND l.side = 'loser'
            WHERE m.status = 'completed' AND m.tour = 'ATP'
            """,
            con,
        )
    finally:
        con.close()

    match_date = pd.to_datetime(df['match_date'], errors='coerce')
    first_seen = pd.to_datetime(df['first_seen_utc'], errors='coerce', utc=True)
    first_seen = first_seen.dt.tz_localize(None).dt.normalize()
    df['effective_date'] = match_date.fillna(first_seen)
    df = df[df['effective_date'].notna()].copy()
    return df


def guess_tourney_level(tournament: str) -> str:
    name = (tournament or '').strip().lower()
    if name in GRAND_SLAMS:
        return 'G'
    if name in MASTERS:
        return 'M'
    if 'davis cup' in name:
        return 'D'
    if 'tour finals' in name or name == 'atp finals':
        return 'F'
    return 'A'


def normalize_to_sackmann(agent: pd.DataFrame) -> pd.DataFrame:
    """Map agent rows to the Sackmann atp_matches column layout."""
    out = pd.DataFrame(index=agent.index, columns=SACKMANN_COLUMNS, dtype=object)

    year = agent['effective_date'].dt.year
    slug = agent['tournament'].fillna('unknown').map(
        lambda t: re.sub(r'[^a-z0-9]+', '-', str(t).lower()).strip('-'))
    out['tourney_id'] = year.astype(str) + '-SUPP-' + slug
    out['tourney_name'] = agent['tournament']
    out['surface'] = agent['surface'].replace('', np.nan)
    out['tourney_level'] = agent['tournament'].map(guess_tourney_level)
    # Sackmann tourney_date is YYYYMMDD; use the (approximate) match date.
    out['tourney_date'] = agent['effective_date'].dt.strftime('%Y%m%d').astype(int)
    out['match_num'] = range(1, len(agent) + 1)
    out['winner_name'] = agent['winner_name']
    out['loser_name'] = agent['loser_name']
    out['score'] = agent['score']
    best_of = pd.to_numeric(agent['best_of'], errors='coerce').fillna(3).astype(int)
    out['best_of'] = best_of
    out['round'] = agent['round']

    # Side stats: pass through whatever the agent captured (usually empty).
    for col in ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
                'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn',
                'l_1stWon', 'l_2ndWon', 'l_bpSaved', 'l_bpFaced']:
        out[col] = pd.to_numeric(agent[col], errors='coerce')

    return out


# ---------------------------------------------------------------------------
# Step (e): dedup against Sackmann
# ---------------------------------------------------------------------------

def _last_name_key(name: str) -> str:
    """Normalized last-name key tolerant of hyphen/space and accent variants.

    Sackmann writes 'Jan-Lennard Struff'; TennisAbstract writes
    'Jan Lennard Struff'. Treating hyphens as spaces and taking the last
    token makes both resolve to 'struff'.
    """
    if not isinstance(name, str) or not name.strip():
        return ''
    s = unicodedata.normalize('NFKD', name)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('-', ' ').lower().strip()
    tokens = s.split()
    return tokens[-1] if tokens else ''


def dedup_supplement(supplement: pd.DataFrame, sackmann: pd.DataFrame) -> pd.DataFrame:
    """Drop supplement rows already covered by Sackmann.

    A row is a duplicate when a Sackmann row matches on the winner's and
    loser's normalized last names and the date is within DEDUP_WINDOW_DAYS.
    """
    if supplement.empty or sackmann.empty:
        return supplement

    sack = pd.DataFrame({
        'w_key': sackmann['winner_name'].map(_last_name_key),
        'l_key': sackmann['loser_name'].map(_last_name_key),
        'date': pd.to_datetime(sackmann['tourney_date'].astype(str), format='%Y%m%d', errors='coerce'),
    })
    sack_index: dict[tuple[str, str], list[pd.Timestamp]] = {}
    for w, l, d in zip(sack['w_key'], sack['l_key'], sack['date']):
        if pd.isna(d):
            continue
        sack_index.setdefault((w, l), []).append(d)

    supp_dates = pd.to_datetime(supplement['tourney_date'].astype(str), format='%Y%m%d', errors='coerce')
    keep_mask = []
    for (_, row), d in zip(supplement.iterrows(), supp_dates):
        key = (_last_name_key(row['winner_name']), _last_name_key(row['loser_name']))
        dup = any(abs((d - sd).days) <= DEDUP_WINDOW_DAYS for sd in sack_index.get(key, ()))
        keep_mask.append(not dup)
    return supplement[pd.Series(keep_mask, index=supplement.index)]


# ---------------------------------------------------------------------------
# Step (d): write supplements
# ---------------------------------------------------------------------------

def build_supplements(db_path: str, data_dir: str, cutoff: pd.Timestamp | None) -> None:
    # Existing supplement files are derived data, regenerated wholesale below;
    # any year not rewritten in this run is pruned afterwards (supersession).
    stale = [f for f in os.listdir(data_dir)
             if re.fullmatch(r'atp_matches_\d{4}_supplement\.csv', f)] if os.path.isdir(data_dir) else []

    if not os.path.exists(db_path):
        print(f'[supplement] agent DB not found at {db_path}; skipping supplement build '
              f'(existing supplement files left untouched)')
        return

    try:
        agent = load_agent_matches(db_path)
    except Exception as exc:
        print(f'[supplement] could not read agent DB ({exc}); skipping supplement build')
        return

    print(f'[supplement] agent DB completed ATP matches: {len(agent)}')
    if cutoff is not None:
        agent = agent[agent['effective_date'] > cutoff].copy()
        print(f'[supplement] after cutoff {cutoff.date()}: {len(agent)} candidate rows')
    else:
        print('[supplement] no Sackmann cutoff available; keeping all agent rows')

    written_years: set[int] = set()
    if not agent.empty:
        normalized = normalize_to_sackmann(agent)
        for year, group in normalized.groupby(normalized['tourney_date'] // 10000):
            year = int(year)
            sack_path = os.path.join(data_dir, f'atp_matches_{year}.csv')
            if os.path.exists(sack_path):
                sackmann = pd.read_csv(sack_path, low_memory=False)
            else:
                sackmann = pd.DataFrame(columns=SACKMANN_COLUMNS)

            kept = dedup_supplement(group, sackmann)
            dropped = len(group) - len(kept)
            print(f'[supplement {year}] kept {len(kept)} rows, dropped {dropped} as Sackmann duplicates')
            if kept.empty:
                continue
            out_path = os.path.join(data_dir, f'atp_matches_{year}_supplement.csv')
            kept.to_csv(out_path, index=False)
            print(f'[supplement {year}] wrote {out_path}')
            written_years.add(year)
    else:
        print('[supplement] no agent rows beyond Sackmann coverage')

    for f in stale:
        year = int(re.fullmatch(r'atp_matches_(\d{4})_supplement\.csv', f).group(1))
        if year not in written_years:
            os.remove(os.path.join(data_dir, f))
            print(f'[supplement] removed stale {f}')


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description='Refresh Sackmann ATP data and build agent supplement files.')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--db-path', default=DEFAULT_DB_PATH,
                        help='Path to the Hermes/agent sqlite database')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip the Sackmann GitHub download (offline mode)')
    args = parser.parse_args()

    today = date.today()
    years = years_to_refresh(today)

    if args.skip_download:
        print('[sackmann] download skipped (--skip-download)')
    else:
        for year in years:
            refresh_sackmann_file(year, args.data_dir)

    cutoff = compute_cutoff(years, args.data_dir)
    if cutoff is not None:
        print(f'[cutoff] Sackmann coverage through {cutoff.date()} (max tourney_date)')

    build_supplements(args.db_path, args.data_dir, cutoff)
    print('[done] refresh complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())
