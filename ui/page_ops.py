"""Ops / Data Status: freshness, model runs, snapshots, and maintenance actions."""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from ui.common import safe_fetch_picks, safe_read_table

DATA_DIR = Path('tennis_datav2')


def render() -> None:
    st.title('⚙️ Ops / Data Status')
    db_path = st.session_state.get('db_path', 'data/live/tennis_live.db')

    _freshness_cards(db_path)
    st.divider()

    runs_col, snapshots_col = st.columns(2)
    with runs_col:
        st.subheader('Model runs')
        runs = safe_read_table('model_runs', db_path)
        if runs.empty:
            st.info('No model runs recorded yet. Download or retrain a model below.')
        else:
            st.dataframe(
                runs.sort_values('trained_at', ascending=False)
                    [['model_run_id', 'trained_at', 'as_of_date', 'training_rows', 'artifact_dir']],
                width='stretch', hide_index=True,
            )
    with snapshots_col:
        st.subheader('Picks by status')
        picks = safe_fetch_picks(db_path)
        if picks.empty:
            st.info('No picks stored yet.')
        else:
            counts = picks['status'].value_counts().rename_axis('status').reset_index(name='count')
            st.dataframe(counts, width='stretch', hide_index=True)

    st.divider()
    st.subheader('Actions')
    refresh_tab, retrain_tab, download_tab = st.tabs(
        ['Refresh match data', 'Retrain model', 'Download release model'])

    with refresh_tab:
        st.caption('Downloads the latest Sackmann CSV for the current year and rebuilds the '
                   'Hermes supplement file (scripts/refresh_match_data.py).')
        if st.button('Run data refresh', type='primary'):
            _stream_subprocess([sys.executable, 'scripts/refresh_match_data.py'])

    with retrain_tab:
        st.warning('Training takes several minutes and is CPU-heavy. Avoid while a '
                   'walk-forward run is executing.')
        col_a, col_b = st.columns(2)
        as_of = col_a.date_input('Train as of date', value=date.today())
        start_year = col_b.number_input('Training start year', min_value=2000, max_value=2026, value=2019, step=1)
        if st.button('Retrain daily model'):
            with st.spinner('Training the daily stacked model... this can take a few minutes.'):
                try:
                    from tennis_ml.daily import train_daily_model

                    run = train_daily_model(as_of_date=str(as_of), db_path=db_path, start_year=int(start_year))
                    st.success(f"Model ready: {run['model_run_id']} ({run.get('training_rows', '?')} training rows)")
                except Exception as exc:
                    st.error(f'Training failed: {exc}')

    with download_tab:
        st.caption('Fetches the latest model artifact published by the GitHub Actions daily '
                   'model workflow (preferred over local retraining).')
        if st.button('Download latest release model'):
            with st.spinner('Downloading latest model artifact...'):
                try:
                    from tennis_ml.daily.model_artifacts import ensure_latest_model_artifact

                    path = ensure_latest_model_artifact()
                    st.success(f'Model available: {path}')
                except Exception as exc:
                    st.error(f'Download failed: {exc}')


def _freshness_cards(db_path: str) -> None:
    cols = st.columns(4)

    year = date.today().year
    latest_date, n_rows = _sackmann_freshness(year)
    cols[0].metric(
        f'Sackmann data ({year})',
        latest_date or 'missing',
        help=f'Max tourney_date in {DATA_DIR}/atp_matches_{year}.csv '
             f'({n_rows:,} rows)' if latest_date else f'No {DATA_DIR}/atp_matches_{year}.csv found.',
    )

    supplement_rows = _supplement_rows(year)
    cols[1].metric(
        'Supplement rows',
        f'{supplement_rows:,}' if supplement_rows is not None else 'missing',
        help=f'Hermes-scraped matches after the Sackmann cutoff '
             f'({DATA_DIR}/atp_matches_{year}_supplement.csv).',
    )

    snapshots = safe_read_table('odds_snapshots', db_path)
    last_snapshot = None
    if not snapshots.empty and 'snapshot_time' in snapshots.columns:
        last_snapshot = str(snapshots['snapshot_time'].max())[:19]
    cols[2].metric('Last Kalshi snapshot', last_snapshot or 'never',
                   help=f'{len(snapshots):,} snapshot rows in the live DB.')

    runs = safe_read_table('model_runs', db_path)
    last_run = None
    if not runs.empty and 'trained_at' in runs.columns:
        last_run = str(runs.sort_values('trained_at').iloc[-1]['trained_at'])[:19]
    cols[3].metric('Last model train', last_run or 'never',
                   help=f'{len(runs):,} model runs recorded.')


def _sackmann_freshness(year: int) -> tuple[str | None, int]:
    path = DATA_DIR / f'atp_matches_{year}.csv'
    if not path.exists():
        # fall back to the most recent year file present
        candidates = sorted(DATA_DIR.glob('atp_matches_20??.csv'))
        if not candidates:
            return None, 0
        path = candidates[-1]
    try:
        frame = pd.read_csv(path, usecols=['tourney_date'])
        latest = pd.to_datetime(frame['tourney_date'].astype(str), format='%Y%m%d', errors='coerce').max()
        return (latest.strftime('%Y-%m-%d') if pd.notna(latest) else None), len(frame)
    except Exception:
        return None, 0


def _supplement_rows(year: int) -> int | None:
    path = DATA_DIR / f'atp_matches_{year}_supplement.csv'
    if not path.exists():
        return None
    try:
        return len(pd.read_csv(path))
    except Exception:
        return None


def _stream_subprocess(command: list[str]) -> None:
    output_box = st.empty()
    lines: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip())
            output_box.code('\n'.join(lines[-40:]) or '(no output yet)')
        returncode = process.wait()
        if returncode == 0:
            st.success('Data refresh finished.')
        else:
            st.error(f'Data refresh exited with code {returncode}.')
    except FileNotFoundError as exc:
        st.error(f'Could not start refresh: {exc}')
    except Exception as exc:
        st.error(f'Refresh failed: {exc}')
