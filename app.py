#!/usr/bin/env python3
"""Streamlit app for tennis model probabilities and paper-trade decisions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.daily import MatchPredictionRequest, TennisPredictionService, train_daily_model
from tennis_ml.daily.model_artifacts import ensure_latest_model_artifact, latest_model_dir
from tennis_ml.live_stats.database import DEFAULT_DB_PATH, latest_model_run, read_table


st.set_page_config(page_title='Tennis Trading Lab', layout='wide')
st.title('Tennis Trading Lab')

db_path = st.sidebar.text_input('SQLite DB', str(DEFAULT_DB_PATH))
model_dir = st.sidebar.text_input('Model directory override', '')

latest_run = latest_model_run(db_path)
local_model_dir = latest_model_dir()
if latest_run:
    st.sidebar.success(f"Latest model: {latest_run['model_run_id']}")
    st.sidebar.caption(latest_run.get('artifact_dir', ''))
elif local_model_dir:
    st.sidebar.success(f"Latest model artifact: {local_model_dir.name}")
    st.sidebar.caption(str(local_model_dir))
else:
    st.sidebar.warning('No daily model run found yet.')
    st.sidebar.caption('Cloud deploys download the latest GitHub Release model artifact, or can train a fallback model here.')

with st.sidebar.expander('Model Setup', expanded=latest_run is None):
    st.caption('Preferred: download the latest model built by GitHub Actions. Fallback: train inside this deployment.')
    if st.button('Download Latest Release Model'):
        with st.spinner('Downloading latest model artifact...'):
            try:
                path = ensure_latest_model_artifact()
                st.success(f'Model downloaded: {path.name}')
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    start_year = st.number_input('Training start year', min_value=2000, max_value=2026, value=2019, step=1)
    if st.button('Initialize / Retrain Model'):
        with st.spinner('Training daily model. This can take a few minutes on Streamlit Cloud...'):
            try:
                run = train_daily_model(
                    as_of_date=pd.Timestamp.utcnow().date(),
                    db_path=db_path,
                    start_year=int(start_year),
                )
                st.success(f"Model ready: {run['model_run_id']}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

tab_predict, tab_markets, tab_history = st.tabs(['Predict Match', 'Kalshi Markets', 'Run History'])

with tab_predict:
    col_a, col_b = st.columns(2)
    with col_a:
        player1 = st.text_input('Player 1', 'Jannik Sinner')
        surface = st.selectbox('Surface', ['Hard', 'Clay', 'Grass', 'Carpet', 'Unknown'])
        player1_odds = st.number_input('Player 1 market odds', min_value=0.0, value=0.0, step=0.01)
    with col_b:
        player2 = st.text_input('Player 2', 'Carlos Alcaraz')
        tournament_level = st.selectbox('Tournament level', ['A', 'M', 'G', 'C', 'F', 'D'], index=0)
        player2_odds = st.number_input('Player 2 market odds', min_value=0.0, value=0.0, step=0.01)

    round_value = st.selectbox('Round', ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F', 'RR'])
    match_date = st.date_input('Match date')

    if st.button('Score Match', type='primary'):
        try:
            service = TennisPredictionService(
                db_path=db_path,
                model_dir=model_dir.strip() or None,
            )
            request = MatchPredictionRequest(
                player1=player1,
                player2=player2,
                surface=surface,
                tournament_level=tournament_level,
                round=round_value,
                match_date=str(match_date),
                player1_odds=player1_odds or None,
                player2_odds=player2_odds or None,
            )
            result = service.predict(request)
            metrics = st.columns(4)
            metrics[0].metric(f'{player1} win probability', f"{result['player1_probability']:.1%}")
            metrics[1].metric(f'{player2} win probability', f"{result['player2_probability']:.1%}")
            metrics[2].metric('Fair odds P1', f"{result['fair_odds_player1']:.2f}")
            metrics[3].metric('Decision', result['strategy_decision'].upper())
            st.json(result)
        except FileNotFoundError:
            st.error('No model is available yet. Open Model Setup in the sidebar and click Initialize / Retrain Model.')
        except Exception as exc:
            st.error(str(exc))

with tab_markets:
    try:
        markets = read_table('odds_snapshots', db_path=db_path)
        if markets.empty:
            st.info('No Kalshi tennis snapshots stored yet.')
        else:
            st.dataframe(markets.sort_values('snapshot_time', ascending=False), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))

with tab_history:
    cols = st.columns(2)
    for table, container in [('model_runs', cols[0]), ('predictions', cols[1])]:
        with container:
            st.subheader(table.replace('_', ' ').title())
            try:
                rows = read_table(table, db_path=db_path)
                st.dataframe(rows.sort_values(rows.columns[1], ascending=False) if not rows.empty else rows, use_container_width=True)
            except Exception as exc:
                st.error(str(exc))
