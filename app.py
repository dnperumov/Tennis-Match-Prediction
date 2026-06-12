#!/usr/bin/env python3
"""Tennis Trading Lab — Streamlit control center for the paper-trading workflow.

Pages:
  - Today's Picks: open strategy picks from the live DB, refreshable from Kalshi.
  - Probability Calculator: fair value for any matchup + Kalshi EV verdict.
  - Paper Trading: open exposure, settled history, cumulative P&L.
  - Research / Evidence: walk-forward accuracy, calibration, market-ceiling story.
  - Ops / Data Status: data freshness, model runs, refresh/retrain actions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_ml.live_stats.database import DEFAULT_DB_PATH  # noqa: E402

from ui import page_calculator, page_ops, page_paper, page_picks, page_research  # noqa: E402

st.set_page_config(page_title='Tennis Trading Lab', page_icon='🎾', layout='wide')


def _sidebar() -> None:
    st.sidebar.title('🎾 Tennis Trading Lab')
    st.session_state['db_path'] = st.sidebar.text_input('SQLite DB', str(DEFAULT_DB_PATH))
    st.session_state['model_dir'] = st.sidebar.text_input('Model directory override', '')

    try:
        from tennis_ml.daily.model_artifacts import latest_model_dir
        from tennis_ml.live_stats.database import latest_model_run

        latest_run = latest_model_run(st.session_state['db_path'])
        local_model_dir = latest_model_dir()
        if latest_run:
            st.sidebar.success(f"Model: {latest_run['model_run_id']}")
            st.sidebar.caption(latest_run.get('artifact_dir') or '')
        elif local_model_dir:
            st.sidebar.success(f'Model artifact: {local_model_dir.name}')
            st.sidebar.caption(str(local_model_dir))
        else:
            st.sidebar.warning('No daily model found yet.')
            st.sidebar.caption('Use Ops / Data Status to download the latest release model or retrain.')
    except Exception as exc:
        st.sidebar.warning(f'Model status unavailable: {exc}')


_sidebar()

navigation = st.navigation([
    st.Page(page_picks.render, title="Today's Picks", icon='📋', url_path='picks', default=True),
    st.Page(page_calculator.render, title='Probability Calculator', icon='🎯', url_path='calculator'),
    st.Page(page_paper.render, title='Paper Trading', icon='📈', url_path='paper-trading'),
    st.Page(page_research.render, title='Research / Evidence', icon='🔬', url_path='research'),
    st.Page(page_ops.render, title='Ops / Data Status', icon='⚙️', url_path='ops'),
])
navigation.run()
