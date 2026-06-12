"""Today's Matchups: every Kalshi ATP matchup scored by the model, plus open picks."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from ui.common import TIER_ORDER, cents, money, pct, safe_fetch_picks, tier_badge

_TIER_LABELS = {
    'value_bet': '💰 Value bet',
    'confident_pick': '✅ Confident',
    'lean': '📈 Lean',
    'no_bet': '— No bet',
}


def render() -> None:
    st.title("📋 Today's Matchups")
    db_path = st.session_state.get('db_path', 'data/live/tennis_live.db')

    date_col, surface_col, button_col = st.columns([1.2, 1, 1.4])
    with date_col:
        board_date = st.date_input('Match date', value=datetime.now(timezone.utc).date())
    with surface_col:
        surface_choice = st.selectbox('Surface', ['Auto', 'Hard', 'Clay', 'Grass'], index=0)
    with button_col:
        st.write('')
        refresh = st.button('Refresh from Kalshi', type='primary', width='stretch')
    st.caption(
        'Pulls every ATP matchup listed on Kalshi for the selected date, scores each with the '
        'production model, and applies the strategy gates. Picks are persisted for paper trading.'
    )

    if refresh:
        with st.spinner('Fetching Kalshi matchups and scoring with the model...'):
            try:
                from tennis_ml.daily import build_matchup_board

                board = build_matchup_board(
                    db_path=db_path,
                    board_date=board_date.isoformat(),
                    surface=None if surface_choice == 'Auto' else surface_choice,
                )
                st.session_state['matchup_board'] = board
                st.session_state['matchup_board_date'] = board_date.isoformat()
            except Exception as exc:
                st.error(f'Matchup board failed: {exc}')

    board = st.session_state.get('matchup_board')
    if board is not None and not board.empty:
        _render_board(board, st.session_state.get('matchup_board_date', ''))
    elif board is not None:
        st.info('Kalshi lists no ATP matchups for the selected date. Try the next day '
                '(markets usually open the evening before).')
    else:
        st.info('Click **Refresh from Kalshi** to load and score the matchups for the selected date.')

    st.divider()
    st.subheader('Open picks')
    picks = safe_fetch_picks(db_path, status='open')
    if picks.empty:
        st.info('No open picks yet — refresh the board above to generate them from live Kalshi markets.')
        return

    picks = _rank_picks(picks)
    actionable = picks[picks['confidence_tier'] != 'no_bet']
    summary_cols = st.columns(4)
    summary_cols[0].metric('Open picks', len(picks))
    summary_cols[1].metric('Value bets', int((picks['confidence_tier'] == 'value_bet').sum()))
    summary_cols[2].metric('Confident picks', int((picks['confidence_tier'] == 'confident_pick').sum()))
    total_stake = pd.to_numeric(actionable['stake_suggested'], errors='coerce').fillna(0).sum()
    summary_cols[3].metric('Suggested exposure', money(total_stake))
    st.divider()

    for _, pick in picks.iterrows():
        _pick_card(pick)


def _render_board(board: pd.DataFrame, board_date: str) -> None:
    board = board.copy()
    quotes = pd.to_numeric(board.get('player1_yes_ask'), errors='coerce')
    if quotes.isna().all():
        st.warning(
            'Kalshi returned no quotes (bid/ask). Quote data requires API credentials: set '
            '`KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PEM` in the app environment. '
            'Model probabilities are still shown below.'
        )

    if 'confidence_tier' in board.columns:
        board['tier_rank'] = board['confidence_tier'].map(TIER_ORDER).fillna(9)
        board = board.sort_values(['tier_rank', 'net_ev'], ascending=[True, False]).drop(columns=['tier_rank'])

    summary_cols = st.columns(4)
    summary_cols[0].metric('Matchups', len(board))
    if 'confidence_tier' in board.columns:
        summary_cols[1].metric('Value bets', int((board['confidence_tier'] == 'value_bet').sum()))
        summary_cols[2].metric('Confident picks', int((board['confidence_tier'] == 'confident_pick').sum()))
        stakes = pd.to_numeric(board.get('stake_suggested'), errors='coerce').fillna(0)
        summary_cols[3].metric('Suggested exposure', money(stakes.sum()))

    display = pd.DataFrame({
        'Matchup': board['player1'] + ' vs ' + board['player2'],
        'P1 model': pd.to_numeric(board.get('player1_probability'), errors='coerce'),
        'P1 ask': pd.to_numeric(board.get('player1_yes_ask'), errors='coerce'),
        'P2 model': pd.to_numeric(board.get('player2_probability'), errors='coerce'),
        'P2 ask': pd.to_numeric(board.get('player2_yes_ask'), errors='coerce'),
        'Pick': board.get('pick_player'),
        'Tier': board.get('confidence_tier', pd.Series(dtype=object)).map(_TIER_LABELS),
        'Net EV': pd.to_numeric(board.get('net_ev'), errors='coerce'),
        'Stake': pd.to_numeric(board.get('stake_suggested'), errors='coerce'),
        'Surface': board.get('surface'),
    })
    if 'error' in board.columns and board['error'].notna().any():
        display['Error'] = board['error']

    st.dataframe(
        display,
        hide_index=True,
        width='stretch',
        column_config={
            'P1 model': st.column_config.NumberColumn(format='percent', help='Model fair probability, player 1'),
            'P2 model': st.column_config.NumberColumn(format='percent', help='Model fair probability, player 2'),
            'P1 ask': st.column_config.NumberColumn(format='dollar', help='Kalshi YES ask, player 1'),
            'P2 ask': st.column_config.NumberColumn(format='dollar', help='Kalshi YES ask, player 2'),
            'Net EV': st.column_config.NumberColumn(format='percent', help='Net EV after Kalshi fees on the picked side'),
            'Stake': st.column_config.NumberColumn(format='dollar'),
        },
    )
    st.caption(f'Board for {board_date} · prices are Kalshi YES asks in dollars (≈ probability).')


def _rank_picks(picks: pd.DataFrame) -> pd.DataFrame:
    picks = picks.copy()
    picks['tier_rank'] = picks['confidence_tier'].map(TIER_ORDER).fillna(9)
    net_ev = pd.to_numeric(picks['net_ev'], errors='coerce')
    fair_prob = pd.to_numeric(picks['fair_prob'], errors='coerce')
    # value bets sort by net EV desc; everything else by fair prob desc
    picks['sort_key'] = (-net_ev).where(picks['confidence_tier'] == 'value_bet', -fair_prob)
    return picks.sort_values(['tier_rank', 'sort_key']).drop(columns=['tier_rank', 'sort_key'])


def _pick_card(pick: pd.Series) -> None:
    with st.container(border=True):
        head, body = st.columns([1.6, 2.4])
        with head:
            st.markdown(f"**{pick['player']}** to beat {pick['opponent']}  \n{tier_badge(pick['confidence_tier'])}")
            details = []
            if pick.get('tournament') and pd.notna(pick.get('tournament')):
                details.append(str(pick['tournament']))
            if pick.get('match_date') and pd.notna(pick.get('match_date')):
                details.append(str(pick['match_date'])[:10])
            if pick.get('kalshi_ticker') and pd.notna(pick.get('kalshi_ticker')):
                details.append(str(pick['kalshi_ticker']))
            if details:
                st.caption(' · '.join(details))
        with body:
            cols = st.columns(5)
            cols[0].metric('Side', str(pick.get('side', '—')).upper())
            cols[1].metric('Fair prob', pct(pick.get('fair_prob')))
            cols[2].metric('Kalshi price', cents(pick.get('kalshi_price')))
            net_ev = pick.get('net_ev')
            cols[3].metric('Net EV', pct(net_ev) if net_ev is not None and pd.notna(net_ev) else '—')
            cols[4].metric('Stake', money(pick.get('stake_suggested')))
        with st.expander('Strategy reasons'):
            for line in _reason_lines(pick.get('reasons')):
                st.markdown(line)
            st.caption(f"Strategy {pick.get('strategy_version', '—')} · generated {str(pick.get('generated_at', ''))[:19]}")


def _reason_lines(raw) -> list[str]:
    try:
        reasons = json.loads(raw) if isinstance(raw, str) else (raw or [])
    except (TypeError, ValueError):
        reasons = [str(raw)] if raw else []
    lines = []
    for reason in reasons:
        text = str(reason)
        if 'PASS' in text:
            lines.append(f'- ✅ {text}')
        elif 'FAIL' in text:
            lines.append(f'- ❌ {text}')
        else:
            lines.append(f'- ℹ️ {text}')
    return lines or ['- (no reasons recorded)']
