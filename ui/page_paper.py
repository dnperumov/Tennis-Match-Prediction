"""Paper Trading: open exposure, settled history, and cumulative P&L."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ui.common import money, pct, safe_fetch_picks, safe_read_table


def render() -> None:
    st.title('📈 Paper Trading')
    db_path = st.session_state.get('db_path', 'data/live/tennis_live.db')

    picks = safe_fetch_picks(db_path)
    trades = safe_read_table('paper_trades', db_path)
    settled_trades = _settled_trades(trades)

    if picks.empty and settled_trades.empty:
        st.info(
            'No paper-trading history yet. Picks appear here once the strategy step has run '
            "(Today's Picks → Refresh picks from Kalshi) and matches settle. "
            'Settled picks with a Kalshi price are booked as paper trades automatically.'
        )
        return

    _summary_metrics(settled_trades)

    if not settled_trades.empty:
        st.subheader('Cumulative P&L')
        st.altair_chart(_pnl_chart(settled_trades), width='stretch')

        breakdown_tier, breakdown_month = st.columns(2)
        with breakdown_tier:
            st.subheader('By tier')
            by_tier = _by_tier(settled_trades, picks)
            if by_tier.empty:
                st.caption('No tier information available.')
            else:
                st.dataframe(by_tier, width='stretch', hide_index=True,
                             column_config=_breakdown_config('Tier'))
        with breakdown_month:
            st.subheader('By month')
            by_month = _by_month(settled_trades)
            st.dataframe(by_month, width='stretch', hide_index=True,
                         column_config=_breakdown_config('Month'))

    st.subheader('Open picks')
    open_picks = picks[picks['status'] == 'open'] if not picks.empty else pd.DataFrame()
    if open_picks.empty:
        st.caption('No open picks.')
    else:
        st.dataframe(
            open_picks[['match_date', 'tournament', 'player', 'opponent', 'side',
                        'fair_prob', 'kalshi_price', 'net_ev', 'stake_suggested', 'confidence_tier']],
            width='stretch', hide_index=True,
            column_config={
                'fair_prob': st.column_config.NumberColumn('Fair prob', format='percent'),
                'kalshi_price': st.column_config.NumberColumn('Kalshi price', format='$%.2f'),
                'net_ev': st.column_config.NumberColumn('Net EV', format='percent'),
                'stake_suggested': st.column_config.NumberColumn('Stake', format='dollar'),
                'confidence_tier': st.column_config.TextColumn('Tier'),
            },
        )

    st.subheader('Settled history')
    settled_picks = picks[picks['status'].isin(['won', 'lost', 'void'])] if not picks.empty else pd.DataFrame()
    if settled_picks.empty:
        st.caption('No settled picks yet.')
    else:
        st.dataframe(
            settled_picks.sort_values('settled_at', ascending=False)[
                ['settled_at', 'match_date', 'player', 'opponent', 'side', 'fair_prob',
                 'kalshi_price', 'stake_suggested', 'confidence_tier', 'status']],
            width='stretch', hide_index=True,
            column_config={
                'fair_prob': st.column_config.NumberColumn('Fair prob', format='percent'),
                'kalshi_price': st.column_config.NumberColumn('Price', format='$%.2f'),
                'stake_suggested': st.column_config.NumberColumn('Stake', format='dollar'),
            },
        )


def _settled_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or 'status' not in trades.columns:
        return pd.DataFrame()
    settled = trades[trades['status'] == 'settled'].copy()
    if settled.empty:
        return settled
    settled['profit'] = pd.to_numeric(settled['profit'], errors='coerce').fillna(0.0)
    settled['stake'] = pd.to_numeric(settled['stake'], errors='coerce').fillna(0.0)
    settled['settled_at'] = pd.to_datetime(settled['settled_at'], errors='coerce', utc=True)
    return settled.sort_values('settled_at')


def _summary_metrics(settled: pd.DataFrame) -> None:
    cols = st.columns(4)
    if settled.empty:
        cols[0].metric('Record', '0-0')
        cols[1].metric('Total staked', money(0))
        cols[2].metric('Profit', money(0))
        cols[3].metric('ROI', '—')
        return
    wins = int(pd.to_numeric(settled['won'], errors='coerce').fillna(0).sum())
    losses = len(settled) - wins
    staked = settled['stake'].sum()
    profit = settled['profit'].sum()
    cols[0].metric('Record', f'{wins}-{losses}', help='Settled paper trades won-lost')
    cols[1].metric('Total staked', money(staked))
    cols[2].metric('Profit (after fees)', money(profit), delta=money(profit))
    cols[3].metric('ROI', pct(profit / staked) if staked else '—')


def _pnl_chart(settled: pd.DataFrame) -> alt.Chart:
    frame = settled[['settled_at', 'profit']].dropna(subset=['settled_at']).copy()
    frame['cumulative_pnl'] = frame['profit'].cumsum()
    frame['trade_n'] = range(1, len(frame) + 1)
    return (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X('settled_at:T', title='Settled'),
            y=alt.Y('cumulative_pnl:Q', title='Cumulative P&L ($)'),
            tooltip=[
                alt.Tooltip('settled_at:T', title='Settled'),
                alt.Tooltip('trade_n:Q', title='Trade #'),
                alt.Tooltip('profit:Q', title='Trade P&L', format='$.2f'),
                alt.Tooltip('cumulative_pnl:Q', title='Cumulative', format='$.2f'),
            ],
        )
        .properties(height=300)
    )


def _by_tier(settled: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    if picks.empty or 'pick_id' not in picks.columns:
        return pd.DataFrame()
    merged = settled.copy()
    merged['pick_id'] = merged['paper_trade_id'].astype(str).str.removeprefix('pick-')
    merged = merged.merge(picks[['pick_id', 'confidence_tier']], on='pick_id', how='left')
    merged['confidence_tier'] = merged['confidence_tier'].fillna('unknown')
    return _aggregate(merged, 'confidence_tier', 'Tier')


def _by_month(settled: pd.DataFrame) -> pd.DataFrame:
    frame = settled.dropna(subset=['settled_at']).copy()
    frame['month'] = frame['settled_at'].dt.strftime('%Y-%m')
    return _aggregate(frame, 'month', 'Month')


def _aggregate(frame: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(key).agg(
        trades=('profit', 'size'),
        wins=('won', lambda s: int(pd.to_numeric(s, errors='coerce').fillna(0).sum())),
        staked=('stake', 'sum'),
        profit=('profit', 'sum'),
    ).reset_index().rename(columns={key: label})
    grouped['roi'] = grouped['profit'] / grouped['staked'].replace(0, pd.NA)
    return grouped


def _breakdown_config(label: str) -> dict:
    return {
        label: st.column_config.TextColumn(label),
        'trades': st.column_config.NumberColumn('Trades'),
        'wins': st.column_config.NumberColumn('Wins'),
        'staked': st.column_config.NumberColumn('Staked', format='dollar'),
        'profit': st.column_config.NumberColumn('Profit', format='dollar'),
        'roi': st.column_config.NumberColumn('ROI', format='percent'),
    }
