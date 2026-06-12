"""Today's Picks: open strategy picks ranked by tier and edge."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ui.common import TIER_ORDER, cents, money, pct, safe_fetch_picks, tier_badge


def render() -> None:
    st.title("📋 Today's Picks")
    db_path = st.session_state.get('db_path', 'data/live/tennis_live.db')

    button_col, caption_col = st.columns([1, 3])
    with button_col:
        refresh = st.button('Refresh picks from Kalshi', type='primary', width='stretch')
    with caption_col:
        st.caption(
            'Matches the latest Kalshi tennis snapshot against recent stored model '
            'predictions, applies the strategy gates, and settles past picks.'
        )

    if refresh:
        with st.spinner('Running strategy step against the latest Kalshi snapshot...'):
            try:
                from tennis_ml.daily import run_strategy_step

                summary = run_strategy_step(db_path=db_path)
                tiers = summary.get('tiers', {})
                settlement = summary.get('settlement', {})
                st.success(
                    f"Markets considered: {summary.get('markets_considered', 0)} · "
                    f"predictions considered: {summary.get('predictions_considered', 0)} · "
                    f"matched: {summary.get('candidates_matched', 0)} · "
                    f"picks written: {summary.get('picks_written', 0)} "
                    f"(value {tiers.get('value_bet', 0)}, confident {tiers.get('confident_pick', 0)}, "
                    f"lean {tiers.get('lean', 0)}, no-bet {tiers.get('no_bet', 0)}) · "
                    f"settled: {settlement.get('settled', 0)} "
                    f"({settlement.get('won', 0)}W-{settlement.get('lost', 0)}L)"
                )
            except Exception as exc:
                st.error(f'Strategy step failed: {exc}')

    picks = safe_fetch_picks(db_path, status='open')
    if picks.empty:
        st.info(
            'No open picks yet. This usually means either no open Kalshi tennis markets '
            'matched a stored model prediction, or the daily pipeline has not run. '
            'Score upcoming matches on the Probability Calculator page (predictions are '
            'stored automatically), then click **Refresh picks from Kalshi** above.'
        )
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
