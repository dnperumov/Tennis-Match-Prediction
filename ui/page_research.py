"""Research / Evidence: walk-forward results, calibration, market-ceiling story."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ui.common import existing_results_dirs, file_mtime, load_predictions, load_results_csv, pct


def render() -> None:
    st.title('🔬 Research / Evidence')

    dirs = existing_results_dirs()
    if not dirs:
        st.info(
            'No walk-forward results found. Run `walk_forward.py` to produce a results '
            'directory (e.g. walk_forward_results_stacked/) with predictions.csv, '
            'fold_metrics.csv, and the summary CSVs.'
        )
        return

    results_dir = st.selectbox('Walk-forward results directory', dirs, index=0)
    predictions_path = Path(results_dir) / 'predictions.csv'
    try:
        df = load_predictions(results_dir, file_mtime(predictions_path))
    except Exception as exc:
        st.error(f'Could not load {predictions_path}: {exc}')
        return
    if df.empty:
        st.info(f'{predictions_path} is empty.')
        return

    market_rows = df[df['market_correct'].notna()]
    _headline_cards(df, market_rows)
    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader('Accuracy by confidence threshold')
        st.dataframe(
            _tier_table(df), width='stretch', hide_index=True,
            column_config={
                'threshold': st.column_config.NumberColumn('Confidence ≥', format='%.2f'),
                'n': st.column_config.NumberColumn('Matches'),
                'coverage': st.column_config.NumberColumn('Coverage', format='percent'),
                'accuracy': st.column_config.NumberColumn('Accuracy', format='percent'),
            },
        )
        st.caption('Confidence = fair probability of the predicted winner. '
                   'Higher thresholds trade coverage for accuracy.')
    with right:
        st.subheader('Model vs market by year')
        if market_rows.empty:
            st.caption('No market probabilities available in this results set.')
        else:
            st.altair_chart(_yearly_chart(market_rows), width='stretch')
            st.caption('Computed on matches with book odds, so both pick the favorite on the same population.')

    cal_col, fold_col = st.columns(2)
    with cal_col:
        st.subheader('Calibration')
        st.altair_chart(_calibration_chart(df), width='stretch')
        st.caption('Binned model probability vs actual player-1 win rate. '
                   'Points on the diagonal mean the fair probabilities are honest.')
    with fold_col:
        st.subheader('Fold metrics')
        folds = load_results_csv(results_dir, 'fold_metrics.csv', file_mtime(Path(results_dir) / 'fold_metrics.csv'))
        if folds.empty:
            st.caption('fold_metrics.csv not found in this results directory.')
        else:
            config = {col: st.column_config.NumberColumn(col, format='%.3f')
                      for col in folds.select_dtypes('number').columns if col != 'test_year'}
            st.dataframe(folds, width='stretch', hide_index=True, column_config=config)
            if 'blend_weight' in folds.columns:
                st.caption('blend_weight is how much logit-space weight the stacked model gives the '
                           'no-vig market probability vs the XGBoost submodel each year.')

    st.divider()
    _explainer()


def _headline_cards(df: pd.DataFrame, market_rows: pd.DataFrame) -> None:
    cols = st.columns(4)
    cols[0].metric(
        'Model accuracy', pct(market_rows['model_correct'].mean()) if not market_rows.empty else pct(df['model_correct'].mean()),
        help=f'On the {len(market_rows):,} matches with book odds (all {len(df):,} matches: '
             f'{pct(df["model_correct"].mean())}).',
    )
    cols[1].metric(
        'Market favorite accuracy', pct(market_rows['market_correct'].mean()) if not market_rows.empty else '—',
        help='Picking the no-vig closing favorite on the same matches. This is the ceiling.',
    )
    confident = df[df['confidence'] >= 0.60]
    cols[2].metric(
        'Accuracy at ≥60% confidence',
        pct(confident['model_correct'].mean()) if not confident.empty else '—',
        help='The honest headline: accuracy when the model is confident.',
    )
    cols[3].metric(
        'Coverage at ≥60% confidence',
        pct(len(confident) / len(df)) if len(df) else '—',
        help='Share of all matches where the model reaches 60% confidence.',
    )


def _tier_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for threshold in np.arange(0.50, 0.8001, 0.05):
        subset = df[df['confidence'] >= threshold - 1e-9]
        rows.append({
            'threshold': round(float(threshold), 2),
            'n': len(subset),
            'coverage': len(subset) / total if total else np.nan,
            'accuracy': subset['model_correct'].mean() if len(subset) else np.nan,
        })
    return pd.DataFrame(rows)


def _yearly_chart(market_rows: pd.DataFrame) -> alt.Chart:
    yearly = market_rows.groupby('test_year').agg(
        Model=('model_correct', 'mean'),
        Market=('market_correct', 'mean'),
        n=('model_correct', 'size'),
    ).reset_index()
    long = yearly.melt(id_vars=['test_year', 'n'], value_vars=['Model', 'Market'],
                       var_name='Picker', value_name='accuracy')
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X('test_year:O', title='Year'),
            y=alt.Y('accuracy:Q', title='Favorite accuracy', scale=alt.Scale(zero=False),
                    axis=alt.Axis(format='.0%')),
            color=alt.Color('Picker:N', title=None),
            tooltip=[
                alt.Tooltip('test_year:O', title='Year'),
                alt.Tooltip('Picker:N'),
                alt.Tooltip('accuracy:Q', format='.1%'),
                alt.Tooltip('n:Q', title='Matches'),
            ],
        )
        .properties(height=320)
    )


def _calibration_chart(df: pd.DataFrame) -> alt.Chart:
    frame = df[['player1_probability', 'p1_won']].dropna().copy()
    frame['bin'] = (frame['player1_probability'] * 10).clip(0, 9.999).astype(int)
    binned = frame.groupby('bin').agg(
        predicted=('player1_probability', 'mean'),
        actual=('p1_won', 'mean'),
        n=('p1_won', 'size'),
    ).reset_index()
    points = (
        alt.Chart(binned)
        .mark_circle()
        .encode(
            x=alt.X('predicted:Q', title='Mean predicted P(player 1 wins)',
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%')),
            y=alt.Y('actual:Q', title='Actual player-1 win rate',
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%')),
            size=alt.Size('n:Q', title='Matches', legend=None),
            tooltip=[
                alt.Tooltip('predicted:Q', format='.1%'),
                alt.Tooltip('actual:Q', format='.1%'),
                alt.Tooltip('n:Q', title='Matches'),
            ],
        )
    )
    line = (
        alt.Chart(binned)
        .mark_line(color='#999')
        .encode(x='predicted:Q', y='actual:Q')
    )
    diagonal = (
        alt.Chart(pd.DataFrame({'x': [0.0, 1.0], 'y': [0.0, 1.0]}))
        .mark_line(strokeDash=[4, 4], color='#bbb')
        .encode(x='x:Q', y='y:Q')
    )
    return (diagonal + line + points).properties(height=320)


def _explainer() -> None:
    st.subheader('What this evidence means')
    st.markdown(
        """
**The market ceiling.** Picking the closing-market favorite wins about **68%** of ATP
matches (consensus no-vig ≈ 68.0%, Pinnacle ≈ 68.1%). Our stacked model — XGBoost blended
with the no-vig market probability in logit space — lands at **67.7%** on the same
matches: it *ties* the market. Full-population accuracy above 70% is not attainable for
anyone, including the books; tennis matches are simply that uncertain.

**The honest headline is confident-pick accuracy.** When the model's fair probability for
its pick reaches **≥0.60**, accuracy is **74.0%** at roughly two-thirds coverage — and that
held between 72% and 76% in every walk-forward year 2019–2026. At ≥0.55 it is ~70.8%
(~84% coverage); at ≥0.70 it is ~80.8%.

**Where the money is: execution price vs fair value.** Since the model can't out-pick the
closing market, the edge comes from *prices*, not picks. Backtest: bet whenever the best
available book price beats consensus no-vig fair value by **EV ≥ 6%** within decimal odds
**1.2–3.5** → **+10.3% ROI**, positive in **all 8** walk-forward years. That is
line-shopping against a sharp benchmark, not forecasting.

**On Kalshi** the same logic applies: compare the model's fair probability with the
executable YES/NO price and pay the taker fee of about `0.07 × P × (1 − P)` per contract.
A pick becomes a **value bet** only when net EV after fees clears 6% inside the odds band.
Currently **paper trading only** — no real money until the live track record validates the
backtest.
        """
    )
