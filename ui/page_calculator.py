"""Probability Calculator: model fair value for a matchup + Kalshi EV verdict."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.common import cents, get_bankroll, money, pct, set_bankroll


@st.cache_resource(show_spinner='Loading model...')
def _get_service(db_path: str, model_dir: str | None):
    from tennis_ml.daily import TennisPredictionService

    return TennisPredictionService(db_path=db_path, model_dir=model_dir)


def render() -> None:
    st.title('🎯 Probability Calculator')
    db_path = st.session_state.get('db_path', 'data/live/tennis_live.db')
    model_dir = (st.session_state.get('model_dir') or '').strip() or None

    with st.form('predict_form'):
        col_a, col_b = st.columns(2)
        with col_a:
            player1 = st.text_input('Player 1', 'Jannik Sinner')
            surface = st.selectbox('Surface', ['Hard', 'Clay', 'Grass', 'Carpet', 'Unknown'])
            player1_odds = st.number_input('Player 1 book odds (optional)', min_value=0.0, value=0.0, step=0.01,
                                           help='Decimal odds, e.g. 1.85. Leave 0 to skip.')
        with col_b:
            player2 = st.text_input('Player 2', 'Carlos Alcaraz')
            tournament_level = st.selectbox('Tournament level', ['A', 'M', 'G', 'C', 'F', 'D'], index=0,
                                            help='A=ATP, M=Masters, G=Grand Slam, C=Challenger, F=Finals, D=Davis Cup')
            player2_odds = st.number_input('Player 2 book odds (optional)', min_value=0.0, value=0.0, step=0.01)

        col_c, col_d, col_e = st.columns(3)
        with col_c:
            round_value = st.selectbox('Round', ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F', 'RR'], index=2)
        with col_d:
            match_date = st.date_input('Match date')
        with col_e:
            kalshi_price_cents = st.number_input(
                f'Kalshi YES price for Player 1 (¢, optional)',
                min_value=0, max_value=99, value=0, step=1,
                help='The executable YES ask for Player 1 winning, in cents. Leave 0 to skip.',
            )

        bankroll = st.number_input('Bankroll ($)', min_value=1.0, value=get_bankroll(), step=50.0,
                                   help='Used to size suggested stakes (fractional Kelly, capped at 2%).')
        submitted = st.form_submit_button('Score match', type='primary')

    if bankroll != get_bankroll():
        set_bankroll(bankroll)

    if not submitted:
        st.caption('Enter a matchup and click **Score match**. Predictions are stored and become '
                   'eligible for pick generation on the Today\'s Picks page.')
        return

    kalshi_price = kalshi_price_cents / 100 if kalshi_price_cents else None
    try:
        from tennis_ml.daily import MatchPredictionRequest

        service = _get_service(db_path, model_dir)
        request = MatchPredictionRequest(
            player1=player1,
            player2=player2,
            surface=surface,
            tournament_level=tournament_level,
            round=round_value,
            match_date=str(match_date),
            player1_odds=player1_odds or None,
            player2_odds=player2_odds or None,
            kalshi_yes_price=kalshi_price,
        )
        with st.spinner('Building features and scoring...'):
            result = service.predict(request)
    except FileNotFoundError:
        st.info('No model is available yet. Use **Ops / Data Status** to download the latest '
                'release model or retrain one, then come back.')
        return
    except ValueError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:
        st.error(f'Prediction failed: {exc}')
        return

    p1 = result['player1_probability']
    p2 = result['player2_probability']

    st.subheader('Fair value')
    cols = st.columns(4)
    cols[0].metric(f'{player1}', pct(p1), help='Model fair probability of winning')
    cols[1].metric(f'{player2}', pct(p2))
    cols[2].metric('Fair Kalshi YES price (P1)', cents(p1))
    cols[3].metric('Fair decimal odds', f"{result['fair_odds_player1']:.2f} / {result['fair_odds_player2']:.2f}")
    captions = [f"Model: {result.get('model_kind', '?')}", f"Confidence: {result.get('confidence', '?')}"]
    if result.get('market_probability_player1') is not None:
        captions.append(f"No-vig market: {pct(result['market_probability_player1'])} / "
                        f"{pct(result['market_probability_player2'])} (blended into the model)")
    st.caption(' · '.join(captions))

    if kalshi_price is not None:
        _kalshi_verdict(p1, kalshi_price, player1, player2, bankroll)
    else:
        st.info('Enter a Kalshi YES price to get an EV verdict and suggested stake.')


def _kalshi_verdict(fair_prob: float, yes_price: float, player1: str, player2: str, bankroll: float) -> None:
    from tennis_ml.strategy import StrategyConfig, evaluate_pick, size_stake

    st.subheader('Kalshi verdict')
    config = StrategyConfig(bankroll=bankroll)
    try:
        evaluation = evaluate_pick(fair_prob, yes_price, fee_rate=config.fee_rate)
    except ValueError as exc:
        st.warning(str(exc))
        return

    side = evaluation['recommended_side']
    side_player = player1 if side == 'yes' else player2
    pick_prob = fair_prob if side == 'yes' else 1 - fair_prob
    net_ev = evaluation['net_ev']
    odds = evaluation['implied_decimal_odds']

    ev_pass = net_ev >= config.ev_threshold
    band_pass = config.min_odds <= odds <= config.max_odds
    if ev_pass and band_pass:
        tier, headline = 'value_bet', f'✅ VALUE BET — buy **{side.upper()}** ({side_player})'
    elif pick_prob >= config.confident_threshold:
        tier, headline = 'confident_pick', f'ℹ️ Confident pick on {side_player}, but the price offers no edge — no bet.'
    elif pick_prob >= config.lean_threshold:
        tier, headline = 'lean', f'ℹ️ Lean toward {side_player}, no betting edge at this price.'
    else:
        tier, headline = 'no_bet', '🚫 No bet — neither side clears the strategy gates.'
    st.markdown(headline)

    cols = st.columns(5)
    cols[0].metric('EV (YES side)', pct(evaluation['ev_yes']), help='Net EV per $1 staked, after fees')
    cols[1].metric('EV (NO side)', pct(evaluation['ev_no']))
    cols[2].metric('Fee per contract', cents(evaluation['fee_per_contract']),
                   help=f"Kalshi taker fee ≈ {config.fee_rate} × P × (1 − P)")
    cols[3].metric('Edge (prob − price)', pct(evaluation['edge']))
    stake = size_stake(pick_prob, evaluation['price'], bankroll,
                       fraction=config.kelly_fraction, cap=config.kelly_cap) if tier == 'value_bet' else 0.0
    cols[4].metric('Suggested stake', money(stake),
                   help=f'25% Kelly capped at 2% of your {money(bankroll)} bankroll. $0 unless gates pass.')

    gate_lines = [
        f"- {'✅' if ev_pass else '❌'} Net EV {pct(net_ev, 2)} {'≥' if ev_pass else '<'} threshold {pct(config.ev_threshold)}",
        f"- {'✅' if band_pass else '❌'} Implied odds {odds:.2f} {'inside' if band_pass else 'outside'} band "
        f"[{config.min_odds}, {config.max_odds}]",
        f"- Tier: **{tier}** (pick-side fair prob {pct(pick_prob)})",
    ]
    with st.expander('Strategy gates', expanded=not (ev_pass and band_pass)):
        st.markdown('\n'.join(gate_lines))
