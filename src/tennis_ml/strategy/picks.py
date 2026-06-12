"""Strategy layer: execution-price-vs-fair-value picks for Kalshi paper trading.

The profitable strategy is NOT "bet model edge vs closing market" — it is
EXECUTION PRICE vs FAIR VALUE: bet when an available price clears the model's
fair probability by enough EV after fees. Backtested: EV >= 6% at decimal odds
1.2-3.5 vs consensus fair value, positive ROI in all 8 walk-forward years.

Kalshi taker fee is approximately 0.07 * price * (1 - price) per contract, so
that fee rate is baked into the EV math here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tennis_ml.backtesting.odds_loader import normalize_player_name
from tennis_ml.live_stats.database import (
    DEFAULT_DB_PATH,
    fetch_picks,
    insert_paper_trade,
    update_pick_status,
    upsert_pick,
)


STRATEGY_VERSION = 'price-vs-fair-v1'


@dataclass
class StrategyConfig:
    ev_threshold: float = 0.06          # net EV per $1 staked, after fees
    min_odds: float = 1.2               # implied decimal odds band for value bets
    max_odds: float = 3.5
    confident_threshold: float = 0.60   # fair prob of the picked side
    lean_threshold: float = 0.55
    fee_rate: float = 0.07              # Kalshi taker fee factor: fee = rate*P*(1-P)
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.02             # max stake as fraction of bankroll
    bankroll: float = 1000.0
    strategy_version: str = STRATEGY_VERSION


def kalshi_fee(price: float, fee_rate: float = 0.07) -> float:
    """Approximate Kalshi taker fee per contract at the given price."""
    return fee_rate * price * (1 - price)


def evaluate_pick(fair_prob: float, kalshi_yes_price: float, fee_rate: float = 0.07) -> dict[str, Any]:
    """Net EV per $1 staked, after Kalshi fees, on both YES and NO sides.

    ``fair_prob`` is the model's fair probability of the YES outcome.
    ``kalshi_yes_price`` is the executable YES price in dollars (0-1).
    A YES contract costs ``price`` and pays $1 if YES; a NO contract costs
    ``1 - price`` and pays $1 if NO. Fee per contract is ``fee_rate*P*(1-P)``.
    """
    if not 0 < kalshi_yes_price < 1:
        raise ValueError('kalshi_yes_price must be between 0 and 1 (exclusive).')
    if not 0 <= fair_prob <= 1:
        raise ValueError('fair_prob must be between 0 and 1.')

    yes_price = float(kalshi_yes_price)
    no_price = 1 - yes_price
    fee = kalshi_fee(yes_price, fee_rate)  # symmetric in price for YES and NO

    # EV per contract: win pays (1 - cost), loss costs cost; fee paid on trade.
    ev_yes_contract = fair_prob * (1 - yes_price) - (1 - fair_prob) * yes_price - fee
    ev_no_contract = (1 - fair_prob) * (1 - no_price) - fair_prob * no_price - fee

    # Per $1 staked (stake = contract cost), comparable to decimal-odds EV.
    ev_yes = ev_yes_contract / yes_price
    ev_no = ev_no_contract / no_price

    recommended_side = 'yes' if ev_yes >= ev_no else 'no'
    if recommended_side == 'yes':
        net_ev, price, prob = ev_yes, yes_price, fair_prob
    else:
        net_ev, price, prob = ev_no, no_price, 1 - fair_prob

    return {
        'fair_prob': float(fair_prob),
        'yes_price': yes_price,
        'no_price': no_price,
        'fee_rate': float(fee_rate),
        'fee_per_contract': float(fee),
        'ev_yes': float(ev_yes),
        'ev_no': float(ev_no),
        'ev_yes_per_contract': float(ev_yes_contract),
        'ev_no_per_contract': float(ev_no_contract),
        'recommended_side': recommended_side,
        'net_ev': float(net_ev),
        'edge': float(prob - price),
        'price': float(price),
        'implied_decimal_odds': float(1 / price),
    }


def size_stake(
    fair_prob: float,
    price: float,
    bankroll: float,
    fraction: float = 0.25,
    cap: float = 0.02,
) -> float:
    """Fractional Kelly stake in dollars, capped at ``cap`` of bankroll.

    ``price`` is the contract cost (0-1); decimal odds are 1/price.
    """
    if not 0 < price < 1 or bankroll <= 0:
        return 0.0
    b = (1 - price) / price  # net odds per $1 staked
    full_kelly = (fair_prob * b - (1 - fair_prob)) / b
    kelly_fraction = max(0.0, full_kelly * fraction)
    return float(min(kelly_fraction, cap) * bankroll)


def generate_picks(predictions: list[dict[str, Any]], config: StrategyConfig | None = None) -> list[dict[str, Any]]:
    """Apply strategy gates to fair-value predictions and label pick tiers.

    Each prediction dict needs:
      - ``match``: dict with at least player1/player2 (match_date, tournament,
        kalshi_ticker, prediction_id optional)
      - ``fair_prob``: model fair probability that player1 wins
      - ``kalshi_price``: executable YES price for player1, or None
      - ``confidence``: optional model confidence label (passed through)

    Tiers: 'value_bet' (EV + odds-band gates pass against a live price),
    'confident_pick' (fair prob of pick side >= 0.60), 'lean' (0.55-0.60),
    'no_bet' otherwise.
    """
    config = config or StrategyConfig()
    picks = []
    for prediction in predictions:
        match = prediction.get('match', {})
        fair_prob = float(prediction['fair_prob'])
        kalshi_price = prediction.get('kalshi_price')
        reasons: list[str] = []

        # Pick side by fair probability (player1 YES market convention).
        pick_player1 = fair_prob >= 0.5
        pick_side = 'yes' if pick_player1 else 'no'
        pick_prob = fair_prob if pick_player1 else 1 - fair_prob
        player = match.get('player1') if pick_player1 else match.get('player2')
        opponent = match.get('player2') if pick_player1 else match.get('player1')

        tier = 'no_bet'
        evaluation = None
        net_ev = None
        stake = 0.0
        price = None

        if kalshi_price is not None and 0 < float(kalshi_price) < 1:
            evaluation = evaluate_pick(fair_prob, float(kalshi_price), fee_rate=config.fee_rate)
            pick_side = evaluation['recommended_side']
            pick_player1 = pick_side == 'yes'
            pick_prob = fair_prob if pick_player1 else 1 - fair_prob
            player = match.get('player1') if pick_player1 else match.get('player2')
            opponent = match.get('player2') if pick_player1 else match.get('player1')
            net_ev = evaluation['net_ev']
            price = evaluation['price']
            odds = evaluation['implied_decimal_odds']

            ev_pass = net_ev >= config.ev_threshold
            band_pass = config.min_odds <= odds <= config.max_odds
            reasons.append(
                f"net_ev={net_ev:.4f} {'>=':s} {config.ev_threshold} -> {'PASS' if ev_pass else 'FAIL'}"
            )
            reasons.append(
                f"implied_odds={odds:.2f} in [{config.min_odds}, {config.max_odds}] -> "
                f"{'PASS' if band_pass else 'FAIL'}"
            )
            if ev_pass and band_pass:
                tier = 'value_bet'
                stake = size_stake(
                    pick_prob, price, config.bankroll,
                    fraction=config.kelly_fraction, cap=config.kelly_cap,
                )
        else:
            reasons.append('no kalshi price available; EV gate not evaluable')

        if tier != 'value_bet':
            if pick_prob >= config.confident_threshold:
                tier = 'confident_pick'
                reasons.append(f'fair_prob={pick_prob:.3f} >= {config.confident_threshold} -> confident pick')
            elif pick_prob >= config.lean_threshold:
                tier = 'lean'
                reasons.append(f'fair_prob={pick_prob:.3f} in [{config.lean_threshold}, {config.confident_threshold}) -> lean')
            else:
                tier = 'no_bet'
                reasons.append(f'fair_prob={pick_prob:.3f} < {config.lean_threshold} -> no bet')

        picks.append({
            'pick_id': _pick_id(match, player, pick_side, config.strategy_version),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'match_date': match.get('match_date'),
            'tournament': match.get('tournament'),
            'player': player,
            'opponent': opponent,
            'side': pick_side,
            'fair_prob': float(pick_prob),
            'confidence_tier': tier,
            'kalshi_ticker': match.get('kalshi_ticker'),
            'kalshi_price': float(price) if price is not None else None,
            'net_ev': float(net_ev) if net_ev is not None else None,
            'stake_suggested': float(stake),
            'strategy_version': config.strategy_version,
            'reasons': json.dumps(reasons),
            'prediction_id': match.get('prediction_id'),
            'status': 'open',
            'evaluation': evaluation,
            'model_confidence': prediction.get('confidence'),
        })
    return picks


def persist_picks(picks: list[dict[str, Any]], db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Upsert generated picks into the picks table. Returns rows written."""
    written = 0
    for pick in picks:
        upsert_pick({key: value for key, value in pick.items() if key != 'evaluation'}, db_path=db_path)
        written += 1
    return written


def settle_picks(
    db_path: str | Path,
    completed_matches_df: pd.DataFrame,
    fee_rate: float = 0.07,
) -> dict[str, int]:
    """Settle open picks against completed matches and write paper trades.

    ``completed_matches_df`` needs player1/player2/winner/match_date columns
    (the live ``matches`` table shape). Picks are matched on normalized player
    names plus match date. Kalshi P&L per contract: win -> (1 - price) - fee,
    lose -> -price. Stake-denominated P&L scales by contracts = stake / price.
    """
    open_picks = fetch_picks(status='open', db_path=db_path)
    if open_picks.empty or completed_matches_df is None or completed_matches_df.empty:
        return {'settled': 0, 'won': 0, 'lost': 0, 'paper_trades': 0}

    completed = completed_matches_df.copy()
    completed = completed[completed['winner'].notna() & (completed['winner'].astype(str).str.strip() != '')]
    if completed.empty:
        return {'settled': 0, 'won': 0, 'lost': 0, 'paper_trades': 0}
    completed['match_date'] = pd.to_datetime(completed['match_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    completed['p1_key'] = completed['player1'].map(normalize_player_name)
    completed['p2_key'] = completed['player2'].map(normalize_player_name)
    completed['winner_key'] = completed['winner'].map(normalize_player_name)

    settled = won = lost = trades = 0
    now = datetime.now(timezone.utc).isoformat()
    for _, pick in open_picks.iterrows():
        player_key = normalize_player_name(pick['player'])
        opponent_key = normalize_player_name(pick['opponent']) if pick.get('opponent') else ''
        pick_date = str(pick.get('match_date') or '')[:10]

        candidates = completed[
            (((completed['p1_key'] == player_key) & (completed['p2_key'] == opponent_key)) |
             ((completed['p2_key'] == player_key) & (completed['p1_key'] == opponent_key)))
        ]
        if pick_date:
            candidates = candidates[candidates['match_date'] == pick_date]
        if candidates.empty:
            continue

        match_row = candidates.iloc[0]
        pick_won = match_row['winner_key'] == player_key
        status = 'won' if pick_won else 'lost'
        update_pick_status(pick['pick_id'], status, settled_at=now, db_path=db_path)
        settled += 1
        won += int(pick_won)
        lost += int(not pick_won)

        price = pick.get('kalshi_price')
        if price is not None and pd.notna(price) and 0 < float(price) < 1:
            price = float(price)
            stake = float(pick.get('stake_suggested') or 0.0) or price  # default: one contract
            contracts = stake / price
            fee = kalshi_fee(price, fee_rate)
            profit_per_contract = (1 - price) - fee if pick_won else -price
            profit = contracts * profit_per_contract
            insert_paper_trade({
                'paper_trade_id': f"pick-{pick['pick_id']}",
                'prediction_id': pick.get('prediction_id'),
                'created_at': pick.get('generated_at'),
                'settled_at': now,
                'player': pick['player'],
                'opponent': pick.get('opponent'),
                'odds': 1 / price,
                'stake': stake,
                'won': int(pick_won),
                'profit': float(profit),
                'friction_adjusted_profit': float(profit),  # Kalshi fee already applied
                'clv': None,
                'status': 'settled',
            }, db_path=db_path)
            trades += 1

    return {'settled': settled, 'won': won, 'lost': lost, 'paper_trades': trades}


def match_kalshi_markets_to_predictions(
    markets: pd.DataFrame,
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Match open Kalshi tennis markets to stored model predictions.

    ``markets`` is the normalized Kalshi snapshot frame (yes_sub_title is the
    YES player, market title/event title mention both players). ``predictions``
    is the predictions table frame (player1/player2/player1_probability...).
    Returns generate_picks-ready prediction dicts with the Kalshi YES ask as
    the executable price for the matched player.
    """
    results: list[dict[str, Any]] = []
    if markets is None or markets.empty or predictions is None or predictions.empty:
        return results

    predictions = predictions.copy()
    predictions['p1_key'] = predictions['player1'].map(normalize_player_name)
    predictions['p2_key'] = predictions['player2'].map(normalize_player_name)

    for _, market in markets.iterrows():
        yes_player = str(market.get('yes_sub_title') or market.get('market_title') or '')
        yes_key = normalize_player_name(yes_player)
        if not yes_key:
            continue
        market_text = normalize_player_name(' '.join(
            str(market.get(column) or '') for column in ['event_title', 'market_title', 'yes_sub_title']
        ))

        for _, prediction in predictions.iterrows():
            p1_key, p2_key = prediction['p1_key'], prediction['p2_key']
            if not p1_key or not p2_key:
                continue
            yes_is_p1 = _name_matches(yes_key, p1_key)
            yes_is_p2 = _name_matches(yes_key, p2_key)
            if not (yes_is_p1 or yes_is_p2):
                continue
            # Require the other player to appear in the market/event text.
            other_key = p2_key if yes_is_p1 else p1_key
            if other_key.split()[-1] not in market_text:
                continue

            p1_prob = float(prediction['player1_probability'])
            fair_yes_prob = p1_prob if yes_is_p1 else 1 - p1_prob
            yes_ask = market.get('yes_ask')
            price = float(yes_ask) if yes_ask is not None and pd.notna(yes_ask) and 0 < float(yes_ask) < 1 else None
            results.append({
                'match': {
                    'player1': prediction['player1'] if yes_is_p1 else prediction['player2'],
                    'player2': prediction['player2'] if yes_is_p1 else prediction['player1'],
                    'match_date': prediction.get('match_date'),
                    'tournament': market.get('event_title'),
                    'kalshi_ticker': market.get('market_ticker'),
                    'prediction_id': prediction.get('prediction_id'),
                },
                'fair_prob': fair_yes_prob,
                'kalshi_price': price,
                'confidence': prediction.get('confidence'),
            })
            break
    return results


def _name_matches(market_key: str, prediction_key: str) -> bool:
    """Loose name match: identical keys, or shared surname + initial."""
    if market_key == prediction_key:
        return True
    market_tokens = market_key.split()
    prediction_tokens = prediction_key.split()
    if not market_tokens or not prediction_tokens:
        return False
    if market_tokens[-1] != prediction_tokens[-1]:
        return False
    return market_tokens[0][0] == prediction_tokens[0][0]


def _pick_id(match: dict[str, Any], player: str | None, side: str, strategy_version: str) -> str:
    payload = '|'.join([
        str(match.get('match_date') or ''),
        str(match.get('kalshi_ticker') or ''),
        str(player or ''),
        side,
        strategy_version,
    ])
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]
