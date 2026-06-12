"""Tests for the matchup board builder (mocked Kalshi client + prediction service)."""

import json

import pandas as pd
import pytest

from tennis_ml.daily.matchups import build_matchup_board, infer_surface
from tennis_ml.live_stats.database import fetch_picks, read_table

from test_kalshi_client import MATCH_EVENT


def _event(event_ticker: str, player1: str, player2: str, p1_ask_cents: int) -> dict:
    event = json.loads(json.dumps(MATCH_EVENT))
    event['event_ticker'] = event_ticker
    event['title'] = f'{player1.split()[-1]} vs {player2.split()[-1]}'
    event['markets'][0].update({
        'ticker': f'{event_ticker}-P1',
        'yes_sub_title': player1,
        'yes_ask': p1_ask_cents,
        'yes_bid': p1_ask_cents - 2,
    })
    event['markets'][1].update({
        'ticker': f'{event_ticker}-P2',
        'yes_sub_title': player2,
        'yes_ask': 100 - p1_ask_cents + 4,
        'yes_bid': 100 - p1_ask_cents + 2,
    })
    return event


class _StubClient:
    def __init__(self, events):
        self.events = events

    def get_match_events(self, series_ticker='KXATPMATCH', status='open', max_pages=10):
        return self.events


class _StubService:
    """Returns a fixed fair probability for player1 of every matchup."""

    def __init__(self, fair_prob: float):
        self.fair_prob = fair_prob
        self.requests = []

    def predict(self, request, persist=True):
        self.requests.append(request)
        return {
            'prediction_id': f'pred-{request.player1}-{request.player2}'.replace(' ', '_'),
            'player1_probability': self.fair_prob,
            'player2_probability': 1 - self.fair_prob,
            'confidence': 'medium',
            'model_kind': 'stacked',
        }


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / 'live.db'


def test_board_filters_to_date_and_scores_each_matchup(db_path):
    events = [
        _event('KXATPMATCH-26JUN12AAABBB', 'Aaa Player', 'Bbb Player', 60),
        _event('KXATPMATCH-26JUN13CCCDDD', 'Ccc Player', 'Ddd Player', 50),  # wrong date
    ]
    service = _StubService(fair_prob=0.65)
    board = build_matchup_board(
        db_path=db_path,
        board_date='2026-06-12',
        client=_StubClient(events),
        service=service,
    )
    assert len(board) == 1
    row = board.iloc[0]
    assert row['player1'] == 'Aaa Player'
    assert row['player1_probability'] == 0.65
    assert len(service.requests) == 1
    assert service.requests[0].kalshi_yes_price == 0.60
    # snapshot persisted for the kept event only
    snapshots = read_table('odds_snapshots', db_path=db_path)
    assert set(snapshots['event_ticker']) == {'KXATPMATCH-26JUN12AAABBB'}


def test_board_generates_and_persists_picks(db_path):
    # fair 0.65 vs ask 0.55 -> strong YES value; pick should persist as value_bet
    events = [_event('KXATPMATCH-26JUN12AAABBB', 'Aaa Player', 'Bbb Player', 55)]
    board = build_matchup_board(
        db_path=db_path,
        board_date='2026-06-12',
        client=_StubClient(events),
        service=_StubService(fair_prob=0.65),
    )
    assert board.iloc[0]['confidence_tier'] == 'value_bet'
    assert board.iloc[0]['net_ev'] > 0.06
    picks = fetch_picks(db_path=db_path)
    assert len(picks) == 1
    assert picks.iloc[0]['player'] == 'Aaa Player'
    assert picks.iloc[0]['status'] == 'open'


def test_board_keeps_no_bet_rows(db_path):
    # fair prob ~ price -> negative EV after fees, low confidence -> no_bet retained
    events = [_event('KXATPMATCH-26JUN12AAABBB', 'Aaa Player', 'Bbb Player', 52)]
    board = build_matchup_board(
        db_path=db_path,
        board_date='2026-06-12',
        client=_StubClient(events),
        service=_StubService(fair_prob=0.52),
    )
    assert len(board) == 1
    assert board.iloc[0]['confidence_tier'] == 'no_bet'


def test_board_survives_prediction_failure(db_path):
    class _FailingService:
        def predict(self, request, persist=True):
            raise RuntimeError('model exploded')

    events = [_event('KXATPMATCH-26JUN12AAABBB', 'Aaa Player', 'Bbb Player', 60)]
    board = build_matchup_board(
        db_path=db_path,
        board_date='2026-06-12',
        client=_StubClient(events),
        service=_FailingService(),
    )
    assert len(board) == 1
    assert 'model exploded' in board.iloc[0]['error']


def test_board_empty_when_no_matchups(db_path):
    board = build_matchup_board(
        db_path=db_path,
        board_date='2026-06-12',
        client=_StubClient([]),
        service=_StubService(0.6),
    )
    assert board.empty


def test_infer_surface_calendar():
    from datetime import date

    assert infer_surface(date(2026, 6, 12)) == 'Grass'
    assert infer_surface(date(2026, 7, 20)) == 'Hard'
    assert infer_surface(date(2026, 4, 20)) == 'Clay'
    assert infer_surface(date(2026, 1, 20)) == 'Hard'
