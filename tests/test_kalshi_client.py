"""Tests for the Kalshi client: auth signing, match-event parsing, quotes."""

import json
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from tennis_ml.markets.kalshi import (
    KalshiClient,
    market_quote,
    match_date_from_ticker,
    parse_match_event,
)


def _make_client() -> tuple[KalshiClient, rsa.RSAPublicKey]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode('utf-8')
    client = KalshiClient(api_key_id='test-key', private_key_pem=pem)
    return client, key.public_key()


def _verify(public_key, headers: dict, method: str, path: str) -> None:
    import base64

    message = f"{headers['KALSHI-ACCESS-TIMESTAMP']}{method}{path}".encode('utf-8')
    public_key.verify(
        base64.b64decode(headers['KALSHI-ACCESS-SIGNATURE']),
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_auth_headers_sign_the_given_path():
    client, public_key = _make_client()
    headers = client.auth_headers('GET', '/trade-api/v2/events')
    _verify(public_key, headers, 'GET', '/trade-api/v2/events')
    assert headers['KALSHI-ACCESS-KEY'] == 'test-key'


def test_get_json_signs_full_api_path_including_prefix():
    """The signed message must use the full URL path (with /trade-api/v2), not the relative path."""
    client, public_key = _make_client()
    captured = {}

    class _Response:
        def read(self):
            return json.dumps({'events': []}).encode('utf-8')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        captured['headers'] = dict(request.header_items())
        captured['url'] = request.full_url
        return _Response()

    with patch('urllib.request.urlopen', side_effect=fake_urlopen):
        client.get_json('/events', {'limit': 1})

    headers = {key.upper().replace('_', '-'): value for key, value in captured['headers'].items()}
    assert 'KALSHI-ACCESS-SIGNATURE' in headers, 'credentials present -> request must be signed'
    _verify(public_key, headers, 'GET', '/trade-api/v2/events')


def test_get_json_unauthenticated_without_credentials():
    client = KalshiClient()
    captured = {}

    class _Response:
        def read(self):
            return b'{}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        captured['headers'] = dict(request.header_items())
        return _Response()

    with patch('urllib.request.urlopen', side_effect=fake_urlopen):
        client.get_json('/events')

    assert not any(key.upper().startswith('KALSHI-ACCESS') for key in captured['headers'])


def test_match_date_from_ticker():
    assert match_date_from_ticker('KXATPMATCH-26JUN12MEDCIL') == '2026-06-12'
    assert match_date_from_ticker('KXATPMATCH-27JAN01ABCDEF') == '2027-01-01'
    assert match_date_from_ticker('KXATPFINALS-26') is None
    assert match_date_from_ticker(None) is None
    assert match_date_from_ticker('KXATPMATCH-26XXX12FOO') is None


def test_market_quote_prefers_dollars_then_cents():
    dollars = market_quote({'yes_ask_dollars': 0.75, 'yes_bid_dollars': 0.73})
    assert dollars['yes_ask'] == 0.75
    cents = market_quote({'yes_ask': 75, 'yes_bid': 73, 'no_ask': 27, 'no_bid': 25, 'last_price': 74})
    assert cents['yes_ask'] == 0.75
    assert cents['no_bid'] == 0.25
    assert cents['last_price'] == 0.74
    empty = market_quote({})
    assert empty['yes_ask'] is None


MATCH_EVENT = {
    'event_ticker': 'KXATPMATCH-26JUN12MEDCIL',
    'series_ticker': 'KXATPMATCH',
    'title': 'Medvedev vs Cilic',
    'markets': [
        {
            'ticker': 'KXATPMATCH-26JUN12MEDCIL-MED',
            'yes_sub_title': 'Daniil Medvedev',
            'yes_bid': 73, 'yes_ask': 75, 'no_bid': 25, 'no_ask': 27,
            'last_price': 74, 'volume': 120,
            'close_time': '2026-06-26T15:25:00Z',
            'expected_expiration_time': '2026-06-12T18:25:00Z',
        },
        {
            'ticker': 'KXATPMATCH-26JUN12MEDCIL-CIL',
            'yes_sub_title': 'Marin Cilic',
            'yes_bid': 24, 'yes_ask': 26, 'no_bid': 74, 'no_ask': 76,
            'last_price': 25, 'volume': 120,
            'close_time': '2026-06-26T15:25:00Z',
            'expected_expiration_time': '2026-06-12T18:25:00Z',
        },
    ],
}


def test_parse_match_event():
    matchup = parse_match_event(MATCH_EVENT)
    assert matchup is not None
    assert matchup['player1'] == 'Daniil Medvedev'
    assert matchup['player2'] == 'Marin Cilic'
    assert matchup['match_date'] == '2026-06-12'
    assert matchup['player1_yes_ask'] == 0.75
    assert matchup['player2_yes_ask'] == 0.26
    assert matchup['player1_market_ticker'].endswith('-MED')


def test_parse_match_event_date_falls_back_to_expiration():
    event = json.loads(json.dumps(MATCH_EVENT))
    event['event_ticker'] = 'KXATPMATCH-NODATE'
    matchup = parse_match_event(event)
    assert matchup['match_date'] == '2026-06-12'


@pytest.mark.parametrize('mutate', [
    lambda e: e['markets'].pop(),                                  # one market only
    lambda e: e['markets'].append(dict(e['markets'][0])),          # three markets
    lambda e: e['markets'][0].update({'yes_sub_title': ''}),       # missing player name
    lambda e: e['markets'][1].update({'yes_sub_title': 'Daniil Medvedev'}),  # duplicate player
])
def test_parse_match_event_rejects_malformed(mutate):
    event = json.loads(json.dumps(MATCH_EVENT))
    mutate(event)
    assert parse_match_event(event) is None
