"""
Read-only Kalshi market data connector.

Event/market discovery works unauthenticated, but Kalshi gates quote fields
(bid/ask/last/volume) behind authentication: requests are signed automatically
whenever KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PEM are available. Order
placement is intentionally not implemented (paper trading only).
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


KALSHI_BASE_URL = 'https://api.elections.kalshi.com/trade-api/v2'

ATP_MATCH_SERIES = 'KXATPMATCH'
WTA_MATCH_SERIES = 'KXWTAMATCH'

_TICKER_DATE_PATTERN = re.compile(r'-(\d{2})([A-Z]{3})(\d{2})')
_MONTHS = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
}


@dataclass
class KalshiClient:
    base_url: str = KALSHI_BASE_URL
    timeout: int = 20
    api_key_id: str | None = None
    private_key_pem: str | None = None

    @classmethod
    def from_env(cls, base_url: str = KALSHI_BASE_URL, timeout: int = 20) -> 'KalshiClient':
        return cls(
            base_url=base_url,
            timeout=timeout,
            api_key_id=os.getenv('KALSHI_API_KEY_ID'),
            private_key_pem=os.getenv('KALSHI_PRIVATE_KEY_PEM'),
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key_id and self.private_key_pem)

    def get_json(self, path: str, params: dict[str, Any] | None = None, authenticated: bool | None = None) -> dict:
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
        url = f'{self.base_url.rstrip("/")}/{path.lstrip("/")}'
        if query:
            url = f'{url}?{query}'
        headers = {'Accept': 'application/json'}
        # Sign whenever credentials exist: Kalshi returns null quote fields on
        # unauthenticated market-data requests.
        if authenticated or (authenticated is None and self.has_credentials):
            signed_path = urllib.parse.urlsplit(url).path
            headers.update(self.auth_headers('GET', signed_path))
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode('utf-8'))

    def get_sports_filters(self) -> dict:
        return self.get_json('/search/filters_by_sport')

    def get_events(
        self,
        status: str = 'open',
        with_nested_markets: bool = True,
        min_close_ts: int | None = None,
        limit: int = 200,
        max_pages: int = 10,
    ) -> list[dict]:
        events = []
        cursor = None
        for _ in range(max_pages):
            payload = self.get_json('/events', {
                'status': status,
                'with_nested_markets': str(with_nested_markets).lower(),
                'min_close_ts': min_close_ts,
                'limit': limit,
                'cursor': cursor,
            })
            events.extend(payload.get('events', []))
            cursor = payload.get('cursor')
            if not cursor:
                break
        return events

    def get_open_tennis_events(self, max_pages: int = 10) -> list[dict]:
        now = int(time.time())
        events = self.get_events(status='open', with_nested_markets=True, min_close_ts=now, max_pages=max_pages)
        return [event for event in events if event_mentions_tennis(event)]

    def get_match_events(
        self,
        series_ticker: str = ATP_MATCH_SERIES,
        status: str = 'open',
        max_pages: int = 10,
    ) -> list[dict]:
        """Fetch head-to-head match events (one event per matchup, one market per player)."""
        events = []
        cursor = None
        for _ in range(max_pages):
            payload = self.get_json('/events', {
                'series_ticker': series_ticker,
                'status': status,
                'with_nested_markets': 'true',
                'limit': 200,
                'cursor': cursor,
            })
            events.extend(payload.get('events', []))
            cursor = payload.get('cursor')
            if not cursor:
                break
        return events

    def get_market_orderbook(self, ticker: str) -> dict:
        return self.get_json(f'/markets/{ticker}/orderbook', authenticated=True)

    def auth_headers(self, method: str, path: str) -> dict[str, str]:
        if not self.api_key_id or not self.private_key_pem:
            raise ValueError('Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PEM for authenticated requests.')
        timestamp_ms = str(int(time.time() * 1000))
        normalized_path = '/' + path.lstrip('/')
        message = f'{timestamp_ms}{method.upper()}{normalized_path}'.encode('utf-8')
        private_key = serialization.load_pem_private_key(
            self.private_key_pem.encode('utf-8'),
            password=None,
        )
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            'KALSHI-ACCESS-KEY': self.api_key_id,
            'KALSHI-ACCESS-TIMESTAMP': timestamp_ms,
            'KALSHI-ACCESS-SIGNATURE': base64.b64encode(signature).decode('utf-8'),
        }


def event_mentions_tennis(event: dict) -> bool:
    text_parts = [
        event.get('title'),
        event.get('sub_title'),
        event.get('category'),
        json.dumps(event.get('product_metadata', {}), sort_keys=True),
    ]
    for market in event.get('markets', []) or []:
        text_parts.extend([
            market.get('title'),
            market.get('subtitle'),
            market.get('yes_sub_title'),
            market.get('no_sub_title'),
            market.get('rules_primary'),
            market.get('rules_secondary'),
        ])
    return 'tennis' in ' '.join(str(part or '') for part in text_parts).lower()


def normalize_kalshi_events(events: list[dict]) -> pd.DataFrame:
    rows = []
    for event in events:
        for market in event.get('markets', []) or []:
            yes_ask = to_float(market.get('yes_ask_dollars'))
            yes_bid = to_float(market.get('yes_bid_dollars'))
            no_ask = to_float(market.get('no_ask_dollars'))
            no_bid = to_float(market.get('no_bid_dollars'))
            rows.append({
                'snapshot_time': pd.Timestamp.utcnow().isoformat(),
                'event_ticker': event.get('event_ticker'),
                'series_ticker': event.get('series_ticker'),
                'event_title': event.get('title'),
                'market_ticker': market.get('ticker'),
                'market_title': market.get('title'),
                'market_subtitle': market.get('subtitle'),
                'yes_sub_title': market.get('yes_sub_title'),
                'no_sub_title': market.get('no_sub_title'),
                'occurrence_datetime': market.get('occurrence_datetime'),
                'close_time': market.get('close_time'),
                'yes_bid': yes_bid,
                'yes_ask': yes_ask,
                'no_bid': no_bid,
                'no_ask': no_ask,
                'last_price': to_float(market.get('last_price_dollars')),
                'liquidity_dollars': to_float(market.get('liquidity_dollars')),
                'volume': to_float(market.get('volume_fp')),
                'volume_24h': to_float(market.get('volume_24h_fp')),
                'yes_decimal_odds': decimal_odds_from_price(yes_ask),
                'no_decimal_odds': decimal_odds_from_price(no_ask),
                'source': 'kalshi',
            })
    return pd.DataFrame(rows)


def market_quote(market: dict) -> dict[str, float | None]:
    """Extract quote fields in 0-1 probability units (prefer dollar fields, fall back to cents)."""
    def price(dollar_key: str, cent_key: str) -> float | None:
        dollars = to_float(market.get(dollar_key))
        if dollars is not None:
            return dollars
        cents = to_float(market.get(cent_key))
        return cents / 100 if cents is not None else None

    return {
        'yes_bid': price('yes_bid_dollars', 'yes_bid'),
        'yes_ask': price('yes_ask_dollars', 'yes_ask'),
        'no_bid': price('no_bid_dollars', 'no_bid'),
        'no_ask': price('no_ask_dollars', 'no_ask'),
        'last_price': price('last_price_dollars', 'last_price'),
        'volume': to_float(market.get('volume_fp')) or to_float(market.get('volume')),
        'liquidity': to_float(market.get('liquidity_dollars')) or to_float(market.get('liquidity')),
    }


def match_date_from_ticker(event_ticker: str | None) -> str | None:
    """KXATPMATCH-26JUN12MEDCIL -> '2026-06-12'."""
    if not event_ticker:
        return None
    found = _TICKER_DATE_PATTERN.search(event_ticker)
    if not found:
        return None
    year_part, month_part, day_part = found.groups()
    month = _MONTHS.get(month_part)
    if not month:
        return None
    try:
        return date(2000 + int(year_part), month, int(day_part)).isoformat()
    except ValueError:
        return None


def parse_match_event(event: dict) -> dict | None:
    """Turn a head-to-head match event into a matchup dict, or None if malformed.

    Expects exactly two nested markets, one per player, with the player's full
    name in yes_sub_title.
    """
    markets = event.get('markets') or []
    if len(markets) != 2:
        return None
    players = [str(market.get('yes_sub_title') or '').strip() for market in markets]
    if not all(players) or players[0] == players[1]:
        return None

    event_ticker = event.get('event_ticker')
    match_date = match_date_from_ticker(event_ticker)
    if match_date is None:
        expiration = markets[0].get('expected_expiration_time') or markets[0].get('close_time')
        if expiration:
            match_date = str(expiration)[:10]

    matchup = {
        'event_ticker': event_ticker,
        'series_ticker': event.get('series_ticker'),
        'title': event.get('title'),
        'match_date': match_date,
        'close_time': markets[0].get('close_time'),
        'player1': players[0],
        'player2': players[1],
    }
    for side, market in zip(('player1', 'player2'), markets):
        quote = market_quote(market)
        matchup[f'{side}_market_ticker'] = market.get('ticker')
        matchup[f'{side}_yes_bid'] = quote['yes_bid']
        matchup[f'{side}_yes_ask'] = quote['yes_ask']
        matchup[f'{side}_last_price'] = quote['last_price']
        matchup[f'{side}_volume'] = quote['volume']
        matchup[f'{side}_liquidity'] = quote['liquidity']
    return matchup


def to_float(value) -> float | None:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decimal_odds_from_price(price: float | None) -> float | None:
    if price is None or price <= 0:
        return None
    return 1 / price
