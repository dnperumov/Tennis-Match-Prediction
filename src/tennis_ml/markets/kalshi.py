"""
Read-only Kalshi market data connector.

This connector uses public market/event endpoints for discovery and top-of-book
quotes. Authenticated full order books and order placement should be added only
after paper-trading gates pass.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import pandas as pd
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


KALSHI_BASE_URL = 'https://external-api.kalshi.com/trade-api/v2'


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

    def get_json(self, path: str, params: dict[str, Any] | None = None, authenticated: bool = False) -> dict:
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
        url = f'{self.base_url.rstrip("/")}/{path.lstrip("/")}'
        if query:
            url = f'{url}?{query}'
        headers = {'Accept': 'application/json'}
        if authenticated:
            headers.update(self.auth_headers('GET', path))
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
