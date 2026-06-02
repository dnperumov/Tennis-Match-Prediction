#!/usr/bin/env python3
"""Research-only odds snapshot/CLV store.

Stores pre-match odds snapshots from CSV/manual exports or read-only market APIs, then
computes closing-line-value fields for model validation. This script NEVER places orders.

Kalshi note: Kalshi markets are event contracts, not a normal sportsbook odds feed. Use
this for read-only market snapshots that you manually map to tennis match sides.
Do not put API keys in chat; use environment variables or local secret files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "odds" / "odds_snapshots.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    snapshot_ts TEXT NOT NULL,
    source TEXT NOT NULL,
    book TEXT,
    market_ticker TEXT,
    player1 TEXT NOT NULL,
    player2 TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('player1','player2')),
    decimal_odds REAL,
    american_odds REAL,
    yes_bid REAL,
    yes_ask REAL,
    implied_prob_raw REAL,
    no_vig_prob REAL,
    minutes_until_match REAL,
    is_closing INTEGER DEFAULT 0,
    raw_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(match_id, snapshot_ts, source, book, market_ticker, side)
);

CREATE INDEX IF NOT EXISTS idx_odds_match_ts ON odds_snapshots(match_id, snapshot_ts);
CREATE INDEX IF NOT EXISTS idx_odds_source ON odds_snapshots(source, market_ticker);

CREATE VIEW IF NOT EXISTS latest_match_clv AS
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY match_id, side ORDER BY snapshot_ts DESC) AS rn_desc,
           ROW_NUMBER() OVER (PARTITION BY match_id, side ORDER BY snapshot_ts ASC) AS rn_asc
    FROM odds_snapshots
    WHERE no_vig_prob IS NOT NULL
),
open AS (
    SELECT match_id, side, no_vig_prob AS open_prob, snapshot_ts AS open_ts
    FROM ranked WHERE rn_asc = 1
),
close AS (
    SELECT match_id, side, no_vig_prob AS close_prob, snapshot_ts AS close_ts
    FROM ranked WHERE is_closing = 1
),
last_seen AS (
    SELECT match_id, side, no_vig_prob AS last_prob, snapshot_ts AS last_ts
    FROM ranked WHERE rn_desc = 1
)
SELECT o.match_id,
       o.side,
       o.open_prob,
       COALESCE(c.close_prob, l.last_prob) AS close_prob,
       COALESCE(c.close_ts, l.last_ts) AS close_ts,
       COALESCE(c.close_prob, l.last_prob) - o.open_prob AS clv_prob_delta
FROM open o
LEFT JOIN close c USING(match_id, side)
LEFT JOIN last_seen l USING(match_id, side);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    return con


def decimal_to_prob(decimal_odds: float | None) -> float | None:
    if decimal_odds is None or decimal_odds <= 1:
        return None
    return 1.0 / decimal_odds


def american_to_decimal(american: float | None) -> float | None:
    if american is None:
        return None
    if american > 0:
        return 1.0 + american / 100.0
    if american < 0:
        return 1.0 + 100.0 / abs(american)
    return None


def kalshi_price_to_prob(bid: float | None, ask: float | None) -> float | None:
    vals = [v for v in [bid, ask] if v is not None]
    if not vals:
        return None
    # Kalshi prices may be cents (0-100) or probabilities (0-1). Normalize defensively.
    mid = sum(vals) / len(vals)
    return mid / 100.0 if mid > 1 else mid


def parse_float(value: Any) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def upsert_snapshot(con: sqlite3.Connection, row: dict[str, Any]) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO odds_snapshots (
            match_id, snapshot_ts, source, book, market_ticker, player1, player2, side,
            decimal_odds, american_odds, yes_bid, yes_ask, implied_prob_raw, no_vig_prob,
            minutes_until_match, is_closing, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["match_id"], row["snapshot_ts"], row["source"], row.get("book") or "", row.get("market_ticker") or "",
            row["player1"], row["player2"], row["side"], row.get("decimal_odds"), row.get("american_odds"),
            row.get("yes_bid"), row.get("yes_ask"), row.get("implied_prob_raw"), row.get("no_vig_prob"),
            row.get("minutes_until_match"), int(bool(row.get("is_closing", False))), json.dumps(row.get("raw", {}), sort_keys=True),
        ),
    )


def import_csv(con: sqlite3.Connection, path: Path) -> int:
    """Import snapshots from a CSV with one row per match/side/snapshot.

    Required columns: match_id,snapshot_ts,source,player1,player2,side
    Optional: book,market_ticker,decimal_odds,american_odds,yes_bid,yes_ask,no_vig_prob,minutes_until_match,is_closing
    """
    count = 0
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            decimal_odds = parse_float(raw.get("decimal_odds"))
            american_odds = parse_float(raw.get("american_odds"))
            if decimal_odds is None and american_odds is not None:
                decimal_odds = american_to_decimal(american_odds)
            yes_bid, yes_ask = parse_float(raw.get("yes_bid")), parse_float(raw.get("yes_ask"))
            implied = parse_float(raw.get("implied_prob_raw")) or decimal_to_prob(decimal_odds) or kalshi_price_to_prob(yes_bid, yes_ask)
            row = {
                "match_id": raw["match_id"],
                "snapshot_ts": raw["snapshot_ts"],
                "source": raw.get("source") or "csv",
                "book": raw.get("book"),
                "market_ticker": raw.get("market_ticker"),
                "player1": raw["player1"],
                "player2": raw["player2"],
                "side": raw["side"],
                "decimal_odds": decimal_odds,
                "american_odds": american_odds,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "implied_prob_raw": implied,
                "no_vig_prob": parse_float(raw.get("no_vig_prob")) or implied,
                "minutes_until_match": parse_float(raw.get("minutes_until_match")),
                "is_closing": str(raw.get("is_closing", "0")).lower() in {"1", "true", "yes"},
                "raw": raw,
            }
            upsert_snapshot(con, row)
            count += 1
    con.commit()
    return count


def fetch_kalshi_market(ticker: str, base_url: str) -> dict[str, Any]:
    # Read-only endpoint. If your Kalshi account/API setup requires signatures, use an approved local wrapper;
    # this function intentionally does not handle private-key signing or order endpoints.
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", f"trade-api/v2/markets/{urllib.parse.quote(ticker)}")
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "tennis-research-odds-snapshot/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:  # nosec B310 - user-configured HTTPS API, read-only
        return json.loads(resp.read().decode("utf-8"))


def snapshot_kalshi_ticker(con: sqlite3.Connection, ticker: str, match_id: str, player1: str, player2: str, side: str, dry_run: bool) -> dict[str, Any]:
    base_url = os.environ.get("KALSHI_BASE_URL", "https://api.elections.kalshi.com")
    payload = fetch_kalshi_market(ticker, base_url)
    market = payload.get("market", payload)
    yes_bid = parse_float(market.get("yes_bid"))
    yes_ask = parse_float(market.get("yes_ask"))
    implied = kalshi_price_to_prob(yes_bid, yes_ask)
    row = {
        "match_id": match_id,
        "snapshot_ts": datetime.now(timezone.utc).isoformat(),
        "source": "kalshi",
        "book": "kalshi",
        "market_ticker": ticker,
        "player1": player1,
        "player2": player2,
        "side": side,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "implied_prob_raw": implied,
        "no_vig_prob": implied,
        "raw": {"market": market},
    }
    if not dry_run:
        upsert_snapshot(con, row)
        con.commit()
    return row


def main() -> int:
    p = argparse.ArgumentParser(description="Research-only odds snapshot and CLV store. No order placement.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    imp = sub.add_parser("import-csv")
    imp.add_argument("csv_path", type=Path)
    kal = sub.add_parser("kalshi-snapshot")
    kal.add_argument("--ticker", required=True)
    kal.add_argument("--match-id", required=True)
    kal.add_argument("--player1", required=True)
    kal.add_argument("--player2", required=True)
    kal.add_argument("--side", choices=["player1", "player2"], required=True)
    kal.add_argument("--dry-run", action="store_true")
    exp = sub.add_parser("export-clv")
    exp.add_argument("--out", type=Path, default=ROOT / "data" / "odds" / "latest_clv.csv")
    args = p.parse_args()

    con = connect(args.db)
    if args.cmd == "init":
        print(json.dumps({"db": str(args.db), "status": "initialized"}, indent=2))
    elif args.cmd == "import-csv":
        n = import_csv(con, args.csv_path)
        print(json.dumps({"db": str(args.db), "rows_imported": n}, indent=2))
    elif args.cmd == "kalshi-snapshot":
        row = snapshot_kalshi_ticker(con, args.ticker, args.match_id, args.player1, args.player2, args.side, args.dry_run)
        safe = {k: v for k, v in row.items() if k != "raw"}
        print(json.dumps({"dry_run": args.dry_run, "snapshot": safe}, indent=2))
    elif args.cmd == "export-clv":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cur = con.execute("SELECT * FROM latest_match_clv ORDER BY match_id, side")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        print(json.dumps({"out": str(args.out), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
