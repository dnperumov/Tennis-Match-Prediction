#!/usr/bin/env python3
"""Backfill 2026 Roland Garros ATP rows from stats.tennismylife.org.

This is intentionally stdlib-only so it can run in the current repo even when the
project venv is missing pandas. It preserves the existing 2026 export schema.

Source limitation: the page exposes winner-side serve stats only (aces and double
faults as counts; 1st-in/1st-won/2nd-won percentages; BP saved). We do not fake
missing loser-side serve/return stats or ace/DF percentages that require serve
point denominators.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_lib
import os
import re
import shutil
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "tennis_matches.sqlite"
EXPORT_PATH = ROOT / "data" / "exports" / "atp_matches_enriched_current.csv"
REPORT_DIR = ROOT / "data" / "reports"
SOURCE_URL = "https://stats.tennismylife.org/tournaments/roland-garros/2026"
TOURNEY_NAME = "Roland Garros"
SURFACE = "Clay"
MATCH_DATE = "2026-05-25"  # tournament start date, matching JeffSackmann yearly CSV convention

EXPORT_COLUMNS = [
    "match_key", "match_date", "tourney_name", "surface", "round", "winner_name", "loser_name", "score",
    "winner_serve_ace_pct", "winner_serve_df_pct", "winner_serve_first_in_pct", "winner_serve_first_won_pct",
    "winner_serve_second_won_pct", "winner_serve_bp_saved_num", "winner_serve_bp_saved_den",
    "winner_total_points_won_pct", "winner_return_points_won_pct", "winner_return_bp_converted_num",
    "winner_return_bp_converted_den", "minutes", "loser_serve_ace_pct", "loser_serve_df_pct",
    "loser_serve_first_in_pct", "loser_serve_first_won_pct", "loser_serve_second_won_pct",
    "loser_serve_bp_saved_num", "loser_serve_bp_saved_den", "loser_total_points_won_pct",
    "loser_return_points_won_pct", "loser_return_bp_converted_num", "loser_return_bp_converted_den",
    "source_url",
]


def fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; tennis-data-backfill/1.0)"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read().decode("utf-8", "ignore")


def strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return html_lib.unescape(s).strip()


def player_name_from_cell(td_html: str) -> str:
    # Prefer the player profile anchor, but exclude ranking and surface anchors.
    for href, text in re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', td_html, flags=re.S):
        if href.startswith("/players/") and "/ranking" not in href and not href.endswith("/clay") and not href.endswith("/hard") and not href.endswith("/grass"):
            return strip_tags(text)
    # Fallback: remove the trailing surface label if present.
    text = strip_tags(td_html)
    for suffix in ("Clay", "Hard", "Grass"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.strip()


def parse_pct(text: str):
    text = text.strip()
    if not text or text in {"-", "NA"}:
        return None
    return float(text.rstrip("%"))


def parse_int(text: str):
    text = text.strip()
    if not text or text in {"-", "NA"}:
        return None
    return int(text)


def parse_bp(text: str):
    text = text.strip()
    if not text or "/" not in text:
        return (None, None)
    a, b = text.split("/", 1)
    return int(a), int(b)


def make_key(row: dict) -> str:
    raw = "|".join(["2026", TOURNEY_NAME, row["round"], row["winner_name"], row["loser_name"], row["score"]])
    return "tml_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def parse_rows(page: str) -> list[dict]:
    table_match = re.search(r"<table\b.*?</table>", page, flags=re.S | re.I)
    if not table_match:
        raise RuntimeError("No table found on TennisMyLife page")
    table = table_match.group(0)
    trs = re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, flags=re.S | re.I)
    rows = []
    for tr in trs[1:]:  # skip header
        tds = re.findall(r"<td\b[^>]*>(.*?)</td>", tr, flags=re.S | re.I)
        if len(tds) < 14:
            continue
        round_ = strip_tags(tds[0])
        winner_name = player_name_from_cell(tds[2])
        loser_name = player_name_from_cell(tds[4])
        score = strip_tags(tds[5])
        if not (round_ and winner_name and loser_name and score):
            continue
        bp_saved_num, bp_saved_den = parse_bp(strip_tags(tds[13]))
        loser_bp_conv_num = (bp_saved_den - bp_saved_num) if bp_saved_num is not None and bp_saved_den is not None else None
        row = {
            "match_date": MATCH_DATE,
            "tourney_name": TOURNEY_NAME,
            "surface": SURFACE,
            "round": round_,
            "winner_name": winner_name,
            "loser_name": loser_name,
            "score": score,
            # Aces/DF are counts in this source, while existing export expects pct. Leave null rather than corrupting semantics.
            "winner_serve_ace_pct": None,
            "winner_serve_df_pct": None,
            "winner_serve_first_in_pct": parse_pct(strip_tags(tds[10])),
            "winner_serve_first_won_pct": parse_pct(strip_tags(tds[11])),
            "winner_serve_second_won_pct": parse_pct(strip_tags(tds[12])),
            "winner_serve_bp_saved_num": bp_saved_num,
            "winner_serve_bp_saved_den": bp_saved_den,
            "winner_total_points_won_pct": None,
            "winner_return_points_won_pct": None,
            "winner_return_bp_converted_num": None,
            "winner_return_bp_converted_den": None,
            "minutes": parse_int(strip_tags(tds[7])),
            "loser_serve_ace_pct": None,
            "loser_serve_df_pct": None,
            "loser_serve_first_in_pct": None,
            "loser_serve_first_won_pct": None,
            "loser_serve_second_won_pct": None,
            "loser_serve_bp_saved_num": None,
            "loser_serve_bp_saved_den": None,
            "loser_total_points_won_pct": None,
            "loser_return_points_won_pct": None,
            "loser_return_bp_converted_num": loser_bp_conv_num,
            "loser_return_bp_converted_den": bp_saved_den,
            "source_url": SOURCE_URL,
        }
        row["match_key"] = make_key(row)
        rows.append(row)
    return rows


def ensure_db(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_key TEXT PRIMARY KEY,
            tourney_name TEXT,
            surface TEXT,
            round TEXT,
            match_date TEXT,
            winner_name TEXT,
            loser_name TEXT,
            score TEXT,
            source_url TEXT,
            updated_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS player_match_stats (
            match_key TEXT,
            side TEXT,
            player_name TEXT,
            serve_ace_pct REAL,
            serve_df_pct REAL,
            serve_first_in_pct REAL,
            serve_first_won_pct REAL,
            serve_second_won_pct REAL,
            serve_bp_saved_num INTEGER,
            serve_bp_saved_den INTEGER,
            return_tpw_pct REAL,
            return_rpw_pct REAL,
            return_bp_converted_num INTEGER,
            return_bp_converted_den INTEGER,
            minutes INTEGER,
            source_url TEXT,
            updated_at TEXT,
            PRIMARY KEY (match_key, side)
        )
    """)


def upsert_rows(rows: list[dict], db_path: Path) -> tuple[int, int, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    ensure_db(con)
    inserted = updated = unchanged = 0
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with con:
        for r in rows:
            existing = con.execute("SELECT tourney_name, surface, round, match_date, winner_name, loser_name, score, source_url FROM matches WHERE match_key=?", (r["match_key"],)).fetchone()
            vals = (r["match_key"], r["tourney_name"], r["surface"], r["round"], r["match_date"], r["winner_name"], r["loser_name"], r["score"], r["source_url"], now)
            if existing is None:
                inserted += 1
            elif existing != vals[1:-1]:
                updated += 1
            else:
                unchanged += 1
            con.execute("""
                INSERT INTO matches(match_key,tourney_name,surface,round,match_date,winner_name,loser_name,score,source_url,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(match_key) DO UPDATE SET
                    tourney_name=excluded.tourney_name, surface=excluded.surface, round=excluded.round,
                    match_date=excluded.match_date, winner_name=excluded.winner_name, loser_name=excluded.loser_name,
                    score=excluded.score, source_url=excluded.source_url, updated_at=excluded.updated_at
            """, vals)
            con.execute("""
                INSERT INTO player_match_stats(match_key,side,player_name,serve_ace_pct,serve_df_pct,serve_first_in_pct,serve_first_won_pct,serve_second_won_pct,serve_bp_saved_num,serve_bp_saved_den,return_tpw_pct,return_rpw_pct,return_bp_converted_num,return_bp_converted_den,minutes,source_url,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(match_key,side) DO UPDATE SET
                    player_name=excluded.player_name, serve_ace_pct=excluded.serve_ace_pct, serve_df_pct=excluded.serve_df_pct,
                    serve_first_in_pct=excluded.serve_first_in_pct, serve_first_won_pct=excluded.serve_first_won_pct,
                    serve_second_won_pct=excluded.serve_second_won_pct, serve_bp_saved_num=excluded.serve_bp_saved_num,
                    serve_bp_saved_den=excluded.serve_bp_saved_den, return_tpw_pct=excluded.return_tpw_pct,
                    return_rpw_pct=excluded.return_rpw_pct, return_bp_converted_num=excluded.return_bp_converted_num,
                    return_bp_converted_den=excluded.return_bp_converted_den, minutes=excluded.minutes,
                    source_url=excluded.source_url, updated_at=excluded.updated_at
            """, (r["match_key"], "winner", r["winner_name"], r["winner_serve_ace_pct"], r["winner_serve_df_pct"], r["winner_serve_first_in_pct"], r["winner_serve_first_won_pct"], r["winner_serve_second_won_pct"], r["winner_serve_bp_saved_num"], r["winner_serve_bp_saved_den"], r["winner_total_points_won_pct"], r["winner_return_points_won_pct"], r["winner_return_bp_converted_num"], r["winner_return_bp_converted_den"], r["minutes"], r["source_url"], now))
            con.execute("""
                INSERT INTO player_match_stats(match_key,side,player_name,serve_ace_pct,serve_df_pct,serve_first_in_pct,serve_first_won_pct,serve_second_won_pct,serve_bp_saved_num,serve_bp_saved_den,return_tpw_pct,return_rpw_pct,return_bp_converted_num,return_bp_converted_den,minutes,source_url,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(match_key,side) DO UPDATE SET
                    player_name=excluded.player_name, serve_ace_pct=excluded.serve_ace_pct, serve_df_pct=excluded.serve_df_pct,
                    serve_first_in_pct=excluded.serve_first_in_pct, serve_first_won_pct=excluded.serve_first_won_pct,
                    serve_second_won_pct=excluded.serve_second_won_pct, serve_bp_saved_num=excluded.serve_bp_saved_num,
                    serve_bp_saved_den=excluded.serve_bp_saved_den, return_tpw_pct=excluded.return_tpw_pct,
                    return_rpw_pct=excluded.return_rpw_pct, return_bp_converted_num=excluded.return_bp_converted_num,
                    return_bp_converted_den=excluded.return_bp_converted_den, minutes=excluded.minutes,
                    source_url=excluded.source_url, updated_at=excluded.updated_at
            """, (r["match_key"], "loser", r["loser_name"], r["loser_serve_ace_pct"], r["loser_serve_df_pct"], r["loser_serve_first_in_pct"], r["loser_serve_first_won_pct"], r["loser_serve_second_won_pct"], r["loser_serve_bp_saved_num"], r["loser_serve_bp_saved_den"], r["loser_total_points_won_pct"], r["loser_return_points_won_pct"], r["loser_return_bp_converted_num"], r["loser_return_bp_converted_den"], r["minutes"], r["source_url"], now))
    con.close()
    return inserted, updated, unchanged


def export_csv(db_path: Path, export_path: Path) -> int:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT
          m.match_key, m.match_date, m.tourney_name, m.surface, m.round, m.winner_name, m.loser_name, m.score,
          ws.serve_ace_pct AS winner_serve_ace_pct,
          ws.serve_df_pct AS winner_serve_df_pct,
          ws.serve_first_in_pct AS winner_serve_first_in_pct,
          ws.serve_first_won_pct AS winner_serve_first_won_pct,
          ws.serve_second_won_pct AS winner_serve_second_won_pct,
          ws.serve_bp_saved_num AS winner_serve_bp_saved_num,
          ws.serve_bp_saved_den AS winner_serve_bp_saved_den,
          ws.return_tpw_pct AS winner_total_points_won_pct,
          ws.return_rpw_pct AS winner_return_points_won_pct,
          ws.return_bp_converted_num AS winner_return_bp_converted_num,
          ws.return_bp_converted_den AS winner_return_bp_converted_den,
          COALESCE(ws.minutes, ls.minutes) AS minutes,
          ls.serve_ace_pct AS loser_serve_ace_pct,
          ls.serve_df_pct AS loser_serve_df_pct,
          ls.serve_first_in_pct AS loser_serve_first_in_pct,
          ls.serve_first_won_pct AS loser_serve_first_won_pct,
          ls.serve_second_won_pct AS loser_serve_second_won_pct,
          ls.serve_bp_saved_num AS loser_serve_bp_saved_num,
          ls.serve_bp_saved_den AS loser_serve_bp_saved_den,
          ls.return_tpw_pct AS loser_total_points_won_pct,
          ls.return_rpw_pct AS loser_return_points_won_pct,
          ls.return_bp_converted_num AS loser_return_bp_converted_num,
          ls.return_bp_converted_den AS loser_return_bp_converted_den,
          m.source_url
        FROM matches m
        LEFT JOIN player_match_stats ws ON ws.match_key=m.match_key AND ws.side='winner'
        LEFT JOIN player_match_stats ls ON ls.match_key=m.match_key AND ls.side='loser'
        ORDER BY m.match_date, m.tourney_name, CASE m.round
          WHEN 'F' THEN 7 WHEN 'SF' THEN 6 WHEN 'QF' THEN 5 WHEN 'R16' THEN 4 WHEN 'R32' THEN 3 WHEN 'R64' THEN 2 WHEN 'R128' THEN 1 ELSE 0 END,
          m.winner_name, m.loser_name
    """).fetchall()
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if export_path.exists():
        backup = export_path.with_suffix(export_path.suffix + f".bak_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        shutil.copy2(export_path, backup)
    with open(export_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({col: row[col] for col in EXPORT_COLUMNS})
    con.close()
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    page = fetch_html(SOURCE_URL)
    parsed = parse_rows(page)
    if not parsed:
        raise RuntimeError("Parsed zero rows; refusing to update DB/export")
    rounds = {}
    for r in parsed:
        rounds[r["round"]] = rounds.get(r["round"], 0) + 1

    if args.dry_run:
        print(f"parsed_rows={len(parsed)} rounds={rounds}")
        print("sample=", {k: parsed[0][k] for k in ['match_key','round','winner_name','loser_name','score','winner_serve_first_in_pct','minutes']})
        return 0

    inserted, updated, unchanged = upsert_rows(parsed, DB_PATH)
    exported = export_csv(DB_PATH, EXPORT_PATH)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"backfill_2026_roland_garros_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 2026 Roland Garros backfill\n\n")
        f.write(f"- Source: {SOURCE_URL}\n")
        f.write(f"- Parsed rows: {len(parsed)}\n")
        f.write(f"- Rounds: {rounds}\n")
        f.write(f"- Inserted: {inserted}\n")
        f.write(f"- Updated: {updated}\n")
        f.write(f"- Unchanged: {unchanged}\n")
        f.write(f"- Export rows after update: {exported}\n")
        f.write("\n## Known stat limitations\n")
        f.write("- Source exposes winner-side first-in/first-won/second-won percentages, winner BP saved, and minutes.\n")
        f.write("- Source exposes aces and double faults as counts, but export columns expect ace/DF percentages, so those cells are left blank.\n")
        f.write("- Loser serve percentages and total/return points won are not exposed; they are left blank rather than fabricated.\n")
    print(f"parsed_rows={len(parsed)} inserted={inserted} updated={updated} unchanged={unchanged} exported_rows={exported} report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
