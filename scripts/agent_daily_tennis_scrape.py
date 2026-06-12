#!/usr/bin/env python3
"""Agent reliability-layer daily ATP scrape.

Parses TennisAbstract current ATP result pages, upserts completed singles matches
into data/tennis_matches.sqlite, and exports enriched current CSV/report.
"""
from __future__ import annotations

import csv
import hashlib
import html as html_lib
import os
import re
import sqlite3
import ssl
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
DB_PATH = REPO / "data" / "tennis_matches.sqlite"
EXPORT_PATH = REPO / "data" / "exports" / "atp_matches_enriched_current.csv"
REPORT_DIR = REPO / "data" / "reports"
USER_AGENT = "Mozilla/5.0 (compatible; Tennis-Match-Prediction daily research scrape; polite)"
TA_HOME = "https://www.tennisabstract.com/"

SCHEMA = """
CREATE TABLE IF NOT EXISTS scrape_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts_utc TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS matches (
    match_key TEXT PRIMARY KEY,
    tour TEXT NOT NULL DEFAULT 'ATP',
    source TEXT NOT NULL,
    source_url TEXT NOT NULL,
    tournament TEXT NOT NULL,
    season INTEGER,
    match_date TEXT,
    round TEXT,
    winner_name TEXT NOT NULL,
    winner_country TEXT,
    loser_name TEXT NOT NULL,
    loser_country TEXT,
    score TEXT,
    best_of INTEGER,
    indoor INTEGER,
    surface TEXT,
    status TEXT NOT NULL DEFAULT 'completed',
    stats_status TEXT NOT NULL DEFAULT 'pending',
    first_seen_utc TEXT NOT NULL,
    last_seen_utc TEXT NOT NULL,
    raw_html_fragment TEXT
);
CREATE TABLE IF NOT EXISTS match_side_stats (
    match_key TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('winner','loser')),
    player_name TEXT NOT NULL,
    aces INTEGER,
    double_faults INTEGER,
    first_serve_in INTEGER,
    first_serve_total INTEGER,
    first_serve_pct REAL,
    first_serve_points_won INTEGER,
    first_serve_points_total INTEGER,
    second_serve_points_won INTEGER,
    second_serve_points_total INTEGER,
    break_points_saved INTEGER,
    break_points_faced INTEGER,
    service_points_won INTEGER,
    service_points_total INTEGER,
    return_points_won INTEGER,
    return_points_total INTEGER,
    break_points_converted INTEGER,
    break_points_opportunities INTEGER,
    total_points_won INTEGER,
    total_points_total INTEGER,
    stats_source TEXT,
    stats_url TEXT,
    updated_utc TEXT NOT NULL,
    PRIMARY KEY (match_key, side),
    FOREIGN KEY(match_key) REFERENCES matches(match_key)
);
CREATE INDEX IF NOT EXISTS idx_matches_seen ON matches(last_seen_utc);
CREATE INDEX IF NOT EXISTS idx_matches_tournament ON matches(season, tournament, round);
"""

@dataclass
class Match:
    source_url: str
    tournament: str
    season: int
    round: str
    winner_name: str
    winner_country: Optional[str]
    loser_name: str
    loser_country: Optional[str]
    score: str
    raw: str

    @property
    def key(self) -> str:
        base = "|".join([
            "ATP", str(self.season), norm(self.tournament), self.round,
            norm(self.winner_name), norm(self.loser_name), norm(self.score)
        ])
        return hashlib.sha256(base.encode()).hexdigest()[:24]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def fetch(url: str) -> str:
    ctx = ssl._create_unverified_context()
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30, context=ctx) as resp:
        return resp.read().decode("utf-8", "ignore")


def discover_current_atp_pages(home_html: str) -> list[tuple[str, str, int]]:
    pages = []
    for m in re.finditer(r'href="([^"]*/current/(\d{4})ATP[^"#]+\.html)"[^>]*>Results and Forecasts</a>', home_html):
        href, year = m.groups()
        url = urljoin(TA_HOME, href)
        # Exclude Challenger pages; current main-tour pages use .../YYYYATP<Event>.html
        if "Challenger" in url:
            continue
        nm = re.search(r"/current/\d{4}ATP(.+)\.html", url)
        if not nm:
            continue
        name = html_lib.unescape(nm.group(1)).replace("%27", "'")
        pages.append((url, name, int(year)))
    # preserve order, de-dupe
    out = []
    seen = set()
    for p in pages:
        if p[0] not in seen:
            out.append(p); seen.add(p[0])
    return out


def clean_player(raw: str) -> tuple[str, Optional[str]]:
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", raw))
    country = None
    cm = re.search(r"\(([A-Z]{3})\)", text)
    if cm:
        country = cm.group(1)
        text = text[:cm.start()] + text[cm.end():]
    # Remove seed/entry annotations such as (3), (Q), (WC), (Alt), or empty remnants.
    text = re.sub(r"\((?:\d+|Q|WC|LL|Alt|PR|SE)?\)", " ", text, flags=re.I)
    text = re.sub(r"\b(?:Q|WC|LL|Alt|PR|SE)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text, country


def parse_score_and_loser(loser_plus_score: str) -> tuple[str, Optional[str], str]:
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", loser_plus_score))
    text = re.sub(r"\s+", " ", text).strip()
    # Score starts at first set-like token, WO/RET handled as part of score if present after set tokens.
    sm = re.search(r"\b(?:\d{1,2}-\d{1,2}(?:\([^)]*\))?|WO|W/O|RET|DEF)\b", text, flags=re.I)
    if not sm:
        loser_text, country = clean_player(text)
        return loser_text, country, ""
    loser_part = text[:sm.start()].strip()
    score = text[sm.start():].strip()
    loser_text, country = clean_player(loser_part)
    return loser_text, country, score


def split_completed_items(completed_html: str) -> list[str]:
    s = completed_html.replace("&nbsp;", " ")
    # split at round labels, retaining label
    label = r"(?:F|SF|QF|R16|R32|R64|R128|RR|BR|Q\d+)"
    parts = re.split(rf"(?=\b{label}\s*:)", s)
    return [p.strip() for p in parts if re.match(rf"^{label}\s*:", p.strip())]


def parse_matches(page_html: str, source_url: str, tournament: str, season: int) -> list[Match]:
    m = re.search(r"var completedSingles\s*=\s*'(.*?)';", page_html, re.S)
    if not m:
        return []
    completed = m.group(1)
    matches: list[Match] = []
    for item in split_completed_items(completed):
        item = item.strip()
        rm = re.match(r"^(?P<round>[^:]+):\s*(?P<body>.*)$", item, re.S)
        if not rm or " d. " not in item:
            continue
        rnd = rm.group("round").strip()
        body = rm.group("body").strip()
        wm = re.search(r"\s+d\.\s+", body)
        if not wm:
            continue
        winner_raw = body[:wm.start()]
        rest = body[wm.end():]
        winner, wcountry = clean_player(winner_raw)
        loser, lcountry, score = parse_score_and_loser(rest)
        if not winner or not loser:
            continue
        matches.append(Match(source_url, tournament, season, rnd, winner, wcountry, loser, lcountry, score, item))
    return matches


def setup_db(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    con.commit()


def upsert(con: sqlite3.Connection, matches: list[Match], now: str) -> tuple[int, int]:
    inserted = updated = 0
    for mt in matches:
        exists = con.execute("SELECT 1 FROM matches WHERE match_key=?", (mt.key,)).fetchone()
        if exists:
            updated += 1
            con.execute("""
                UPDATE matches SET source_url=?, tournament=?, season=?, round=?, winner_name=?, winner_country=?,
                    loser_name=?, loser_country=?, score=?, last_seen_utc=?, raw_html_fragment=?
                WHERE match_key=?
            """, (mt.source_url, mt.tournament, mt.season, mt.round, mt.winner_name, mt.winner_country,
                  mt.loser_name, mt.loser_country, mt.score, now, mt.raw, mt.key))
        else:
            inserted += 1
            con.execute("""
                INSERT INTO matches (match_key,tour,source,source_url,tournament,season,match_date,round,
                    winner_name,winner_country,loser_name,loser_country,score,best_of,indoor,surface,status,
                    stats_status,first_seen_utc,last_seen_utc,raw_html_fragment)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (mt.key, 'ATP', 'TennisAbstract current', mt.source_url, mt.tournament, mt.season, None,
                  mt.round, mt.winner_name, mt.winner_country, mt.loser_name, mt.loser_country, mt.score,
                  None, None, None, 'completed', 'pending', now, now, mt.raw))
            # Create null/pending side stat rows so coverage is explicit.
            for side, player in [('winner', mt.winner_name), ('loser', mt.loser_name)]:
                con.execute("""
                    INSERT OR IGNORE INTO match_side_stats (match_key, side, player_name, stats_source, stats_url, updated_utc)
                    VALUES (?,?,?,?,?,?)
                """, (mt.key, side, player, None, None, now))
    con.commit()
    return inserted, updated


def export_csv(con: sqlite3.Connection) -> int:
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = con.execute("""
        SELECT m.match_key,m.tour,m.season,m.match_date,m.tournament,m.round,m.winner_name,m.winner_country,
               m.loser_name,m.loser_country,m.score,m.status,m.stats_status,m.source,m.source_url,m.first_seen_utc,m.last_seen_utc,
               ws.aces AS winner_aces, ls.aces AS loser_aces,
               ws.first_serve_pct AS winner_first_serve_pct, ls.first_serve_pct AS loser_first_serve_pct,
               ws.service_points_won AS winner_service_points_won, ws.service_points_total AS winner_service_points_total,
               ls.service_points_won AS loser_service_points_won, ls.service_points_total AS loser_service_points_total,
               ws.return_points_won AS winner_return_points_won, ws.return_points_total AS winner_return_points_total,
               ls.return_points_won AS loser_return_points_won, ls.return_points_total AS loser_return_points_total,
               ws.total_points_won AS winner_total_points_won, ls.total_points_won AS loser_total_points_won
        FROM matches m
        LEFT JOIN match_side_stats ws ON ws.match_key=m.match_key AND ws.side='winner'
        LEFT JOIN match_side_stats ls ON ls.match_key=m.match_key AND ls.side='loser'
        ORDER BY COALESCE(m.match_date,''), m.season, m.tournament, m.round, m.winner_name
    """).fetchall()
    cols = [d[0] for d in con.execute("""
        SELECT m.match_key,m.tour,m.season,m.match_date,m.tournament,m.round,m.winner_name,m.winner_country,
               m.loser_name,m.loser_country,m.score,m.status,m.stats_status,m.source,m.source_url,m.first_seen_utc,m.last_seen_utc,
               ws.aces AS winner_aces, ls.aces AS loser_aces,
               ws.first_serve_pct AS winner_first_serve_pct, ls.first_serve_pct AS loser_first_serve_pct,
               ws.service_points_won AS winner_service_points_won, ws.service_points_total AS winner_service_points_total,
               ls.service_points_won AS loser_service_points_won, ls.service_points_total AS loser_service_points_total,
               ws.return_points_won AS winner_return_points_won, ws.return_points_total AS winner_return_points_total,
               ls.return_points_won AS loser_return_points_won, ls.return_points_total AS loser_return_points_total,
               ws.total_points_won AS winner_total_points_won, ls.total_points_won AS loser_total_points_won
        FROM matches m
        LEFT JOIN match_side_stats ws ON ws.match_key=m.match_key AND ws.side='winner'
        LEFT JOIN match_side_stats ls ON ls.match_key=m.match_key AND ls.side='loser'
        LIMIT 0
    """).description]
    with EXPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)
    return len(rows)


def write_report(con: sqlite3.Connection, now_local: str, sources: list[str], inserted: int, updated: int, blockers: list[str]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    path = REPORT_DIR / f"daily_scrape_{date}.md"
    total = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    side_total = con.execute("SELECT COUNT(*) FROM match_side_stats").fetchone()[0]
    complete_stats = con.execute("SELECT COUNT(*) FROM matches WHERE stats_status='complete'").fetchone()[0]
    pending_stats = con.execute("SELECT COUNT(*) FROM matches WHERE stats_status!='complete'").fetchone()[0]
    sample = con.execute("""
        SELECT tournament, round, winner_name, loser_name, score, stats_status, match_key
        FROM matches ORDER BY last_seen_utc DESC, tournament, round LIMIT 10
    """).fetchall()
    lines = [
        f"# Daily Tennis Scrape Report — {date}", "",
        f"Run time: {now_local}", "",
        "## Sources checked", *[f"- {s}" for s in sources], "",
        "## Rows inserted/updated", f"- Inserted matches: {inserted}", f"- Updated matches: {updated}", f"- Total matches in DB: {total}", f"- Side-stat rows: {side_total}", "",
        "## Stat coverage", f"- Matches with complete match-level stats: {complete_stats}", f"- Matches pending/null stats: {pending_stats}", "",
        "## Failures / blockers",
    ]
    if blockers:
        lines += [f"- {b}" for b in blockers]
    else:
        lines += ["- None for match-list collection. Match-level serve/return stats were not exposed on the accessible TennisAbstract current result pages and remain pending."]
    lines += ["", "## Sample rows", "", "| Tournament | Round | Winner | Loser | Score | Stats | Key |", "|---|---:|---|---|---|---|---|"]
    for row in sample:
        lines.append("| " + " | ".join(str(x) if x is not None else "" for x in row) + " |")
    lines += ["", "## Next recommended fix", "- Add a TennisAbstract match-stat endpoint/parser (or browser-assisted extraction if stats are rendered only after interaction). Current TA tournament pages provided completed singles result lists but not side-specific serve/return stat tables, so stats were explicitly left null/pending.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    now_local = datetime.now().astimezone().isoformat(timespec="seconds")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    sources, blockers = [], []
    all_matches: list[Match] = []
    try:
        home = fetch(TA_HOME)
        sources.append(TA_HOME)
    except Exception as e:
        blockers.append(f"TennisAbstract homepage fetch failed: {e!r}")
        home = ""
    pages = discover_current_atp_pages(home) if home else []
    if not pages:
        blockers.append("No current ATP TennisAbstract result pages discovered from homepage.")
    for url, tournament, season in pages:
        time.sleep(1.0)
        try:
            page = fetch(url)
            sources.append(url)
            parsed = parse_matches(page, url, tournament, season)
            all_matches.extend(parsed)
            if not parsed:
                blockers.append(f"No completedSingles parsed from {url}")
        except Exception as e:
            blockers.append(f"Fetch/parse failed for {url}: {e!r}")
    # de-dupe by stable key
    deduped = {}
    for mt in all_matches:
        deduped[mt.key] = mt
    all_matches = list(deduped.values())
    with sqlite3.connect(DB_PATH) as con:
        setup_db(con)
        con.execute("INSERT INTO scrape_runs (run_ts_utc, source, status, notes) VALUES (?,?,?,?)", (now_utc, "TennisAbstract current ATP", "ok" if all_matches else "blocked", f"parsed_matches={len(all_matches)}; blockers={'; '.join(blockers)}"))
        inserted, updated = upsert(con, all_matches, now_utc)
        exported = export_csv(con)
        report = write_report(con, now_local, sources, inserted, updated, blockers)
        counts = con.execute("SELECT COUNT(*), SUM(CASE WHEN stats_status='complete' THEN 1 ELSE 0 END), SUM(CASE WHEN stats_status!='complete' THEN 1 ELSE 0 END) FROM matches").fetchone()
        sample = con.execute("SELECT tournament, round, winner_name, loser_name, score, stats_status FROM matches ORDER BY last_seen_utc DESC LIMIT 5").fetchall()
    print(f"inserted={inserted} updated={updated} parsed={len(all_matches)} exported={exported}")
    print(f"db={DB_PATH}")
    print(f"export={EXPORT_PATH}")
    print(f"report={report}")
    print(f"counts total={counts[0]} stats_complete={counts[1] or 0} stats_pending={counts[2] or 0}")
    print("sample:")
    for s in sample:
        print(s)
    if blockers:
        print("blockers:")
        for b in blockers: print("-", b)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
