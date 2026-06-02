#!/usr/bin/env python3
"""Backfill missing winner_stats/loser_stats in the 2025 WITH_SLM_STATS dataset.

Goal (Dennis): Use TennisAbstract tournament pages + player profiles to fill missing per-match
stats, keeping the dataset format the same as `atp_matches_2025_WITH_SLM_STATS.csv`.

This script is intentionally conservative:
- Only touches rows where winner_stats or loser_stats are missing/empty
- Matches profile rows back to dataset rows by:
  (tourney_name, round, winner_name, loser_name, score)
- Uses TennisAbstract player profile match logs to source stats

Usage (run from repo root):
  python3 ta_tourney_scrape/tools/backfill_missing_profile_stats_2025.py \
    --input ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS.csv \
    --output ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS_BACKFILLED.csv \
    --year 2025

Notes:
- Requires the existing ta_tourney_scrape Python deps (requests-cache, bs4, tenacity, unidecode).
- Writes a small JSON report alongside the output file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running without installing the ta_tourney_scrape package.
# (We don't have pip/venv in this environment.)
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_TA_SRC = _REPO_ROOT / "ta_tourney_scrape" / "src"
if _TA_SRC.exists():
    sys.path.insert(0, str(_TA_SRC))


def _is_missing(v: object) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"na", "nan", "none", "null"}


def _norm_space(s: str) -> str:
    return " ".join((s or "").strip().split())


def _norm_score(s: str) -> str:
    # use existing helper if available
    try:
        from ta_tourney_scrape.parse_profile import normalize_score

        return normalize_score(s or "")
    except Exception:
        return _norm_space(s)


def _norm_player(s: str) -> str:
    try:
        from ta_tourney_scrape.parse_profile import normalize_player_name

        return normalize_player_name(s or "")
    except Exception:
        return _norm_space(s).lower()


def _match_sig(row: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    return (
        _norm_space(row.get("tourney_name", "")),
        _norm_space(row.get("round", "")).upper(),
        _norm_player(row.get("winner_name", "")),
        _norm_player(row.get("loser_name", "")),
        _norm_score(row.get("score", "")),
    )


@dataclass
class BackfillHit:
    sig: Tuple[str, str, str, str, str]
    winner_profile_player: Optional[str] = None
    loser_profile_player: Optional[str] = None
    filled_winner_stats: bool = False
    filled_loser_stats: bool = False


def _find_in_profile(
    profile_matches: List[Dict[str, Any]],
    want_round: str,
    want_winner: str,
    want_loser: str,
    want_score: str,
) -> Optional[Dict[str, Any]]:
    """Find a match record in a player's profile scrape.

    We match primarily on opponent + score. Round is used as a tiebreak.
    """

    wl = _norm_player(want_winner)
    ll = _norm_player(want_loser)
    rs = _norm_score(want_score)
    rr = _norm_space(want_round).upper()

    candidates = []
    for m in profile_matches:
        mw = _norm_player(m.get("winner_name", ""))
        ml = _norm_player(m.get("loser_name", ""))
        ms = _norm_score(m.get("score", ""))
        mr = _norm_space(str(m.get("round", ""))).upper()

        if ms != rs:
            continue

        # Either orientation is acceptable, but we want exact pairing.
        if {mw, ml} != {wl, ll}:
            continue

        candidates.append((mr == rr, m))

    if not candidates:
        return None

    # Prefer round match
    candidates.sort(key=lambda t: (t[0],), reverse=True)
    return candidates[0][1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS.csv"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS_BACKFILLED.csv"),
    )
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--only-tournaments",
        type=str,
        default="Wimbledon,US Open",
        help="Comma-separated list of tourney_name values to target (default: Wimbledon,US Open)",
    )
    args = ap.parse_args()

    inp: Path = args.input
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    target_tourneys = {t.strip() for t in (args.only_tournaments or "").split(",") if t.strip()}

    # Import inside main so this script can be imported without deps.
    from ta_tourney_scrape.parse_profile import scrape_player_profile_matches
    from ta_tourney_scrape.ta_names import resolve_ta_player_name

    # Load rows
    with inp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit("CSV missing header")
        rows = list(reader)

    # Identify rows needing backfill
    needs = []
    for i, row in enumerate(rows):
        tn = _norm_space(row.get("tourney_name", ""))
        if target_tourneys and tn not in target_tourneys:
            continue

        ws_missing = _is_missing(row.get("winner_stats"))
        ls_missing = _is_missing(row.get("loser_stats"))
        if ws_missing or ls_missing:
            needs.append(i)

    report: Dict[str, Any] = {
        "input": str(inp),
        "output": str(args.output),
        "year": args.year,
        "target_tourneys": sorted(list(target_tourneys)),
        "rows_total": len(rows),
        "rows_targeted": len(needs),
        "hits": [],
        "misses": [],
    }

    # Backfill
    for idx in needs:
        row = rows[idx]
        sig = _match_sig(row)
        tn, rnd, w, l, sc = sig

        hit = BackfillHit(sig=sig)

        winner_name = _norm_space(row.get("winner_name", ""))
        loser_name = _norm_space(row.get("loser_name", ""))

        # Resolve to TA canonical names (improves profile URL resolution)
        w_ta = resolve_ta_player_name(winner_name)
        l_ta = resolve_ta_player_name(loser_name)

        # Winner profile
        try:
            w_profile = scrape_player_profile_matches(w_ta, args.year, tn)
            m = _find_in_profile(w_profile, rnd, winner_name, loser_name, row.get("score", ""))
            if m:
                if _is_missing(row.get("winner_stats")) and m.get("winner_stats"):
                    row["winner_stats"] = json.dumps(m.get("winner_stats"), ensure_ascii=False)
                    hit.filled_winner_stats = True
                if _is_missing(row.get("loser_stats")) and m.get("loser_stats"):
                    row["loser_stats"] = json.dumps(m.get("loser_stats"), ensure_ascii=False)
                    hit.filled_loser_stats = True
                hit.winner_profile_player = w_ta
        except Exception:
            pass

        # Loser profile (for cases where winner profile doesn't include needed side)
        try:
            l_profile = scrape_player_profile_matches(l_ta, args.year, tn)
            m2 = _find_in_profile(l_profile, rnd, winner_name, loser_name, row.get("score", ""))
            if m2:
                if _is_missing(row.get("winner_stats")) and m2.get("winner_stats"):
                    row["winner_stats"] = json.dumps(m2.get("winner_stats"), ensure_ascii=False)
                    hit.filled_winner_stats = True
                if _is_missing(row.get("loser_stats")) and m2.get("loser_stats"):
                    row["loser_stats"] = json.dumps(m2.get("loser_stats"), ensure_ascii=False)
                    hit.filled_loser_stats = True
                hit.loser_profile_player = l_ta
        except Exception:
            pass

        if hit.filled_winner_stats or hit.filled_loser_stats:
            report["hits"].append(hit.__dict__)
        else:
            report["misses"].append({
                "sig": sig,
                "winner": winner_name,
                "loser": loser_name,
                "score": row.get("score", ""),
            })

    # Write output CSV
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Write report
    rep_path = out.with_suffix(".backfill_report.json")
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"✅ wrote {out}")
    print(f"✅ wrote {rep_path}")
    print(f"hits: {len(report['hits'])}  misses: {len(report['misses'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
