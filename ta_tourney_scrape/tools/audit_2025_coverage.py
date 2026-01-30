#!/usr/bin/env python3
"""Audit coverage/completeness of atp_matches_2025_WITH_SLM_STATS.csv.

Dependency-free (stdlib only). Produces a markdown report summarizing:
- row count
- tournament-level missingness (stats + metadata)
- surface normalization issues (HardDraw/ClayDraw/etc)

Usage:
  python3 ta_tourney_scrape/tools/audit_2025_coverage.py \
    --input ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS.csv \
    --out ta_tourney_scrape/data/output/coverage_audit_2025.md

Note: This script does not modify data.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


def is_missing(v: object) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"na", "nan", "none", "null"}


def has_stats_blob(v: object) -> bool:
    # winner_stats/loser_stats appear to be string blobs (json-ish). Treat non-empty as present.
    return not is_missing(v)


@dataclass
class TourneyAgg:
    rows: int = 0
    missing_surface: int = 0
    missing_start_date: int = 0
    missing_match_num: int = 0
    missing_winner_stats: int = 0
    missing_loser_stats: int = 0
    missing_any_stats: int = 0
    surface_vals: Counter = None
    levels: Counter = None
    rounds: Counter = None

    def __post_init__(self):
        self.surface_vals = Counter()
        self.levels = Counter()
        self.rounds = Counter()


def pct(m: int, n: int) -> str:
    return f"{(m / n * 100.0):.1f}%" if n else "0.0%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("ta_tourney_scrape/data/output/coverage_audit_2025.md"),
    )
    ap.add_argument("--top", type=int, default=40, help="Top N tournaments to list")
    args = ap.parse_args()

    inp: Path = args.input
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    by_tourney: dict[str, TourneyAgg] = defaultdict(TourneyAgg)
    rows = 0

    with inp.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows += 1
            tn = (row.get("tourney_name") or "").strip() or "__MISSING_TOURNEY_NAME__"
            agg = by_tourney[tn]
            agg.rows += 1

            surface = (row.get("surface") or "").strip()
            start_date = (row.get("start_date") or "").strip()
            match_num = (row.get("match_num") or "").strip()
            level = (row.get("tourney_level") or "").strip()
            rnd = (row.get("round") or "").strip().upper()

            if not surface:
                agg.missing_surface += 1
            if not start_date:
                agg.missing_start_date += 1
            if not match_num:
                agg.missing_match_num += 1

            ws_ok = has_stats_blob(row.get("winner_stats"))
            ls_ok = has_stats_blob(row.get("loser_stats"))
            if not ws_ok:
                agg.missing_winner_stats += 1
            if not ls_ok:
                agg.missing_loser_stats += 1
            if not (ws_ok and ls_ok):
                agg.missing_any_stats += 1

            agg.surface_vals[surface] += 1
            agg.levels[level] += 1
            agg.rounds[rnd] += 1

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    # Surface weirdness
    tourneys_with_draw_surface = [
        tn
        for tn, agg in by_tourney.items()
        if any((s or "").endswith("Draw") for s in agg.surface_vals.keys())
    ]

    # Rank tournaments by missing stats
    ranked = sorted(by_tourney.items(), key=lambda kv: kv[1].missing_any_stats, reverse=True)

    lines: list[str] = []
    lines.append(f"# 2025 coverage audit (WITH_SLM_STATS)\n\n")
    lines.append(f"Input: `{inp}`\n\n")
    lines.append(f"Total rows: **{rows}**\n\n")

    lines.append("## Top tournaments with missing stats\n\n")
    lines.append("(Missing stats = winner_stats or loser_stats empty)\n\n")
    lines.append("| Tournament | Rows | Missing stats | Missing surface | Missing start_date | Missing match_num |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")

    shown = 0
    for tn, agg in ranked:
        if shown >= args.top:
            break
        if agg.missing_any_stats == 0:
            continue
        lines.append(
            f"| {tn} | {agg.rows} | {agg.missing_any_stats} ({pct(agg.missing_any_stats, agg.rows)})"
            f" | {agg.missing_surface} ({pct(agg.missing_surface, agg.rows)})"
            f" | {agg.missing_start_date} ({pct(agg.missing_start_date, agg.rows)})"
            f" | {agg.missing_match_num} ({pct(agg.missing_match_num, agg.rows)}) |\n"
        )
        shown += 1

    if shown == 0:
        lines.append("No tournaments have missing winner/loser stats blobs (by this heuristic).\n")

    lines.append("\n## Surface normalization issues\n\n")
    lines.append(
        f"Tournaments with `*Draw` surface values present: **{len(tourneys_with_draw_surface)}**\n\n"
    )
    for tn in tourneys_with_draw_surface[:50]:
        agg = by_tourney[tn]
        lines.append(f"- {tn}: {dict(agg.surface_vals)}\n")

    out.write_text("".join(lines), encoding="utf-8")
    print(f"✅ wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
