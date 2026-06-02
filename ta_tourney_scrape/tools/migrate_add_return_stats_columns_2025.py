#!/usr/bin/env python3
"""Migrate 2025 WITH_SLM_STATS CSV to include separate serve/return JSON fields.

Input currently has:
- winner_stats (serve-style JSON)
- loser_stats (serve-style JSON)

Output adds:
- winner_stats_serve, winner_stats_return
- loser_stats_serve, loser_stats_return

Rules:
- winner_stats -> winner_stats_serve (if winner_stats_serve empty)
- loser_stats  -> loser_stats_serve (if loser_stats_serve empty)
- winner_stats_return / loser_stats_return initialized empty
- Keeps original winner_stats/loser_stats columns (for backward compatibility)

Usage:
  python3 ta_tourney_scrape/tools/migrate_add_return_stats_columns_2025.py \
    --in ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS.csv \
    --out ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS_SR.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def is_missing(v: object) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"na", "nan", "none", "null"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    inp: Path = args.inp
    out: Path = args.out

    with inp.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        rows = list(r)

    new_cols = [
        "winner_stats_serve",
        "winner_stats_return",
        "loser_stats_serve",
        "loser_stats_return",
    ]

    # Insert new cols right after existing winner_stats/loser_stats if present; else append.
    def insert_after(cols, after, inserts):
        if after in cols:
            idx = cols.index(after) + 1
            for c in inserts:
                if c not in cols:
                    cols.insert(idx, c)
                    idx += 1
        else:
            for c in inserts:
                if c not in cols:
                    cols.append(c)

    insert_after(fieldnames, "winner_stats", ["winner_stats_serve", "winner_stats_return"])
    insert_after(fieldnames, "loser_stats", ["loser_stats_serve", "loser_stats_return"])

    moved_w = 0
    moved_l = 0

    for row in rows:
        for c in new_cols:
            row.setdefault(c, "")

        if is_missing(row.get("winner_stats_serve")) and not is_missing(row.get("winner_stats")):
            row["winner_stats_serve"] = row.get("winner_stats", "")
            moved_w += 1
        if is_missing(row.get("loser_stats_serve")) and not is_missing(row.get("loser_stats")):
            row["loser_stats_serve"] = row.get("loser_stats", "")
            moved_l += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"✅ wrote {out}")
    print(f"moved winner_stats -> winner_stats_serve: {moved_w}")
    print(f"moved loser_stats  -> loser_stats_serve: {moved_l}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
