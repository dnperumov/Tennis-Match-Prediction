#!/usr/bin/env python3
"""Fill missing winner_stats/loser_stats in 2025 WITH_SLM_STATS using other 2025 CSV snapshots.

Why:
- Some matches have empty stats blobs in the canonical file.
- We already have multiple 2025 CSV exports in ta_tourney_scrape/data/* that may contain those stats.
- This avoids scraping TennisAbstract when robots.txt disallows some endpoints.

Conservative:
- Only fills fields when target row is missing/empty.
- Match identity: (tourney_name, round, winner_name, loser_name, score)
- Keeps file format identical to the input (same columns/order).

Usage:
  python3 ta_tourney_scrape/tools/fill_missing_stats_from_other_2025_csvs.py \
    --target ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS.csv \
    --out ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS_FILLED.csv

Optionally replace in-place yourself after verifying.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def is_missing(v: object) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"na", "nan", "none", "null"}


def norm_space(s: str) -> str:
    return " ".join((s or "").strip().split())


def sig(row: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    return (
        norm_space(row.get("tourney_name", "")),
        norm_space(row.get("round", "")).upper(),
        norm_space(row.get("winner_name", "")).lower(),
        norm_space(row.get("loser_name", "")).lower(),
        norm_space(row.get("score", "")),
    )


@dataclass
class Hit:
    signature: Tuple[str, str, str, str, str]
    source_file: str
    filled_winner_stats: bool
    filled_loser_stats: bool


def load_source_rows(files: Iterable[Path]) -> Dict[Tuple[str, str, str, str, str], Dict[str, str]]:
    """Return map signature -> best source row (prefers rows with more stats present)."""
    best: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = {}
    best_score: Dict[Tuple[str, str, str, str, str], int] = {}

    def score_row(r: Dict[str, str]) -> int:
        sc = 0
        for k in ("winner_stats", "loser_stats", "dr", "time"):
            if k in r and not is_missing(r.get(k)):
                sc += 1
        return sc

    for p in files:
        try:
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    s = sig(row)
                    sc = score_row(row)
                    if s not in best_score or sc > best_score[s]:
                        best[s] = {**row, "__source_file": str(p)}
                        best_score[s] = sc
        except Exception:
            continue

    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        type=Path,
        default=Path("ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS_FILLED.csv"),
    )
    ap.add_argument(
        "--sources-glob",
        type=str,
        default="ta_tourney_scrape/data/atp_matches_2025_*.csv",
        help="Glob for candidate 2025 CSV sources (excluding target)",
    )
    args = ap.parse_args()

    target = args.target
    if not target.exists():
        raise SystemExit(f"Missing target: {target}")

    candidates = [Path(p) for p in glob.glob(args.sources_glob)]
    candidates = [p for p in candidates if p.exists() and p.resolve() != target.resolve()]

    src_map = load_source_rows(candidates)

    with target.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit("Target missing header")
        rows = list(reader)

    hits: List[Hit] = []
    misses = 0

    for row in rows:
        ws_missing = is_missing(row.get("winner_stats"))
        ls_missing = is_missing(row.get("loser_stats"))
        if not (ws_missing or ls_missing):
            continue

        s = sig(row)
        src = src_map.get(s)
        if not src:
            misses += 1
            continue

        filled_w = False
        filled_l = False
        if ws_missing and not is_missing(src.get("winner_stats")):
            row["winner_stats"] = src.get("winner_stats", "")
            filled_w = True
        if ls_missing and not is_missing(src.get("loser_stats")):
            row["loser_stats"] = src.get("loser_stats", "")
            filled_l = True

        # Also fill metadata if blank (safe)
        for k in ("ta_url", "scrape_method", "winner_seed", "loser_seed", "winner_ioc", "loser_ioc", "match_num", "surface", "start_date"):
            if k in row and is_missing(row.get(k)) and k in src and not is_missing(src.get(k)):
                row[k] = src.get(k, "")

        hits.append(Hit(signature=s, source_file=src.get("__source_file", ""), filled_winner_stats=filled_w, filled_loser_stats=filled_l))

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    rep = {
        "target": str(target),
        "out": str(out),
        "sources_considered": [str(p) for p in candidates],
        "hits": [h.__dict__ for h in hits],
        "misses": misses,
    }
    rep_path = out.with_suffix(out.suffix + ".fill_report.json")
    rep_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"✅ wrote {out}")
    print(f"✅ wrote {rep_path}")
    print(f"hits: {len(hits)}  misses: {misses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
