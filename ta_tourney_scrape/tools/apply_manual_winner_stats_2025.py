#!/usr/bin/env python3
"""Apply manual winner_stats for specific 2025 matches into WITH_SLM_STATS dataset.

Use when we have the TennisAbstract winner match-log row (DR/A%/DF%/1stIn/1st%/2nd%/BPSvd/Time)
from Dennis and need to update the canonical CSV.

Conservative:
- Creates a backup copy first (caller can also copy externally)
- Matches rows by (tourney_name, round, winner_name, loser_name, score)
- Only writes winner_stats if currently empty

Usage:
  python3 ta_tourney_scrape/tools/apply_manual_winner_stats_2025.py \
    --csv ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS.csv \
    --out ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS_MANUAL.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def is_missing(v: object) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"na", "nan", "none", "null"}


def sig(row: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    return (
        (row.get("tourney_name") or "").strip(),
        (row.get("round") or "").strip(),
        (row.get("winner_name") or "").strip(),
        (row.get("loser_name") or "").strip(),
        (row.get("score") or "").strip(),
    )


def pct_to_float(p: str) -> Optional[float]:
    p = (p or "").strip().replace("%", "")
    if not p:
        return None
    try:
        return float(p) / 100.0
    except Exception:
        return None


@dataclass
class ManualStatRow:
    tourney_name: str
    round: str
    winner: str
    loser: str
    score: str
    dr: float
    ace_pct: float
    df_pct: float
    first_in_pct: float
    first_won_pct: float
    second_won_pct: float
    bpsvd_num: int
    bpsvd_den: int
    time: str

    def to_winner_stats(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "dr": self.dr,
            "ace_pct": self.ace_pct,
            "df_pct": self.df_pct,
            "first_in_pct": self.first_in_pct,
            "first_won_pct": self.first_won_pct,
            "second_won_pct": self.second_won_pct,
            "bpsvd_num": self.bpsvd_num,
            "bpsvd_den": self.bpsvd_den,
            "time": self.time,
        }
        if self.bpsvd_den > 0:
            out["bpsvd_pct"] = self.bpsvd_num / self.bpsvd_den
        else:
            out["bpsvd_pct"] = None
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = args.csv
    if not src.exists():
        raise SystemExit(f"Missing: {src}")

    # Manual rows provided by Dennis (Wimbledon 4 matches)
    manual: List[ManualStatRow] = [
        ManualStatRow(
            tourney_name="Wimbledon",
            round="Q1",
            winner="Filip Misolic",
            loser="Guy Den Ouden",
            score="3-6 6-1 7-5",
            dr=1.24,
            ace_pct=pct_to_float("2.4%") or 0.024,
            df_pct=pct_to_float("3.5%") or 0.035,
            first_in_pct=pct_to_float("47.1%") or 0.471,
            first_won_pct=pct_to_float("72.5%") or 0.725,
            second_won_pct=pct_to_float("68.9%") or 0.689,
            bpsvd_num=4,
            bpsvd_den=5,
            time="1:54",
        ),
        ManualStatRow(
            tourney_name="Wimbledon",
            round="Q2",
            winner="Filip Misolic",
            loser="Gauthier Onclin",
            score="6-4 6-2",
            dr=1.25,
            ace_pct=pct_to_float("4.3%") or 0.043,
            df_pct=pct_to_float("4.3%") or 0.043,
            first_in_pct=pct_to_float("73.9%") or 0.739,
            first_won_pct=pct_to_float("58.8%") or 0.588,
            second_won_pct=pct_to_float("50.0%") or 0.5,
            bpsvd_num=5,
            bpsvd_den=8,
            time="1:23",
        ),
        ManualStatRow(
            tourney_name="Wimbledon",
            round="R128",
            winner="Taylor Fritz",
            loser="Mpetshi Perricard",
            score="6-7(6) 6-7(8) 6-4 7-6(6) 6-4",
            dr=1.48,
            ace_pct=pct_to_float("18.1%") or 0.181,
            df_pct=pct_to_float("1.3%") or 0.013,
            first_in_pct=pct_to_float("67.5%") or 0.675,
            first_won_pct=pct_to_float("85.2%") or 0.852,
            second_won_pct=pct_to_float("71.2%") or 0.712,
            bpsvd_num=0,
            bpsvd_den=0,
            time="3:25",
        ),
        ManualStatRow(
            tourney_name="Wimbledon",
            round="R64",
            winner="Flavio Cobolli",
            loser="Jack Pinnington Jones",
            score="6-1 7-6(6) 6-2",
            dr=1.47,
            ace_pct=pct_to_float("2.1%") or 0.021,
            df_pct=pct_to_float("4.2%") or 0.042,
            first_in_pct=pct_to_float("56.3%") or 0.563,
            first_won_pct=pct_to_float("64.8%") or 0.648,
            second_won_pct=pct_to_float("69.0%") or 0.69,
            bpsvd_num=2,
            bpsvd_den=3,
            time="1:55",
        ),
    ]

    manual_map = {
        (m.tourney_name, m.round, m.winner, m.loser, m.score): m for m in manual
    }

    with src.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        if not fieldnames:
            raise SystemExit("Missing header")
        rows = list(r)

    applied = 0
    skipped_present = 0
    not_found = []

    for key, m in manual_map.items():
        found = False
        for row in rows:
            if sig(row) == key:
                found = True
                if not is_missing(row.get("winner_stats")):
                    skipped_present += 1
                else:
                    row["winner_stats"] = json.dumps(m.to_winner_stats(), ensure_ascii=False)
                    applied += 1
                break
        if not found:
            not_found.append(key)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"✅ wrote {out}")
    print(f"applied={applied} skipped_present={skipped_present} not_found={len(not_found)}")
    if not_found:
        print("not found:")
        for k in not_found:
            print(k)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
