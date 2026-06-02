#!/usr/bin/env python3
"""Apply manual serve/return stats (winner+loser) into the 2025 SR CSV.

Targets columns:
- winner_stats_serve, winner_stats_return
- loser_stats_serve, loser_stats_return

Matches by exact:
(tourney_name, round, winner_name, loser_name, score)

Usage:
  python3 ta_tourney_scrape/tools/apply_manual_serve_return_stats_2025.py \
    --csv ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS_SR.csv \
    --out ta_tourney_scrape/data/atp_matches_2025_WITH_SLM_STATS_SR_MANUAL.csv
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
class ServeStats:
    dr: float
    ace_pct: float
    df_pct: float
    first_in_pct: float
    first_won_pct: float
    second_won_pct: float
    bpsvd_num: int
    bpsvd_den: int
    time: str

    def to_json(self) -> Dict[str, object]:
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
        out["bpsvd_pct"] = (self.bpsvd_num / self.bpsvd_den) if self.bpsvd_den else None
        return out


@dataclass
class ReturnStats:
    dr: float
    tpw_pct: float
    rpw_pct: float
    v_ace_pct: float
    v_first_won_pct: float
    v_second_won_pct: float
    bpcnv_num: int
    bpcnv_den: int
    time: str

    def to_json(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "dr": self.dr,
            "tpw_pct": self.tpw_pct,
            "rpw_pct": self.rpw_pct,
            "v_ace_pct": self.v_ace_pct,
            "v_first_won_pct": self.v_first_won_pct,
            "v_second_won_pct": self.v_second_won_pct,
            "bpcnv_num": self.bpcnv_num,
            "bpcnv_den": self.bpcnv_den,
            "time": self.time,
        }
        out["bpcnv_pct"] = (self.bpcnv_num / self.bpcnv_den) if self.bpcnv_den else None
        return out


@dataclass
class ManualMatch:
    key: Tuple[str, str, str, str, str]
    winner_serve: Optional[ServeStats] = None
    winner_return: Optional[ReturnStats] = None
    loser_serve: Optional[ServeStats] = None
    loser_return: Optional[ReturnStats] = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = args.csv
    out = args.out

    with src.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        rows = list(r)

    required_cols = ["winner_stats_serve", "winner_stats_return", "loser_stats_serve", "loser_stats_return"]
    for c in required_cols:
        if c not in fieldnames:
            raise SystemExit(f"Missing required column {c}. Run migrate_add_return_stats_columns_2025.py first.")

    # Manual data from Dennis (Wimbledon 4 matches)
    manual: List[ManualMatch] = [
        ManualMatch(
            key=("Wimbledon", "Q1", "Filip Misolic", "Guy Den Ouden", "3-6 6-1 7-5"),
            winner_return=ReturnStats(
                dr=1.24,
                tpw_pct=pct_to_float("53.5%") or 0.535,
                rpw_pct=pct_to_float("36.5%") or 0.365,
                v_ace_pct=pct_to_float("4.7%") or 0.047,
                v_first_won_pct=pct_to_float("32.6%") or 0.326,
                v_second_won_pct=pct_to_float("41.0%") or 0.41,
                bpcnv_num=3,
                bpcnv_den=9,
                time="1:54",
            ),
            loser_serve=ServeStats(
                dr=0.81,
                ace_pct=pct_to_float("4.7%") or 0.047,
                df_pct=pct_to_float("2.4%") or 0.024,
                first_in_pct=pct_to_float("54.1%") or 0.541,
                first_won_pct=pct_to_float("67.4%") or 0.674,
                second_won_pct=pct_to_float("59.0%") or 0.59,
                bpsvd_num=6,
                bpsvd_den=9,
                time="1:54",
            ),
            loser_return=ReturnStats(
                dr=0.81,
                tpw_pct=pct_to_float("46.5%") or 0.465,
                rpw_pct=pct_to_float("29.4%") or 0.294,
                v_ace_pct=pct_to_float("2.4%") or 0.024,
                v_first_won_pct=pct_to_float("27.5%") or 0.275,
                v_second_won_pct=pct_to_float("31.1%") or 0.311,
                bpcnv_num=1,
                bpcnv_den=5,
                time="1:54",
            ),
        ),
        ManualMatch(
            key=("Wimbledon", "Q2", "Filip Misolic", "Gauthier Onclin", "6-4 6-2"),
            winner_return=ReturnStats(
                dr=1.25,
                tpw_pct=pct_to_float("55.6%") or 0.556,
                rpw_pct=pct_to_float("54.5%") or 0.545,
                v_ace_pct=pct_to_float("3.6%") or 0.036,
                v_first_won_pct=pct_to_float("61.1%") or 0.611,
                v_second_won_pct=pct_to_float("42.1%") or 0.421,
                bpcnv_num=6,
                bpcnv_den=11,
                time="1:23",
            ),
            loser_serve=ServeStats(
                dr=0.80,
                ace_pct=pct_to_float("3.6%") or 0.036,
                df_pct=pct_to_float("5.5%") or 0.055,
                first_in_pct=pct_to_float("65.5%") or 0.655,
                first_won_pct=pct_to_float("38.9%") or 0.389,
                second_won_pct=pct_to_float("57.9%") or 0.579,
                bpsvd_num=5,
                bpsvd_den=11,
                time="1:23",
            ),
            loser_return=ReturnStats(
                dr=0.80,
                tpw_pct=pct_to_float("44.4%") or 0.444,
                rpw_pct=pct_to_float("43.5%") or 0.435,
                v_ace_pct=pct_to_float("4.3%") or 0.043,
                v_first_won_pct=pct_to_float("41.2%") or 0.412,
                v_second_won_pct=pct_to_float("50.0%") or 0.5,
                bpcnv_num=3,
                bpcnv_den=8,
                time="1:23",
            ),
        ),
        ManualMatch(
            key=("Wimbledon", "R128", "Taylor Fritz", "Mpetshi Perricard", "6-7(6) 6-7(8) 6-4 7-6(6) 6-4"),
            winner_return=ReturnStats(
                dr=1.48,
                tpw_pct=pct_to_float("52.8%") or 0.528,
                rpw_pct=pct_to_float("28.6%") or 0.286,
                v_ace_pct=pct_to_float("20.0%") or 0.20,
                v_first_won_pct=pct_to_float("17.9%") or 0.179,
                v_second_won_pct=pct_to_float("50.0%") or 0.5,
                bpcnv_num=2,
                bpcnv_den=6,
                time="3:25",
            ),
            loser_serve=ServeStats(
                dr=0.68,
                ace_pct=pct_to_float("20.0%") or 0.20,
                df_pct=pct_to_float("7.0%") or 0.07,
                first_in_pct=pct_to_float("66.5%") or 0.665,
                first_won_pct=pct_to_float("82.1%") or 0.821,
                second_won_pct=pct_to_float("50.0%") or 0.5,
                bpsvd_num=4,
                bpsvd_den=6,
                time="3:25",
            ),
            loser_return=ReturnStats(
                dr=0.68,
                tpw_pct=pct_to_float("47.2%") or 0.472,
                rpw_pct=pct_to_float("19.4%") or 0.194,
                v_ace_pct=pct_to_float("18.1%") or 0.181,
                v_first_won_pct=pct_to_float("14.8%") or 0.148,
                v_second_won_pct=pct_to_float("28.8%") or 0.288,
                bpcnv_num=0,
                bpcnv_den=0,
                time="3:25",
            ),
        ),
        ManualMatch(
            key=("Wimbledon", "R64", "Flavio Cobolli", "Jack Pinnington Jones", "6-1 7-6(6) 6-2"),
            winner_return=ReturnStats(
                dr=1.47,
                tpw_pct=pct_to_float("58.1%") or 0.581,
                rpw_pct=pct_to_float("48.9%") or 0.489,
                v_ace_pct=pct_to_float("4.4%") or 0.044,
                v_first_won_pct=pct_to_float("40.0%") or 0.40,
                v_second_won_pct=pct_to_float("62.9%") or 0.629,
                bpcnv_num=6,
                bpcnv_den=9,
                time="1:55",
            ),
            loser_serve=ServeStats(
                dr=0.68,
                ace_pct=pct_to_float("4.4%") or 0.044,
                df_pct=pct_to_float("4.4%") or 0.044,
                first_in_pct=pct_to_float("61.1%") or 0.611,
                first_won_pct=pct_to_float("60.0%") or 0.60,
                second_won_pct=pct_to_float("37.1%") or 0.371,
                bpsvd_num=3,
                bpsvd_den=9,
                time="1:55",
            ),
            loser_return=ReturnStats(
                dr=0.68,
                tpw_pct=pct_to_float("41.9%") or 0.419,
                rpw_pct=pct_to_float("33.3%") or 0.333,
                v_ace_pct=pct_to_float("2.1%") or 0.021,
                v_first_won_pct=pct_to_float("35.2%") or 0.352,
                v_second_won_pct=pct_to_float("31.0%") or 0.31,
                bpcnv_num=1,
                bpcnv_den=3,
                time="1:55",
            ),
        ),
    ]

    index = {sig(r): r for r in rows}

    applied = 0
    missing = []

    for m in manual:
        row = index.get(m.key)
        if not row:
            missing.append(m.key)
            continue

        if m.winner_serve and is_missing(row.get("winner_stats_serve")):
            row["winner_stats_serve"] = json.dumps(m.winner_serve.to_json(), ensure_ascii=False)
            applied += 1
        if m.winner_return and is_missing(row.get("winner_stats_return")):
            row["winner_stats_return"] = json.dumps(m.winner_return.to_json(), ensure_ascii=False)
            applied += 1
        if m.loser_serve and is_missing(row.get("loser_stats_serve")):
            row["loser_stats_serve"] = json.dumps(m.loser_serve.to_json(), ensure_ascii=False)
            applied += 1
        if m.loser_return and is_missing(row.get("loser_stats_return")):
            row["loser_stats_return"] = json.dumps(m.loser_return.to_json(), ensure_ascii=False)
            applied += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    report = {
        "csv": str(src),
        "out": str(out),
        "applied_fields": applied,
        "missing_rows": missing,
    }
    rep_path = out.with_suffix(out.suffix + ".manual_report.json")
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"✅ wrote {out}")
    print(f"✅ wrote {rep_path}")
    print(f"applied_fields={applied} missing_rows={len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
