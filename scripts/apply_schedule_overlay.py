#!/usr/bin/env python3
"""Apply real schedule/date overlays to the current enriched export.

Research-only data utility. This script does not call sportsbooks or place bets.

Expected overlay CSV columns, with flexible matching:
- match_date: required real scheduled/completed match date
- Optional exact key: match_key
- Or fuzzy key fields: tourney_name/tournament, round, player1/player2 OR winner_name/loser_name

The script preserves all existing export columns and only updates/creates schedule
metadata columns. Use this when a reliable official schedule/draw source has been
collected separately and you want to fix collapsed fallback dates.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = ROOT / "data" / "exports" / "atp_matches_enriched_current.csv"
DEFAULT_REPORT = ROOT / "data" / "reports" / "schedule_overlay_report.json"


def _norm(value: object) -> str:
    return " ".join(str(value or "").lower().strip().split())


def _players_key(a: object, b: object) -> tuple[str, str]:
    first, second = sorted((_norm(a), _norm(b)))
    return first, second


def _overlay_row_key(row: pd.Series) -> tuple[str, str, tuple[str, str]]:
    tournament = _norm(row.get("tourney_name") or row.get("tournament"))
    round_name = _norm(row.get("round"))
    p1 = row.get("player1") if "player1" in row else row.get("winner_name")
    p2 = row.get("player2") if "player2" in row else row.get("loser_name")
    return tournament, round_name, _players_key(p1, p2)


def apply_schedule_overlay(export_df: pd.DataFrame, overlay_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if "match_date" not in overlay_df.columns:
        raise ValueError("overlay CSV must include match_date")
    out = export_df.copy()
    out["schedule_source"] = out.get("schedule_source", "original_export")
    out["schedule_date_quality"] = out.get("schedule_date_quality", "original_or_fallback")

    updated = 0
    unmatched = []
    exact_by_key = "match_key" in out.columns and "match_key" in overlay_df.columns
    fuzzy_index = {}
    if not exact_by_key:
        for idx, row in out.iterrows():
            fuzzy_index.setdefault(_overlay_row_key(row), []).append(idx)

    for _, row in overlay_df.iterrows():
        schedule_date = pd.to_datetime(row.get("match_date"), errors="coerce")
        if pd.isna(schedule_date):
            unmatched.append({"reason": "invalid_match_date", "row": row.to_dict()})
            continue
        source = str(row.get("source") or row.get("schedule_source") or "schedule_overlay_csv")
        target_indices: list[int] = []
        if exact_by_key and row.get("match_key") in set(out["match_key"]):
            target_indices = out.index[out["match_key"] == row.get("match_key")].tolist()
        else:
            target_indices = fuzzy_index.get(_overlay_row_key(row), [])
        if len(target_indices) != 1:
            unmatched.append({"reason": "no_unique_match", "matches": len(target_indices), "row": row.to_dict()})
            continue
        idx = target_indices[0]
        out.at[idx, "match_date"] = schedule_date.strftime("%Y-%m-%d")
        out.at[idx, "schedule_source"] = source
        out.at[idx, "schedule_date_quality"] = "real_schedule_overlay"
        updated += 1

    report = {
        "export_rows": int(len(export_df)),
        "overlay_rows": int(len(overlay_df)),
        "updated_rows": int(updated),
        "unmatched_rows": int(len(unmatched)),
        "unmatched_examples": unmatched[:20],
    }
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overlay_csv", type=Path)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    export_df = pd.read_csv(args.export)
    overlay_df = pd.read_csv(args.overlay_csv)
    updated, report = apply_schedule_overlay(export_df, overlay_df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.no_backup:
        backup = args.out.with_suffix(args.out.suffix + ".bak")
        shutil.copy2(args.out, backup)
        report["backup"] = str(backup)
    updated.to_csv(args.out, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
