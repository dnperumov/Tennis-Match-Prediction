#!/usr/bin/env python3
"""Generate the Streamlit/dashboard decision summary from current research artifacts.

Research only: this script summarizes model-quality diagnostics and paper-tracking
artifacts. It never places bets, connects to sportsbook execution, or performs
financial operations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tennis_ml.dashboard_summary import write_dashboard_summary_file  # noqa: E402

DEFAULT_ADVANCED = ROOT / "data" / "betting_research" / "latest_advanced_feature_model_research.json"
DEFAULT_ABILITY = ROOT / "data" / "betting_research" / "latest_ability_pressure_diagnostic.json"
DEFAULT_OUTPUT = ROOT / "data" / "betting_research" / "latest_dashboard_decision_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write latest tennis dashboard decision summary JSON.")
    parser.add_argument("--advanced", type=Path, default=DEFAULT_ADVANCED, help="Advanced all-data research JSON path.")
    parser.add_argument("--ability", type=Path, default=DEFAULT_ABILITY, help="Ability/routing diagnostic JSON path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Decision summary JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = write_dashboard_summary_file(args.advanced, args.ability, args.output)
    decision = summary.get("automation_recommendation", {}).get("decision", "unknown")
    best = summary.get("current_best", {}).get("model", "unknown")
    rows = summary.get("current_best", {}).get("rows", "unknown")
    print(f"wrote {args.output} decision={decision} current_best={best} rows={rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
