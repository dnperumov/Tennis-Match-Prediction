#!/usr/bin/env python3
"""Paper PnL backtest for the non-lopsided high-confidence tennis strategy.

Research only. This script never places bets or connects to any exchange/sportsbook.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from high_confidence_strategy import apply_strategy_filter  # noqa: E402

DEFAULT_PREDICTIONS = ROOT / "data" / "betting_research" / "latest_advanced_feature_predictions.csv"
DEFAULT_OUTPUT = ROOT / "data" / "betting_research" / "latest_high_confidence_pnl_backtest.json"
DEFAULT_PICKS = ROOT / "data" / "betting_research" / "latest_high_confidence_pnl_picks.csv"


def _picked_side_fields(row: pd.Series) -> tuple[str, float, int, float, float]:
    """Return picked side label, decimal odds, win flag, model prob, no-vig market prob."""
    pick_p1 = int(row["strategy_pick"]) == 1
    if pick_p1:
        return (
            "p1",
            float(row["p1_odds"]),
            int(row["result"] == 1),
            float(row["strategy_prob"]),
            float(row["strategy_market_prob"]),
        )
    return (
        "p2",
        float(row["p2_odds"]),
        int(row["result"] == 0),
        1.0 - float(row["strategy_prob"]),
        1.0 - float(row["strategy_market_prob"]),
    )


def build_paper_bets(
    df: pd.DataFrame,
    stake: float = 1.0,
    min_raw_edge: float = 0.0,
    **strategy_kwargs,
) -> pd.DataFrame:
    required = {"p1_odds", "p2_odds", "result"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns for PnL backtest: {sorted(missing)}")

    selected = apply_strategy_filter(df, **strategy_kwargs).copy()
    if selected.empty:
        return selected

    picked = selected.apply(_picked_side_fields, axis=1, result_type="expand")
    picked.columns = ["bet_side", "decimal_odds", "won", "bet_model_prob", "bet_market_no_vig_prob"]
    bets = pd.concat([selected.reset_index(drop=True), picked.reset_index(drop=True)], axis=1)
    bets["break_even_prob"] = 1.0 / bets["decimal_odds"].astype(float)
    bets["raw_edge"] = bets["bet_model_prob"] - bets["break_even_prob"]
    bets = bets[bets["raw_edge"] >= float(min_raw_edge)].copy()
    if bets.empty:
        return bets

    bets["stake"] = float(stake)
    bets["profit"] = bets["won"].astype(int).map({1: 1.0, 0: -1.0})
    bets.loc[bets["won"].eq(1), "profit"] = bets.loc[bets["won"].eq(1), "stake"] * (bets.loc[bets["won"].eq(1), "decimal_odds"] - 1.0)
    bets.loc[bets["won"].eq(0), "profit"] = -bets.loc[bets["won"].eq(0), "stake"]
    bets["running_profit"] = bets["profit"].cumsum()
    return bets


def summarize_bets(bets: pd.DataFrame, total_rows: int, stake: float, min_raw_edge: float) -> dict:
    if bets.empty:
        return {
            "strategy": "non_lopsided_high_confidence_paper_pnl",
            "total_rows": int(total_rows),
            "bets": 0,
            "coverage": 0.0,
            "stake": float(stake),
            "min_raw_edge": float(min_raw_edge),
            "profit": 0.0,
            "roi": None,
            "hit_rate": None,
        }
    profit = float(bets["profit"].sum())
    amount_staked = float(bets["stake"].sum())
    out = {
        "strategy": "non_lopsided_high_confidence_paper_pnl",
        "total_rows": int(total_rows),
        "bets": int(len(bets)),
        "coverage": float(len(bets) / total_rows) if total_rows else 0.0,
        "stake": float(stake),
        "min_raw_edge": float(min_raw_edge),
        "amount_staked": amount_staked,
        "profit": profit,
        "roi": float(profit / amount_staked) if amount_staked else None,
        "hit_rate": float(bets["won"].mean()),
        "avg_decimal_odds": float(bets["decimal_odds"].mean()),
        "avg_raw_edge": float(bets["raw_edge"].mean()),
        "avg_model_prob": float(bets["bet_model_prob"].mean()),
        "avg_market_no_vig_prob": float(bets["bet_market_no_vig_prob"].mean()),
    }
    if "date" in bets.columns:
        by_year = []
        dated = bets.copy()
        dated["year"] = pd.to_datetime(dated["date"], errors="coerce").dt.year
        for year, part in dated.dropna(subset=["year"]).groupby("year"):
            year_profit = float(part["profit"].sum())
            year_staked = float(part["stake"].sum())
            by_year.append({
                "year": int(year),
                "bets": int(len(part)),
                "profit": year_profit,
                "roi": float(year_profit / year_staked) if year_staked else None,
                "hit_rate": float(part["won"].mean()),
                "avg_decimal_odds": float(part["decimal_odds"].mean()),
            })
        out["by_year"] = by_year
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--picks-output", default=str(DEFAULT_PICKS))
    ap.add_argument("--stake", type=float, default=1.0)
    ap.add_argument("--min-raw-edge", type=float, default=0.0, help="Minimum model probability minus raw decimal-odds break-even probability for the picked side.")
    ap.add_argument("--model-col", default="residual_overlay_segment_tuned_p1")
    ap.add_argument("--market-col", default="implied_p1_no_vig")
    ap.add_argument("--min-model-confidence", type=float, default=0.65)
    ap.add_argument("--min-market-confidence", type=float, default=0.55)
    ap.add_argument("--max-market-confidence", type=float, default=0.80)
    ap.add_argument("--max-underperformance-risk", type=float, default=0.65)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    bets = build_paper_bets(
        df,
        stake=args.stake,
        min_raw_edge=args.min_raw_edge,
        model_col=args.model_col,
        market_col=args.market_col,
        min_model_confidence=args.min_model_confidence,
        min_market_confidence=args.min_market_confidence,
        max_market_confidence=args.max_market_confidence,
        max_underperformance_risk=args.max_underperformance_risk,
    )
    summary = summarize_bets(bets, total_rows=len(df), stake=args.stake, min_raw_edge=args.min_raw_edge)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")

    picks_output = Path(args.picks_output)
    picks_output.parent.mkdir(parents=True, exist_ok=True)
    export_cols = [
        c for c in [
            "date", "tourney_name", "round", "player1", "player2", "bet_side", "strategy_pick",
            "bet_model_prob", "bet_market_no_vig_prob", "decimal_odds", "break_even_prob", "raw_edge",
            "result", "won", "stake", "profit", "running_profit",
        ] if c in bets.columns
    ]
    bets.to_csv(picks_output, index=False, columns=export_cols if export_cols else None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
