#!/usr/bin/env python3
"""Evaluate a non-lopsided high-confidence tennis model strategy.

Research only. This script never places bets or executes financial transactions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "data" / "betting_research" / "latest_advanced_feature_predictions.csv"
DEFAULT_OUTPUT = ROOT / "data" / "betting_research" / "latest_high_confidence_strategy.json"


def _series_or_default(df: pd.DataFrame, column: str, default) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def strategy_masks(
    df: pd.DataFrame,
    model_col: str = "residual_overlay_segment_tuned_p1",
    market_col: str = "implied_p1_no_vig",
    min_model_confidence: float = 0.65,
    min_market_confidence: float = 0.55,
    max_market_confidence: float = 0.80,
    max_underperformance_risk: float = 0.65,
) -> dict[str, pd.Series]:
    model_p = df[model_col].astype(float)
    market_p = df[market_col].astype(float)
    model_pick = model_p.ge(0.5)
    market_pick = market_p.ge(0.5)
    model_conf = np.maximum(model_p, 1.0 - model_p)
    market_conf = np.maximum(market_p, 1.0 - market_p)
    risk = _series_or_default(df, "underperformance_risk", 0.0).astype(float)
    round_group = _series_or_default(df, "round_group", "unknown").astype(str)

    masks = {
        "model_confident": model_conf.ge(min_model_confidence),
        "market_confident": market_conf.ge(min_market_confidence),
        "not_extreme_market_favorite": market_conf.le(max_market_confidence),
        "model_market_agree": model_pick.eq(market_pick),
        "risk_allowed": risk.lt(max_underperformance_risk),
        "not_early_high_risk": ~(round_group.eq("early") & risk.ge(0.50)),
    }
    masks["selected"] = pd.Series(True, index=df.index)
    for mask in masks.values():
        masks["selected"] &= mask
    return masks


def apply_strategy_filter(
    df: pd.DataFrame,
    model_col: str = "residual_overlay_segment_tuned_p1",
    market_col: str = "implied_p1_no_vig",
    **kwargs,
) -> pd.DataFrame:
    masks = strategy_masks(df, model_col=model_col, market_col=market_col, **kwargs)
    selected = df[masks["selected"]].copy()
    selected["strategy_prob"] = selected[model_col].astype(float)
    selected["strategy_pick"] = selected["strategy_prob"].ge(0.5).astype(int)
    selected["strategy_market_prob"] = selected[market_col].astype(float)
    return selected


def exclusion_reasons(df: pd.DataFrame, masks: dict[str, pd.Series]) -> dict[str, int]:
    reasons = {}
    labels = {
        "model_confident": "model_not_confident",
        "market_confident": "market_not_confident",
        "not_extreme_market_favorite": "extreme_market_favorite",
        "model_market_agree": "model_market_disagree",
        "risk_allowed": "underperformance_risk_high",
        "not_early_high_risk": "early_round_high_risk",
    }
    for key, label in labels.items():
        reasons[label] = int((~masks[key]).sum())
    return reasons


def summarize_strategy(
    df: pd.DataFrame,
    model_col: str = "residual_overlay_segment_tuned_p1",
    market_col: str = "implied_p1_no_vig",
    **kwargs,
) -> dict:
    masks = strategy_masks(df, model_col=model_col, market_col=market_col, **kwargs)
    selected = apply_strategy_filter(df, model_col=model_col, market_col=market_col, **kwargs)
    if selected.empty:
        return {
            "strategy": "non_lopsided_model_market_agreement",
            "rows": int(len(df)),
            "selected_rows": 0,
            "coverage": 0.0,
            "accuracy": None,
            "log_loss": None,
            "brier": None,
            "exclusion_reasons": exclusion_reasons(df, masks),
        }

    y = selected["result"].astype(int)
    p = selected["strategy_prob"].astype(float).clip(1e-6, 1 - 1e-6)
    pick = selected["strategy_pick"].astype(int)
    market_conf = np.maximum(selected[market_col].astype(float), 1.0 - selected[market_col].astype(float))
    model_conf = np.maximum(selected[model_col].astype(float), 1.0 - selected[model_col].astype(float))
    return {
        "strategy": "non_lopsided_model_market_agreement",
        "model_col": model_col,
        "market_col": market_col,
        "criteria": {
            "min_model_confidence": kwargs.get("min_model_confidence", 0.65),
            "min_market_confidence": kwargs.get("min_market_confidence", 0.55),
            "max_market_confidence": kwargs.get("max_market_confidence", 0.80),
            "max_underperformance_risk": kwargs.get("max_underperformance_risk", 0.65),
            "requires_model_market_agreement": True,
            "excludes_early_round_high_risk": True,
        },
        "rows": int(len(df)),
        "selected_rows": int(len(selected)),
        "coverage": float(len(selected) / len(df)) if len(df) else 0.0,
        "accuracy": float((pick == y).mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "avg_model_confidence": float(model_conf.mean()),
        "avg_market_confidence": float(market_conf.mean()),
        "exclusion_reasons": exclusion_reasons(df, masks),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--model-col", default="residual_overlay_segment_tuned_p1")
    ap.add_argument("--market-col", default="implied_p1_no_vig")
    ap.add_argument("--min-model-confidence", type=float, default=0.65)
    ap.add_argument("--min-market-confidence", type=float, default=0.55)
    ap.add_argument("--max-market-confidence", type=float, default=0.80)
    ap.add_argument("--max-underperformance-risk", type=float, default=0.65)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    summary = summarize_strategy(
        df,
        model_col=args.model_col,
        market_col=args.market_col,
        min_model_confidence=args.min_model_confidence,
        min_market_confidence=args.min_market_confidence,
        max_market_confidence=args.max_market_confidence,
        max_underperformance_risk=args.max_underperformance_risk,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
