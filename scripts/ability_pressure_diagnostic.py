#!/usr/bin/env python3
"""Ability-pressure diagnostics for all-data tennis walk-forward predictions.

Research only: this script benchmarks model probabilities against market baselines and
writes diagnostic artifacts. It never places bets or performs financial execution.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "betting_research"

SIGNAL_SPECS = {
    "elo_diff": {"cuts": [-150, -50, 50, 150], "labels": ["fav_big_elo_deficit", "fav_small_elo_deficit", "fav_elo_neutral", "fav_small_elo_edge", "fav_big_elo_edge"]},
    "surface_elo_diff": {"cuts": [-150, -50, 50, 150], "labels": ["fav_big_surface_elo_deficit", "fav_small_surface_elo_deficit", "fav_surface_elo_neutral", "fav_small_surface_elo_edge", "fav_big_surface_elo_edge"]},
    "serve_return_ability_diff": {"cuts": [-0.08, -0.025, 0.025, 0.08], "labels": ["fav_big_serve_return_deficit", "fav_small_serve_return_deficit", "fav_serve_return_neutral", "fav_small_serve_return_edge", "fav_big_serve_return_edge"]},
    "fatigue_adjusted_ability_diff": {"cuts": [-0.08, -0.025, 0.025, 0.08], "labels": ["fav_big_fatigue_adj_deficit", "fav_small_fatigue_adj_deficit", "fav_fatigue_adj_neutral", "fav_small_fatigue_adj_edge", "fav_big_fatigue_adj_edge"]},
    "surface_ability_diff": {"cuts": [-0.08, -0.025, 0.025, 0.08], "labels": ["fav_big_surface_ability_deficit", "fav_small_surface_ability_deficit", "fav_surface_ability_neutral", "fav_small_surface_ability_edge", "fav_big_surface_ability_edge"]},
    "recent_form_ability_diff": {"cuts": [-0.08, -0.025, 0.025, 0.08], "labels": ["fav_big_recent_form_deficit", "fav_small_recent_form_deficit", "fav_recent_form_neutral", "fav_small_recent_form_edge", "fav_big_recent_form_edge"]},
}

PRESSURE_CUTS = [-0.10, -0.04, -0.015, 0.015, 0.04, 0.10]
PRESSURE_LABELS = [
    "model_much_lower_on_favorite",
    "model_lower_on_favorite",
    "model_slightly_lower_on_favorite",
    "model_near_baseline_on_favorite",
    "model_slightly_higher_on_favorite",
    "model_higher_on_favorite",
    "model_much_higher_on_favorite",
]
FAVORITE_STRENGTH_CUTS = [0.58, 0.66, 0.75, 0.85]
FAVORITE_STRENGTH_LABELS = ["near_pickem_fav", "modest_fav", "solid_fav", "heavy_fav", "overwhelming_fav"]


def clip_prob(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(1e-6, 1 - 1e-6)


def bucket_numeric_signal(series: pd.Series, cuts: list[float], labels: list[str]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    bucketed = pd.cut(values, bins=[-np.inf, *cuts, np.inf], labels=labels, include_lowest=True)
    as_object = bucketed.astype("object")
    return as_object.where(pd.notna(as_object), "missing").astype(str)


def add_favorite_perspective_columns(
    df: pd.DataFrame,
    *,
    baseline_prob_col: str,
    model_prob_col: str,
    signal_cols: Iterable[str],
) -> pd.DataFrame:
    out = df.copy()
    baseline_p1 = clip_prob(out[baseline_prob_col])
    model_p1 = clip_prob(out[model_prob_col])
    p1_fav = baseline_p1 >= 0.5
    out["favorite_side"] = np.where(p1_fav, "p1", "p2")
    out["baseline_favorite_prob"] = np.where(p1_fav, baseline_p1, 1 - baseline_p1)
    out["model_favorite_prob"] = np.where(p1_fav, model_p1, 1 - model_p1)
    out["model_minus_baseline_favorite_prob"] = out["model_favorite_prob"] - out["baseline_favorite_prob"]
    out["favorite_won"] = np.where(p1_fav, out["result"].astype(int), 1 - out["result"].astype(int))
    out["baseline_favorite_pick_correct"] = out["favorite_won"]
    for col in signal_cols:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            out[f"favorite_{col}"] = np.where(p1_fav, values, -values)
    return out


def metrics_for(df: pd.DataFrame, prob_col: str, baseline_prob_col: str, *, actual_col: str = "result") -> dict:
    if df.empty:
        return {"rows": 0}
    actual = pd.to_numeric(df[actual_col], errors="coerce").astype(int)
    prob = clip_prob(df[prob_col])
    baseline = clip_prob(df[baseline_prob_col])
    out = {
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(actual, prob >= 0.5)),
        "actual_rate": float(actual.mean()),
        "mean_prob": float(prob.mean()),
        "baseline_probability_col": baseline_prob_col,
        "log_loss": float(log_loss(actual, prob, labels=[0, 1])),
        "market_log_loss": float(log_loss(actual, baseline, labels=[0, 1])),
        "brier": float(brier_score_loss(actual, prob)),
        "market_brier": float(brier_score_loss(actual, baseline)),
    }
    out["model_minus_market_log_loss"] = out["log_loss"] - out["market_log_loss"]
    out["model_minus_market_brier"] = out["brier"] - out["market_brier"]
    try:
        out["roc_auc"] = float(roc_auc_score(actual, prob)) if actual.nunique() > 1 else None
    except ValueError:
        out["roc_auc"] = None
    return out


def stable_segment_summary(
    df: pd.DataFrame,
    segment_col: str,
    prob_col: str,
    baseline_prob_col: str,
    *,
    actual_col: str = "result",
    min_rows: int = 150,
    min_years: int = 3,
    min_stable_year_share: float = 0.6,
    direction: str = "lags",
    top_n: int = 15,
) -> list[dict]:
    rows = []
    for segment, seg_df in df.groupby(segment_col, dropna=False):
        if len(seg_df) < min_rows:
            continue
        overall = metrics_for(seg_df, prob_col, baseline_prob_col, actual_col=actual_col)
        yearly = []
        beat_ll = beat_brier = lag_ll = lag_brier = 0
        min_year_rows = math.inf
        for year, year_df in seg_df.groupby("year"):
            ym = metrics_for(year_df, prob_col, baseline_prob_col, actual_col=actual_col)
            ym = {k: ym[k] for k in ["rows", "log_loss", "market_log_loss", "brier", "market_brier", "model_minus_market_log_loss", "model_minus_market_brier"]}
            ym["year"] = int(year)
            yearly.append(ym)
            min_year_rows = min(min_year_rows, int(ym["rows"]))
            beat_ll += int(ym["model_minus_market_log_loss"] < 0)
            beat_brier += int(ym["model_minus_market_brier"] < 0)
            lag_ll += int(ym["model_minus_market_log_loss"] > 0)
            lag_brier += int(ym["model_minus_market_brier"] > 0)
        year_count = len(yearly)
        if year_count < min_years:
            continue
        needed = math.ceil(min_stable_year_share * year_count)
        if direction == "beats":
            if not (overall["model_minus_market_log_loss"] < 0 and overall["model_minus_market_brier"] < 0 and beat_ll >= needed and beat_brier >= needed):
                continue
            weighted = -overall["model_minus_market_log_loss"] * overall["rows"]
        elif direction == "lags":
            if not (overall["model_minus_market_log_loss"] > 0 and overall["model_minus_market_brier"] > 0 and lag_ll >= needed and lag_brier >= needed):
                continue
            weighted = overall["model_minus_market_log_loss"] * overall["rows"]
        else:
            raise ValueError("direction must be 'beats' or 'lags'")
        overall.update(
            {
                "segment_col": segment_col,
                "segment": str(segment),
                "years": sorted(int(y) for y in seg_df["year"].dropna().unique()),
                "year_count": int(year_count),
                "min_year_rows": int(min_year_rows if min_year_rows is not math.inf else 0),
                "years_model_beats_market_log_loss": int(beat_ll),
                "years_model_beats_market_brier": int(beat_brier),
                "years_model_lags_market_log_loss": int(lag_ll),
                "years_model_lags_market_brier": int(lag_brier),
                "yearly_model_minus_market": yearly,
                "weighted_log_loss_damage" if direction == "lags" else "weighted_log_loss_improvement": float(weighted),
            }
        )
        rows.append(overall)
    sort_key = "weighted_log_loss_damage" if direction == "lags" else "weighted_log_loss_improvement"
    return sorted(rows, key=lambda r: r[sort_key], reverse=True)[:top_n]


def add_ability_consensus_segments(df: pd.DataFrame, signal_cols: Iterable[str]) -> pd.DataFrame:
    """Add multi-signal favorite-side ability consensus buckets.

    Each supplied signal is expected in favorite perspective as ``favorite_<signal>``.
    Positive values above the neutral cut support the market favorite; negative values
    below the neutral cut oppose it. Missing/neutral signals do not vote. The combined
    segment is reporting-only evidence for routing/shrinkage hypotheses.
    """
    out = df.copy()
    support = pd.Series(0, index=out.index, dtype=int)
    oppose = pd.Series(0, index=out.index, dtype=int)
    for col in signal_cols:
        fav_col = f"favorite_{col}"
        spec = SIGNAL_SPECS.get(col)
        if fav_col not in out.columns or not spec:
            continue
        cuts = spec["cuts"]
        # The middle two cuts define the neutral zone for all current signal specs.
        low_neutral = float(cuts[1])
        high_neutral = float(cuts[2])
        values = pd.to_numeric(out[fav_col], errors="coerce")
        support += (values > high_neutral).fillna(False).astype(int)
        oppose += (values < low_neutral).fillna(False).astype(int)
    out["favorite_ability_support_count"] = support
    out["favorite_ability_oppose_count"] = oppose
    net = support - oppose
    out["favorite_ability_consensus_bucket"] = np.select(
        [net >= 3, net >= 1, net <= -3, net <= -1],
        [
            "ability_strongly_supports_favorite",
            "ability_mildly_supports_favorite",
            "ability_strongly_opposes_favorite",
            "ability_mildly_opposes_favorite",
        ],
        default="ability_split_or_neutral",
    )
    if "favorite_strength_bucket" not in out.columns:
        out["favorite_strength_bucket"] = bucket_numeric_signal(out["baseline_favorite_prob"], FAVORITE_STRENGTH_CUTS, FAVORITE_STRENGTH_LABELS)
    if "model_favorite_pressure_bucket" not in out.columns:
        out["model_favorite_pressure_bucket"] = bucket_numeric_signal(out["model_minus_baseline_favorite_prob"], PRESSURE_CUTS, PRESSURE_LABELS)
    out["ability_consensus_pressure_segment"] = out["favorite_ability_consensus_bucket"].astype(str) + " | " + out["favorite_strength_bucket"].astype(str) + " | " + out["model_favorite_pressure_bucket"].astype(str)
    return out


def add_diagnostic_buckets(df: pd.DataFrame, signal_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    out["favorite_strength_bucket"] = bucket_numeric_signal(out["baseline_favorite_prob"], FAVORITE_STRENGTH_CUTS, FAVORITE_STRENGTH_LABELS)
    out["model_favorite_pressure_bucket"] = bucket_numeric_signal(out["model_minus_baseline_favorite_prob"], PRESSURE_CUTS, PRESSURE_LABELS)
    for col in signal_cols:
        fav_col = f"favorite_{col}"
        if fav_col in out.columns and col in SIGNAL_SPECS:
            spec = SIGNAL_SPECS[col]
            out[f"favorite_{col}_bucket"] = bucket_numeric_signal(out[fav_col], spec["cuts"], spec["labels"])
            out[f"{col}_pressure_segment"] = out[f"favorite_{col}_bucket"] + " | " + out["model_favorite_pressure_bucket"]
            out[f"{col}_strength_pressure_segment"] = out[f"favorite_{col}_bucket"] + " | " + out["favorite_strength_bucket"] + " | " + out["model_favorite_pressure_bucket"]
    out = add_ability_consensus_segments(out, signal_cols)
    return out


def no_lookahead_segment_fallback_routing(
    df: pd.DataFrame,
    *,
    segment_col: str,
    model_col: str,
    baseline_col: str,
    actual_col: str = "result",
    min_train_rows: int = 180,
    min_train_years: int = 3,
    min_stable_year_share: float = 0.6,
) -> dict:
    """Route future rows in prior-OOS stable lagging segments back to baseline.

    For each held-out year, only earlier OOS years are scanned for stable lagging
    ability-pressure segments. Matching current-year rows are routed to the selected
    baseline probability. This is a diagnostic risk-control policy, not betting
    execution or live model selection.
    """
    work = df.copy()
    years = sorted(int(y) for y in pd.to_numeric(work["year"], errors="coerce").dropna().unique())
    routed_col = f"{model_col}_ability_pressure_fallback"
    work[routed_col] = clip_prob(work[model_col])
    work["ability_pressure_fallback_routed"] = False
    yearly_routing = []
    for year in years:
        train = work[work["year"] < year].copy()
        current_mask = work["year"].eq(year)
        if train.empty or int(train["year"].nunique()) < min_train_years:
            yearly_routing.append({"year": int(year), "train_rows": int(len(train)), "flagged_segments": [], "routed_rows": 0})
            continue
        lagging = stable_segment_summary(
            train,
            segment_col,
            model_col,
            baseline_col,
            actual_col=actual_col,
            min_rows=min_train_rows,
            min_years=min_train_years,
            min_stable_year_share=min_stable_year_share,
            direction="lags",
            top_n=10_000,
        )
        flagged = sorted({row["segment"] for row in lagging})
        route_mask = current_mask & work[segment_col].astype(str).isin(flagged)
        work.loc[route_mask, routed_col] = clip_prob(work.loc[route_mask, baseline_col])
        work.loc[route_mask, "ability_pressure_fallback_routed"] = True
        yearly_routing.append(
            {
                "year": int(year),
                "train_rows": int(len(train)),
                "flagged_segments": flagged,
                "routed_rows": int(route_mask.sum()),
            }
        )
    model_metrics = metrics_for(work, model_col, baseline_col, actual_col=actual_col)
    routed_metrics = metrics_for(work, routed_col, baseline_col, actual_col=actual_col)
    baseline_metrics = metrics_for(work.assign(**{baseline_col: clip_prob(work[baseline_col])}), baseline_col, baseline_col, actual_col=actual_col)
    return {
        "segment_col": segment_col,
        "model_probability_col": model_col,
        "baseline_probability_col": baseline_col,
        "routed_probability_col": routed_col,
        "rows": int(len(work)),
        "routed_rows": int(work["ability_pressure_fallback_routed"].sum()),
        "model_metrics": model_metrics,
        "routed_metrics": routed_metrics,
        "baseline_metrics": baseline_metrics,
        "routed_minus_model_log_loss": float(routed_metrics["log_loss"] - model_metrics["log_loss"]),
        "routed_minus_model_brier": float(routed_metrics["brier"] - model_metrics["brier"]),
        "routed_minus_baseline_log_loss": float(routed_metrics["log_loss"] - routed_metrics["market_log_loss"]),
        "routed_minus_baseline_brier": float(routed_metrics["brier"] - routed_metrics["market_brier"]),
        "yearly_routing": yearly_routing,
        "research_only_guardrail": "Historical no-lookahead diagnostic only; no betting execution.",
    }


def no_lookahead_multi_segment_fallback_routing(
    df: pd.DataFrame,
    *,
    segment_cols: list[str],
    model_col: str,
    baseline_col: str,
    actual_col: str = "result",
    min_train_rows: int = 180,
    min_train_years: int = 3,
    min_stable_year_share: float = 0.6,
) -> dict:
    """Route future rows matching any prior stable-lag ability segment.

    This combines several ability-pressure segment families into one no-lookahead
    risk-control policy. For each held-out year, every segment column is scanned
    using only earlier OOS years; a current-year row is routed to the baseline if
    it matches at least one flagged segment in any selected segment column. Rows
    matched by multiple diagnostics are routed once.
    """
    work = df.copy()
    years = sorted(int(y) for y in pd.to_numeric(work["year"], errors="coerce").dropna().unique())
    routed_col = f"{model_col}_ability_pressure_multi_fallback"
    work[routed_col] = clip_prob(work[model_col])
    work["ability_pressure_multi_fallback_routed"] = False
    yearly_routing = []
    available_segment_cols = [col for col in segment_cols if col in work.columns]

    for year in years:
        train = work[work["year"] < year].copy()
        current_mask = work["year"].eq(year)
        flagged_by_column: dict[str, list[str]] = {}
        route_mask = pd.Series(False, index=work.index)
        if train.empty or int(train["year"].nunique()) < min_train_years:
            yearly_routing.append(
                {
                    "year": int(year),
                    "train_rows": int(len(train)),
                    "flagged_segments_by_column": flagged_by_column,
                    "routed_rows": 0,
                }
            )
            continue

        for segment_col in available_segment_cols:
            lagging = stable_segment_summary(
                train,
                segment_col,
                model_col,
                baseline_col,
                actual_col=actual_col,
                min_rows=min_train_rows,
                min_years=min_train_years,
                min_stable_year_share=min_stable_year_share,
                direction="lags",
                top_n=10_000,
            )
            flagged = sorted({row["segment"] for row in lagging})
            if flagged:
                flagged_by_column[segment_col] = flagged
                route_mask = route_mask | (current_mask & work[segment_col].astype(str).isin(flagged))

        work.loc[route_mask, routed_col] = clip_prob(work.loc[route_mask, baseline_col])
        work.loc[route_mask, "ability_pressure_multi_fallback_routed"] = True
        yearly_routing.append(
            {
                "year": int(year),
                "train_rows": int(len(train)),
                "flagged_segments_by_column": flagged_by_column,
                "routed_rows": int(route_mask.sum()),
            }
        )

    model_metrics = metrics_for(work, model_col, baseline_col, actual_col=actual_col)
    routed_metrics = metrics_for(work, routed_col, baseline_col, actual_col=actual_col)
    baseline_metrics = metrics_for(work.assign(**{baseline_col: clip_prob(work[baseline_col])}), baseline_col, baseline_col, actual_col=actual_col)
    return {
        "segment_cols": available_segment_cols,
        "model_probability_col": model_col,
        "baseline_probability_col": baseline_col,
        "routed_probability_col": routed_col,
        "rows": int(len(work)),
        "routed_rows": int(work["ability_pressure_multi_fallback_routed"].sum()),
        "model_metrics": model_metrics,
        "routed_metrics": routed_metrics,
        "baseline_metrics": baseline_metrics,
        "routed_minus_model_log_loss": float(routed_metrics["log_loss"] - model_metrics["log_loss"]),
        "routed_minus_model_brier": float(routed_metrics["brier"] - model_metrics["brier"]),
        "routed_minus_baseline_log_loss": float(routed_metrics["log_loss"] - routed_metrics["market_log_loss"]),
        "routed_minus_baseline_brier": float(routed_metrics["brier"] - routed_metrics["market_brier"]),
        "yearly_routing": yearly_routing,
        "research_only_guardrail": "Historical no-lookahead diagnostic only; no betting execution.",
    }


def build_report(
    df: pd.DataFrame,
    *,
    model_col: str,
    baseline_col: str,
    min_rows: int,
    min_years: int,
    top_n: int,
) -> dict:
    signal_cols = [c for c in SIGNAL_SPECS if c in df.columns]
    work = add_favorite_perspective_columns(df, baseline_prob_col=baseline_col, model_prob_col=model_col, signal_cols=signal_cols)
    work = add_diagnostic_buckets(work, signal_cols)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "data/betting_research/latest_advanced_feature_predictions.csv",
        "model_probability_col": model_col,
        "baseline_probability_col": baseline_col,
        "rows": int(len(work)),
        "years": sorted(int(y) for y in work["year"].dropna().unique()),
        "overall_model_vs_baseline": metrics_for(work, model_col, baseline_col),
        "research_only_guardrail": "No betting execution; historical model-quality diagnostics only.",
        "min_rows": min_rows,
        "min_years": min_years,
        "top_n": top_n,
        "sections": {},
        "routing_diagnostics": {},
    }
    strength_segment_cols = []
    for col in signal_cols:
        for segment_col in [f"{col}_pressure_segment", f"{col}_strength_pressure_segment"]:
            if segment_col not in work.columns:
                continue
            report["sections"][f"where_{model_col}_ability_pressure_stably_lags_{baseline_col}_{segment_col}"] = stable_segment_summary(
                work, segment_col, model_col, baseline_col, min_rows=min_rows, min_years=min_years, direction="lags", top_n=top_n
            )
            report["sections"][f"where_{model_col}_ability_pressure_beats_{baseline_col}_{segment_col}"] = stable_segment_summary(
                work, segment_col, model_col, baseline_col, min_rows=min_rows, min_years=min_years, direction="beats", top_n=top_n
            )
            if segment_col.endswith("_strength_pressure_segment"):
                strength_segment_cols.append(segment_col)
                report["routing_diagnostics"][f"{model_col}_fallback_to_{baseline_col}_{segment_col}"] = no_lookahead_segment_fallback_routing(
                    work,
                    segment_col=segment_col,
                    model_col=model_col,
                    baseline_col=baseline_col,
                    min_train_rows=min_rows,
                    min_train_years=min_years,
                )
    if "ability_consensus_pressure_segment" in work.columns:
        consensus_segment_col = "ability_consensus_pressure_segment"
        report["sections"][f"where_{model_col}_ability_pressure_stably_lags_{baseline_col}_{consensus_segment_col}"] = stable_segment_summary(
            work, consensus_segment_col, model_col, baseline_col, min_rows=min_rows, min_years=min_years, direction="lags", top_n=top_n
        )
        report["sections"][f"where_{model_col}_ability_pressure_beats_{baseline_col}_{consensus_segment_col}"] = stable_segment_summary(
            work, consensus_segment_col, model_col, baseline_col, min_rows=min_rows, min_years=min_years, direction="beats", top_n=top_n
        )
        strength_segment_cols.append(consensus_segment_col)
        report["routing_diagnostics"][f"{model_col}_fallback_to_{baseline_col}_{consensus_segment_col}"] = no_lookahead_segment_fallback_routing(
            work,
            segment_col=consensus_segment_col,
            model_col=model_col,
            baseline_col=baseline_col,
            min_train_rows=min_rows,
            min_train_years=min_years,
        )
    if strength_segment_cols:
        report["routing_diagnostics"][f"{model_col}_fallback_to_{baseline_col}_all_ability_strength_pressure_segments"] = no_lookahead_multi_segment_fallback_routing(
            work,
            segment_cols=strength_segment_cols,
            model_col=model_col,
            baseline_col=baseline_col,
            min_train_rows=min_rows,
            min_train_years=min_years,
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=OUT_DIR / "latest_advanced_feature_predictions.csv")
    parser.add_argument("--model-col", default="residual_overlay_filtered_p1")
    parser.add_argument("--baseline-col", default="market_bin_recalibrated_p1")
    parser.add_argument("--min-rows", type=int, default=180)
    parser.add_argument("--min-years", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "latest_ability_pressure_diagnostic.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    needed = {"year", "result", args.model_col, args.baseline_col}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    report = build_report(
        df,
        model_col=args.model_col,
        baseline_col=args.baseline_col,
        min_rows=args.min_rows,
        min_years=args.min_years,
        top_n=args.top_n,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    stamped = args.output.with_name(f"ability_pressure_diagnostic_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    stamped.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "stamped_output": str(stamped), "rows": report["rows"], "years": report["years"]}, indent=2))


if __name__ == "__main__":
    main()
