#!/usr/bin/env python3
"""Walk-forward specialist-model research for tennis betting predictions.

Research only: no betting execution, no sportsbook connectivity.

Compares the existing global model against multiple specialized model families:
- per-surface models
- Grand Slam vs non-Grand-Slam models
- top-10 involvement buckets
- combined surface x top-10 and surface x Grand Slam buckets

All features come from `make_side_dataset`, which is rolling/no-lookahead and updates
player/H2H state only after each date is emitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from betting_research_pipeline import build_model, feature_columns, load_odds_data, make_side_dataset  # noqa: E402
from niche_submarket_research import add_candidate_columns  # noqa: E402

OUT_DIR = ROOT / "data" / "betting_research"


def top10_bucket(df: pd.DataFrame) -> pd.Series:
    both = df["both_top10"].fillna(0).astype(int).eq(1)
    any_ = df["any_top10"].fillna(0).astype(int).eq(1)
    return pd.Series(np.where(both, "both_top10", np.where(any_, "one_top10", "no_top10")), index=df.index)


def grand_slam_bucket(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(df["series"].astype(str).eq("Grand Slam"), "grand_slam", "non_grand_slam"), index=df.index)


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["top10_bucket"] = top10_bucket(df)
    df["slam_bucket"] = grand_slam_bucket(df)
    df["surface_bucket"] = df["surface"].fillna("Unknown").astype(str)
    df["surface_x_top10"] = df["surface_bucket"] + " | " + df["top10_bucket"]
    df["surface_x_slam"] = df["surface_bucket"] + " | " + df["slam_bucket"]
    return df


def safe_metric(fn, *args):
    try:
        return float(fn(*args))
    except Exception:
        return None


def metrics_for(df: pd.DataFrame, prob_col: str) -> dict:
    if df.empty:
        return {"rows": 0}
    y = df["result"].astype(int)
    p = df[prob_col].clip(1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    return {
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": safe_metric(roc_auc_score, y, p) if y.nunique() == 2 else None,
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "market_log_loss": float(log_loss(y, df["implied_p1_no_vig"].clip(1e-6, 1 - 1e-6))),
        "market_brier": float(brier_score_loss(y, df["implied_p1_no_vig"].clip(1e-6, 1 - 1e-6))),
        "mean_prob": float(p.mean()),
        "actual_rate": float(y.mean()),
    }


def candidate_betting(df: pd.DataFrame, prob_col: str, threshold: float) -> dict:
    tmp = df.copy()
    tmp["model_p1"] = tmp[prob_col]
    tmp = add_candidate_columns(tmp)
    bets = tmp[tmp["candidate_edge"] >= threshold]
    if bets.empty:
        return {"threshold": threshold, "bets": 0, "profit": 0.0, "roi": 0.0, "hit_rate": None, "avg_odds": None, "avg_edge": None}
    return {
        "threshold": threshold,
        "bets": int(len(bets)),
        "profit": float(bets["candidate_profit"].sum()),
        "roi": float(bets["candidate_profit"].mean()),
        "hit_rate": float(bets["candidate_result"].mean()),
        "avg_odds": float(bets["candidate_odds"].mean()),
        "avg_edge": float(bets["candidate_edge"].mean()),
    }


def fit_predict_global(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    model = build_model()
    model.fit(train[features], train["result"].astype(int))
    return model.predict_proba(test[features])[:, 1]


def fit_predict_specialist(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    bucket_col: str,
    min_train: int,
    min_test: int,
) -> tuple[np.ndarray, list[dict]]:
    probs = np.full(len(test), np.nan)
    logs: list[dict] = []
    global_fallback = None
    for bucket in sorted(test[bucket_col].dropna().unique()):
        train_mask = train[bucket_col].eq(bucket)
        test_mask = test[bucket_col].eq(bucket)
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_test < min_test:
            logs.append({"bucket": str(bucket), "status": "skipped_small_test", "train_rows": n_train, "test_rows": n_test})
            continue
        if n_train >= min_train and train.loc[train_mask, "result"].nunique() == 2:
            model = build_model()
            model.fit(train.loc[train_mask, features], train.loc[train_mask, "result"].astype(int))
            probs[test_mask.to_numpy()] = model.predict_proba(test.loc[test_mask, features])[:, 1]
            logs.append({"bucket": str(bucket), "status": "specialist", "train_rows": n_train, "test_rows": n_test})
        else:
            if global_fallback is None:
                global_fallback = build_model()
                global_fallback.fit(train[features], train["result"].astype(int))
            probs[test_mask.to_numpy()] = global_fallback.predict_proba(test.loc[test_mask, features])[:, 1]
            logs.append({"bucket": str(bucket), "status": "global_fallback", "train_rows": n_train, "test_rows": n_test})
    # If any rows were skipped due to tiny test buckets, fallback to global for complete metrics.
    if np.isnan(probs).any():
        if global_fallback is None:
            global_fallback = build_model()
            global_fallback.fit(train[features], train["result"].astype(int))
        missing = np.isnan(probs)
        probs[missing] = global_fallback.predict_proba(test.loc[missing, features])[:, 1]
    return probs, logs


def calibration_bins(df: pd.DataFrame, prob_col: str, bins: int = 10) -> list[dict]:
    tmp = df.copy()
    tmp["prob_bin"] = pd.qcut(tmp[prob_col].rank(method="first"), bins, labels=False, duplicates="drop")
    rows = []
    for b, g in tmp.groupby("prob_bin"):
        rows.append({
            "bin": int(b),
            "rows": int(len(g)),
            "avg_prob": float(g[prob_col].mean()),
            "actual_rate": float(g["result"].mean()),
            "calibration_error": float(g[prob_col].mean() - g["result"].mean()),
            "avg_market": float(g["implied_p1_no_vig"].mean()),
        })
    return rows


def segment_error(df: pd.DataFrame, prob_col: str, segment_col: str, min_rows: int = 80) -> list[dict]:
    rows = []
    for seg, g in df.groupby(segment_col, dropna=False):
        if len(g) < min_rows:
            continue
        mm = metrics_for(g, prob_col)
        rows.append({
            "segment_col": segment_col,
            "segment": str(seg),
            **mm,
            "model_minus_market_log_loss": None if mm.get("log_loss") is None else float(mm["log_loss"] - mm["market_log_loss"]),
            "model_minus_market_brier": None if mm.get("brier") is None else float(mm["brier"] - mm["market_brier"]),
        })
    return sorted(rows, key=lambda r: (r.get("model_minus_market_log_loss") is None, r.get("model_minus_market_log_loss", 999)), reverse=True)


def run(years: list[int], test_years: list[int], min_train: int, min_test: int, force_download: bool = False) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_odds_data(years, force_download=force_download)
    data = add_buckets(make_side_dataset(raw, seed=42))
    data = data.dropna(subset=["result", "p1_odds", "p2_odds", "implied_p1_no_vig"])
    numeric, categorical = feature_columns()
    features = numeric + categorical
    specs: dict[str, str | None] = {
        "global": None,
        "surface_specialist": "surface_bucket",
        "grand_slam_specialist": "slam_bucket",
        "top10_specialist": "top10_bucket",
        "surface_x_top10_specialist": "surface_x_top10",
        "surface_x_slam_specialist": "surface_x_slam",
    }
    all_tests = []
    yearly_rows = []
    fit_logs: dict[str, list[dict]] = {k: [] for k in specs}
    for year in test_years:
        train = data[data["date"].dt.year < year].copy()
        test = data[data["date"].dt.year == year].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        year_test = test.copy()
        year_test["global_p1"] = fit_predict_global(train, test, features)
        for spec_name, bucket_col in specs.items():
            if spec_name == "global":
                year_test[f"{spec_name}_p1"] = year_test["global_p1"]
                continue
            probs, logs = fit_predict_specialist(train, test, features, str(bucket_col), min_train=min_train, min_test=min_test)
            year_test[f"{spec_name}_p1"] = probs
            for log in logs:
                fit_logs[spec_name].append({"year": year, "bucket_col": bucket_col, **log})
        for spec_name in specs:
            col = f"{spec_name}_p1"
            m = metrics_for(year_test, col)
            yearly_rows.append({"year": year, "model": spec_name, **m})
        all_tests.append(year_test)
    preds = pd.concat(all_tests, ignore_index=True)

    overall = []
    for spec_name in specs:
        col = f"{spec_name}_p1"
        m = metrics_for(preds, col)
        betting = [candidate_betting(preds, col, t) for t in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]]
        overall.append({
            "model": spec_name,
            **m,
            "log_loss_delta_vs_global": float(m["log_loss"] - metrics_for(preds, "global_p1")["log_loss"]),
            "brier_delta_vs_global": float(m["brier"] - metrics_for(preds, "global_p1")["brier"]),
            "best_betting_by_profit": max(betting, key=lambda x: x["profit"]),
            "betting_thresholds": betting,
        })

    # Per-segment comparison for the best/most requested specialist families.
    segment_comparisons = []
    for seg_col in ["surface_bucket", "slam_bucket", "top10_bucket", "surface_x_top10", "surface_x_slam"]:
        for seg, g in preds.groupby(seg_col, dropna=False):
            if len(g) < 80:
                continue
            base = metrics_for(g, "global_p1")
            row = {"segment_col": seg_col, "segment": str(seg), "rows": int(len(g)), "global_log_loss": base["log_loss"], "market_log_loss": base["market_log_loss"]}
            for spec_name in specs:
                col = f"{spec_name}_p1"
                mm = metrics_for(g, col)
                row[f"{spec_name}_log_loss"] = mm["log_loss"]
                row[f"{spec_name}_delta_vs_global"] = float(mm["log_loss"] - base["log_loss"])
            segment_comparisons.append(row)

    # Where the global model is going wrong.
    error_segments = []
    for seg_col in ["surface_bucket", "slam_bucket", "top10_bucket", "series", "round", "surface_x_top10", "surface_x_slam"]:
        error_segments.extend(segment_error(preds, "global_p1", seg_col, min_rows=80))
    error_segments = sorted(error_segments, key=lambda r: r["model_minus_market_log_loss"], reverse=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "years": years,
        "test_years": test_years,
        "min_train_per_specialist_bucket": min_train,
        "min_test_per_specialist_bucket": min_test,
        "features": features,
        "overall_model_comparison": sorted(overall, key=lambda r: r["log_loss"]),
        "yearly_model_comparison": yearly_rows,
        "segment_comparisons": sorted(segment_comparisons, key=lambda r: min([v for k, v in r.items() if k.endswith("_delta_vs_global")]), reverse=False),
        "global_calibration_bins": calibration_bins(preds, "global_p1"),
        "where_global_model_underperforms_market": error_segments[:30],
        "fit_logs": fit_logs,
        "disclaimer": "Research only; no betting execution. Specialist positives are hypotheses until they survive forward paper tracking and CLV checks.",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pred_path = OUT_DIR / f"specialist_predictions_{stamp}.csv"
    report_path = OUT_DIR / f"specialist_model_research_{stamp}.json"
    latest_pred = OUT_DIR / "latest_specialist_predictions.csv"
    latest_report = OUT_DIR / "latest_specialist_model_research.json"
    preds.to_csv(pred_path, index=False)
    preds.to_csv(latest_pred, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2027)))
    p.add_argument("--test-years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    p.add_argument("--min-train", type=int, default=450)
    p.add_argument("--min-test", type=int, default=40)
    p.add_argument("--force-download", action="store_true")
    args = p.parse_args()
    report = run(args.years, args.test_years, args.min_train, args.min_test, force_download=args.force_download)
    print(json.dumps({
        "generated_at": report["generated_at"],
        "overall_model_comparison": report["overall_model_comparison"],
        "top_underperformance_segments": report["where_global_model_underperforms_market"][:10],
        "disclaimer": report["disclaimer"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
