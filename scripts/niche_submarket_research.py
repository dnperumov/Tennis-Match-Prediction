#!/usr/bin/env python3
"""Walk-forward niche submarket research for tennis betting model.

Research only. No bet execution. This script tests whether model-vs-market edge
survives across years and submarkets, instead of cherry-picking one profitable
slice from one season.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from betting_research_pipeline import build_model, feature_columns, load_odds_data, make_side_dataset  # noqa: E402

OUT_DIR = ROOT / "data" / "betting_research"


def odds_bucket(odds: float) -> str:
    if pd.isna(odds):
        return "unknown"
    if odds < 1.25:
        return "<1.25 huge favorite"
    if odds < 1.50:
        return "1.25-1.50 favorite"
    if odds < 1.80:
        return "1.50-1.80 favorite"
    if odds < 2.20:
        return "1.80-2.20 near coinflip"
    if odds < 3.00:
        return "2.20-3.00 dog"
    if odds < 5.00:
        return "3.00-5.00 long dog"
    return ">=5.00 very long dog"


def round_group(r: str) -> str:
    r = str(r)
    if r in {"1st Round", "Round Robin", "R128", "R64", "R32"}:
        return "early"
    if r in {"2nd Round", "3rd Round", "4th Round"}:
        return "middle"
    if r in {"Quarterfinals", "Semifinals", "The Final", "QF", "SF", "F"}:
        return "late"
    return r


def signed_bucket(x: float, cuts: list[float], labels: list[str]) -> str:
    if pd.isna(x):
        return "unknown"
    for cut, label in zip(cuts, labels):
        if x < cut:
            return label
    return labels[-1]


def rank_gap_bucket(rank_diff: float) -> str:
    # Lower tennis rank is better. Negative means candidate is better-ranked.
    return signed_bucket(
        rank_diff,
        [-100, -50, -20, 20, 50, 100],
        ["candidate 100+ ranks better", "candidate 50-100 better", "candidate 20-50 better", "similar +/-20", "candidate 20-50 worse", "candidate 50-100 worse", "candidate 100+ worse"],
    )


def prior_matches_bucket(n: float) -> str:
    if pd.isna(n):
        return "unknown"
    if n < 5:
        return "0-4 prior matches"
    if n < 20:
        return "5-19 prior matches"
    if n < 50:
        return "20-49 prior matches"
    return "50+ prior matches"


def form_bucket(x: float) -> str:
    return signed_bucket(x, [-0.4, -0.2, 0.2, 0.4], ["form much worse", "form worse", "form similar", "form better", "form much better"])


def h2h_bucket(x: float) -> str:
    if pd.isna(x) or x == 0:
        return "no/even prior h2h"
    if x > 0:
        return "candidate leads h2h"
    return "candidate trails h2h"


def add_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_p2"] = 1.0 - df["model_p1"]
    df["implied_p2_no_vig"] = 1.0 - df["implied_p1_no_vig"]
    df["edge_p1"] = df["model_p1"] - df["implied_p1_no_vig"]
    df["edge_p2"] = df["model_p2"] - df["implied_p2_no_vig"]
    choose_p1 = df["edge_p1"] >= df["edge_p2"]
    df["candidate_side"] = np.where(choose_p1, "p1", "p2")
    df["candidate_player"] = np.where(choose_p1, df["player1"], df["player2"])
    df["candidate_odds"] = np.where(choose_p1, df["p1_odds"], df["p2_odds"])
    df["candidate_prob"] = np.where(choose_p1, df["model_p1"], df["model_p2"])
    df["candidate_implied"] = np.where(choose_p1, df["implied_p1_no_vig"], df["implied_p2_no_vig"])
    df["candidate_edge"] = df["candidate_prob"] - df["candidate_implied"]
    df["candidate_result"] = np.where(choose_p1, df["result"], 1 - df["result"])
    df["candidate_profit"] = np.where(df["candidate_result"] == 1, df["candidate_odds"] - 1.0, -1.0)
    df["candidate_odds_bucket"] = df["candidate_odds"].map(odds_bucket)
    df["candidate_favorite_status"] = np.where(df["candidate_odds"] < 2.0, "favorite/short", "underdog/plus")
    df["candidate_rank_diff"] = np.where(choose_p1, df["rank_diff"], -df["rank_diff"])
    df["candidate_prior_matches"] = np.where(choose_p1, df["p1_prior_matches"], df["p2_prior_matches"])
    df["candidate_last5_diff"] = np.where(choose_p1, df["last5_diff"], -df["last5_diff"])
    df["candidate_surface_pct_diff"] = np.where(choose_p1, df["surface_pct_diff"], -df["surface_pct_diff"])
    df["candidate_h2h_diff"] = np.where(choose_p1, df["h2h_diff"], -df["h2h_diff"])
    df["candidate_rank_bucket"] = df["candidate_rank_diff"].map(rank_gap_bucket)
    df["candidate_experience_bucket"] = pd.Series(df["candidate_prior_matches"]).map(prior_matches_bucket)
    df["candidate_form_bucket"] = pd.Series(df["candidate_last5_diff"]).map(form_bucket)
    df["candidate_surface_form_bucket"] = pd.Series(df["candidate_surface_pct_diff"]).map(form_bucket)
    df["candidate_h2h_bucket"] = pd.Series(df["candidate_h2h_diff"]).map(h2h_bucket)
    df["round_group"] = df["round"].map(round_group)
    df["year"] = df["date"].dt.year
    return df


def fit_predict_walkforward(years: list[int], test_years: list[int], force_download: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    raw = load_odds_data(years, force_download=force_download)
    data = make_side_dataset(raw, seed=42)
    data = data.dropna(subset=["result", "p1_odds", "p2_odds", "implied_p1_no_vig"])
    numeric, categorical = feature_columns()
    features = numeric + categorical
    all_preds = []
    yearly = []
    for year in test_years:
        train = data[data["date"].dt.year < year].copy()
        test = data[data["date"].dt.year == year].copy()
        if len(train) < 1000 or len(test) < 100:
            yearly.append({"year": year, "skipped": True, "train_rows": len(train), "test_rows": len(test)})
            continue
        model = build_model()
        model.fit(train[features], train["result"].astype(int))
        test["model_p1"] = model.predict_proba(test[features])[:, 1]
        pred = (test["model_p1"] >= 0.5).astype(int)
        yearly.append({
            "year": year,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "accuracy": float(accuracy_score(test["result"], pred)),
            "roc_auc": float(roc_auc_score(test["result"], test["model_p1"])),
            "log_loss": float(log_loss(test["result"], test["model_p1"])),
            "market_log_loss": float(log_loss(test["result"], test["implied_p1_no_vig"])),
            "brier": float(brier_score_loss(test["result"], test["model_p1"])),
            "market_brier": float(brier_score_loss(test["result"], test["implied_p1_no_vig"])),
        })
        all_preds.append(test)
    if not all_preds:
        raise RuntimeError("No walk-forward test years produced predictions")
    preds = add_candidate_columns(pd.concat(all_preds, ignore_index=True))
    return preds, yearly


def eval_slice(df: pd.DataFrame, threshold: float) -> dict:
    bets = df[df["candidate_edge"] >= threshold].copy()
    if bets.empty:
        return {"bets": 0, "profit": 0.0, "roi": 0.0, "hit_rate": None, "avg_edge": None, "years": 0, "min_year_bets": 0, "profit_t": None}
    profit = bets["candidate_profit"]
    by_year = bets.groupby("year").size()
    return {
        "bets": int(len(bets)),
        "profit": float(profit.sum()),
        "roi": float(profit.mean()),
        "hit_rate": float(bets["candidate_result"].mean()),
        "avg_edge": float(bets["candidate_edge"].mean()),
        "avg_odds": float(bets["candidate_odds"].mean()),
        "years": int(by_year.size),
        "min_year_bets": int(by_year.min()),
        "profit_t": float(profit.mean() / (profit.std(ddof=1) / np.sqrt(len(profit)))) if len(profit) > 1 and profit.std(ddof=1) > 0 else None,
    }


def scan_segments(preds: pd.DataFrame, min_bets: int = 150) -> pd.DataFrame:
    segment_defs = [
        ("all", lambda d: pd.Series("all", index=d.index)),
        ("surface", lambda d: d["surface"].fillna("unknown").astype(str)),
        ("series", lambda d: d["series"].fillna("unknown").astype(str)),
        ("round_group", lambda d: d["round_group"].fillna("unknown").astype(str)),
        ("odds_bucket", lambda d: d["candidate_odds_bucket"].fillna("unknown").astype(str)),
        ("favorite_status", lambda d: d["candidate_favorite_status"].fillna("unknown").astype(str)),
        ("rank_bucket", lambda d: d["candidate_rank_bucket"].fillna("unknown").astype(str)),
        ("experience_bucket", lambda d: d["candidate_experience_bucket"].fillna("unknown").astype(str)),
        ("form_bucket", lambda d: d["candidate_form_bucket"].fillna("unknown").astype(str)),
        ("surface_form_bucket", lambda d: d["candidate_surface_form_bucket"].fillna("unknown").astype(str)),
        ("h2h_bucket", lambda d: d["candidate_h2h_bucket"].fillna("unknown").astype(str)),
        ("surface_x_odds", lambda d: d["surface"].astype(str) + " | " + d["candidate_odds_bucket"].astype(str)),
        ("surface_x_round", lambda d: d["surface"].astype(str) + " | " + d["round_group"].astype(str)),
        ("series_x_odds", lambda d: d["series"].astype(str) + " | " + d["candidate_odds_bucket"].astype(str)),
        ("rank_x_odds", lambda d: d["candidate_rank_bucket"].astype(str) + " | " + d["candidate_odds_bucket"].astype(str)),
        ("form_x_odds", lambda d: d["candidate_form_bucket"].astype(str) + " | " + d["candidate_odds_bucket"].astype(str)),
        ("surface_form_x_odds", lambda d: d["candidate_surface_form_bucket"].astype(str) + " | " + d["candidate_odds_bucket"].astype(str)),
        ("h2h_x_odds", lambda d: d["candidate_h2h_bucket"].astype(str) + " | " + d["candidate_odds_bucket"].astype(str)),
    ]
    thresholds = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    rows = []
    for seg_name, seg_func in segment_defs:
        labels = seg_func(preds)
        for label in sorted(labels.dropna().unique()):
            sdf = preds[labels == label]
            for t in thresholds:
                res = eval_slice(sdf, t)
                if res["bets"] >= min_bets:
                    rows.append({"segment_type": seg_name, "segment": str(label), "threshold": t, **res})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["robust_flag"] = (out["roi"] > 0) & (out["years"] >= 3) & (out["min_year_bets"] >= 20) & (out["profit_t"].fillna(0) >= 1.0)
    return out.sort_values(["robust_flag", "roi", "profit"], ascending=[False, False, False])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2026)))
    p.add_argument("--test-years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    p.add_argument("--min-bets", type=int, default=150)
    p.add_argument("--force-download", action="store_true")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds, yearly = fit_predict_walkforward(args.years, args.test_years, force_download=args.force_download)
    segments = scan_segments(preds, min_bets=args.min_bets)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pred_path = OUT_DIR / f"walkforward_predictions_{stamp}.csv"
    seg_path = OUT_DIR / f"niche_segments_{stamp}.csv"
    latest_pred_path = OUT_DIR / "latest_walkforward_predictions.csv"
    latest_seg_path = OUT_DIR / "latest_niche_segments.csv"
    report_path = OUT_DIR / "latest_niche_research.json"
    preds.to_csv(pred_path, index=False)
    preds.to_csv(latest_pred_path, index=False)
    segments.to_csv(seg_path, index=False)
    segments.to_csv(latest_seg_path, index=False)
    all_results = [eval_slice(preds, t) | {"threshold": t} for t in [0, .02, .04, .06, .08, .10, .12]]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "years": args.years,
        "test_years": args.test_years,
        "yearly_metrics": yearly,
        "overall_threshold_results": all_results,
        "top_segments": segments.head(25).to_dict(orient="records") if not segments.empty else [],
        "robust_positive_segments": segments[segments["robust_flag"]].to_dict(orient="records") if not segments.empty else [],
        "disclaimer": "Research only; multiple-comparison risk is high. No autonomous betting or financial execution.",
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
