#!/usr/bin/env python3
"""Daily betting-research model retrain/backtest.

Research only: trains a probability model from historical ATP odds/results,
compares against closing/average market probabilities, and writes reproducible
metrics/artifacts. It does not connect to sportsbooks or place bets.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _odds_to_prob(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    pa = 1.0 / pd.to_numeric(a, errors="coerce")
    pb = 1.0 / pd.to_numeric(b, errors="coerce")
    s = pa + pb
    return pa / s, pb / s


def _load_odds(path: Path, years: list[int]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing odds file: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    df["year"] = df["Date"].dt.year.astype(int)
    df = df[df["year"].isin(years)].copy()
    df = df[df["Comment"].fillna("").astype(str).str.lower().eq("completed")].copy()
    # Prefer Avg odds for market/backtest; fall back to Max/B365/PS if Avg is missing.
    for side in ["W", "L"]:
        cols = [f"Avg{side}", f"Max{side}", f"B365{side}", f"PS{side}"]
        present = [c for c in cols if c in df.columns]
        df[f"odds_{side}"] = df[present].bfill(axis=1).iloc[:, 0]
    df = df[(df["odds_W"] > 1.0) & (df["odds_L"] > 1.0)].copy()
    return df.reset_index(drop=True)


def _make_sides(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    # Deterministic alternating orientation avoids always labeling the winner as side A.
    flip = (np.arange(len(m)) % 2) == 1
    p_mkt_w, p_mkt_l = _odds_to_prob(m["odds_W"], m["odds_L"])
    rows = pd.DataFrame({
        "date": m["Date"],
        "year": m["year"],
        "tournament": m.get("Tournament"),
        "series": m.get("Series"),
        "court": m.get("Court"),
        "surface": m.get("Surface"),
        "round": m.get("Round"),
        "best_of": pd.to_numeric(m.get("Best of"), errors="coerce"),
        "player_a": np.where(flip, m["Loser"], m["Winner"]),
        "player_b": np.where(flip, m["Winner"], m["Loser"]),
        "rank_a": np.where(flip, m.get("LRank"), m.get("WRank")),
        "rank_b": np.where(flip, m.get("WRank"), m.get("LRank")),
        "pts_a": np.where(flip, m.get("LPts"), m.get("WPts")),
        "pts_b": np.where(flip, m.get("WPts"), m.get("LPts")),
        "odds_a": np.where(flip, m["odds_L"], m["odds_W"]),
        "odds_b": np.where(flip, m["odds_W"], m["odds_L"]),
        "market_prob_a": np.where(flip, p_mkt_l, p_mkt_w),
        "target": np.where(flip, 0, 1),
    })
    rows["rank_a"] = pd.to_numeric(rows["rank_a"], errors="coerce")
    rows["rank_b"] = pd.to_numeric(rows["rank_b"], errors="coerce")
    rows["pts_a"] = pd.to_numeric(rows["pts_a"], errors="coerce")
    rows["pts_b"] = pd.to_numeric(rows["pts_b"], errors="coerce")
    rows["odds_a"] = pd.to_numeric(rows["odds_a"], errors="coerce")
    rows["odds_b"] = pd.to_numeric(rows["odds_b"], errors="coerce")
    rows["rank_diff"] = rows["rank_a"] - rows["rank_b"]
    rows["pts_diff"] = rows["pts_a"] - rows["pts_b"]
    rows["log_odds_ratio"] = np.log(rows["odds_a"]) - np.log(rows["odds_b"])
    rows = rows.replace([np.inf, -np.inf], np.nan)
    rows = rows[rows["market_prob_a"].between(0.001, 0.999) & rows["target"].notna()].copy()
    return rows.reset_index(drop=True)


def _build_model() -> Pipeline:
    numeric = ["best_of", "rank_a", "rank_b", "pts_a", "pts_b", "rank_diff", "pts_diff", "odds_a", "odds_b", "market_prob_a", "log_odds_ratio"]
    categorical = ["series", "court", "surface", "round"]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), categorical),
    ])
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.035, l2_regularization=0.05, random_state=42)
    return Pipeline([("preprocess", pre), ("model", clf)])


def _safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def _flat_stake(y: np.ndarray, model_p: np.ndarray, market_p: np.ndarray, odds: np.ndarray) -> dict:
    best = {"threshold": None, "bets": 0, "roi": None, "profit": 0.0}
    for th in np.round(np.arange(0.00, 0.201, 0.005), 3):
        mask = (model_p - market_p) >= th
        bets = int(mask.sum())
        if bets < 20:
            continue
        profit = np.where(y[mask] == 1, odds[mask] - 1.0, -1.0).sum()
        roi = profit / bets
        if best["roi"] is None or roi > best["roi"]:
            best = {"threshold": float(th), "bets": bets, "roi": float(roi), "profit": float(profit)}
    if best["threshold"] is None:
        mask = (model_p - market_p) >= 0
        bets = int(mask.sum())
        profit = float(np.where(y[mask] == 1, odds[mask] - 1.0, -1.0).sum()) if bets else 0.0
        best = {"threshold": 0.0, "bets": bets, "roi": (profit / bets if bets else None), "profit": profit}
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--test-year", type=int, required=True)
    ap.add_argument("--odds-file", default="data/odds/atp_odds_2013_2026.csv")
    args = ap.parse_args()

    root = Path.cwd()
    out_dir = root / "data" / "betting_research"
    model_dir = root / "models" / "betting_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    matches = _load_odds(root / args.odds_file, args.years)
    sides = _make_sides(matches)
    train = sides[sides["year"] < args.test_year].copy()
    test = sides[sides["year"] == args.test_year].copy()
    prod = sides[sides["year"] <= max(args.years)].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"Insufficient rows: train={len(train)} test={len(test)}")

    feature_cols = ["series", "court", "surface", "round", "best_of", "rank_a", "rank_b", "pts_a", "pts_b", "rank_diff", "pts_diff", "odds_a", "odds_b", "market_prob_a", "log_odds_ratio"]
    model = _build_model()
    model.fit(train[feature_cols], train["target"].astype(int))
    p = np.clip(model.predict_proba(test[feature_cols])[:, 1], 1e-6, 1 - 1e-6)
    y = test["target"].astype(int).to_numpy()
    mkt = np.clip(test["market_prob_a"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)

    flat = _flat_stake(y, p, mkt, test["odds_a"].to_numpy(dtype=float))
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "years": args.years,
        "test_year": args.test_year,
        "source_rows": int(len(matches)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "production_train_rows": int(len(prod)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": _safe_auc(y, p),
        "model_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "market_log_loss": float(log_loss(y, mkt, labels=[0, 1])),
        "model_brier": float(brier_score_loss(y, p)),
        "market_brier": float(brier_score_loss(y, mkt)),
        "best_flat_stake": flat,
        "note": "Research/backtest only; no bets placed and no sportsbook connections made.",
    }

    # Refit production artifact on all selected rows after test metrics are frozen.
    prod_model = _build_model()
    prod_model.fit(prod[feature_cols], prod["target"].astype(int))
    artifact = {"model": prod_model, "feature_cols": feature_cols, "metrics": metrics}
    joblib.dump(artifact, model_dir / "latest_model.pkl")
    with (out_dir / "latest_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    sides.to_csv(out_dir / "latest_modeling_rows.csv", index=False)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
