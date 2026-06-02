#!/usr/bin/env python3
"""Research-only tennis betting model/backtest pipeline.

This script is for offline analysis. It does not place bets, connect to sportsbooks,
or execute financial transactions.

It uses tennis-data.co.uk result/odds files because they include historical closing
odds. Features are restricted to information available before the match: rank,
points, surface/series/round, and rolling player form/H2H computed strictly from
prior rows.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "odds_raw"
OUT_DIR = ROOT / "data" / "betting_research"
MODEL_DIR = ROOT / "models" / "betting_research"

TENNIS_DATA_URL = "http://www.tennis-data.co.uk/{year}/{year}.xlsx"


def american_safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def download_tennis_data(years: Iterable[int], force: bool = False) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in years:
        path = RAW_DIR / f"tennis_data_atp_{year}.xlsx"
        if path.exists() and not force:
            paths.append(path)
            continue
        url = TENNIS_DATA_URL.format(year=year)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 tennis-research/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        path.write_bytes(data)
        paths.append(path)
    return paths


@dataclass
class PlayerState:
    wins: int = 0
    matches: int = 0
    last5: list[int] | None = None
    surface_wins: dict | None = None
    surface_matches: dict | None = None

    def __post_init__(self):
        self.last5 = [] if self.last5 is None else self.last5
        self.surface_wins = {} if self.surface_wins is None else self.surface_wins
        self.surface_matches = {} if self.surface_matches is None else self.surface_matches

    def win_pct(self) -> float:
        return self.wins / self.matches if self.matches else 0.5

    def last5_pct(self) -> float:
        return sum(self.last5) / len(self.last5) if self.last5 else 0.5

    def surface_pct(self, surface: str) -> float:
        n = self.surface_matches.get(surface, 0)
        return self.surface_wins.get(surface, 0) / n if n else 0.5

    def update(self, won: int, surface: str):
        self.matches += 1
        self.wins += int(won)
        self.last5.append(int(won))
        if len(self.last5) > 5:
            self.last5.pop(0)
        self.surface_matches[surface] = self.surface_matches.get(surface, 0) + 1
        self.surface_wins[surface] = self.surface_wins.get(surface, 0) + int(won)


def load_odds_data(years: Iterable[int], force_download: bool = False) -> pd.DataFrame:
    paths = download_tennis_data(years, force=force_download)
    frames = []
    for p in paths:
        df = pd.read_excel(p)
        df["source_file"] = str(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df.get("Comment", "Completed").fillna("Completed").astype(str).str.contains("Completed|Retired|Walkover|Awarded", case=False, regex=True)]
    required = ["Date", "Winner", "Loser", "WRank", "LRank", "WPts", "LPts", "B365W", "B365L", "Surface", "Series", "Round"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["WRank", "LRank", "WPts", "LPts", "B365W", "B365L"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Winner", "Loser", "B365W", "B365L"])
    df = df[(df["B365W"] > 1.0) & (df["B365L"] > 1.0)]
    df = df.sort_values(["Date", "Tournament", "Round", "Winner", "Loser"]).reset_index(drop=True)
    return df


def make_side_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states: dict[str, PlayerState] = {}
    h2h: dict[tuple[str, str], list[int]] = {}
    rows = []

    def st(name: str) -> PlayerState:
        if name not in states:
            states[name] = PlayerState()
        return states[name]

    # Process one date at a time. Tennis-data rows do not include match start
    # time, so the strictest defensible pre-match simulation is: create every
    # feature row for date D from state ending before date D, then update states
    # with all results from D. This prevents accidental leakage from another
    # same-day match into a later row on the same date.
    for _, day in df.groupby("Date", sort=True):
        pending_updates: list[tuple[str, str, str]] = []
        for _, m in day.iterrows():
            winner = str(m["Winner"])
            loser = str(m["Loser"])
            surface = str(m.get("Surface", "Unknown"))
            swap = bool(rng.random() < 0.5)
            p1, p2 = (winner, loser) if not swap else (loser, winner)
            p1_is_winner = int(p1 == winner)
            p1_odds = float(m["B365W"] if p1 == winner else m["B365L"])
            p2_odds = float(m["B365L"] if p1 == winner else m["B365W"])
            p1_rank = american_safe_float(m["WRank"] if p1 == winner else m["LRank"])
            p2_rank = american_safe_float(m["LRank"] if p1 == winner else m["WRank"])
            p1_pts = american_safe_float(m["WPts"] if p1 == winner else m["LPts"])
            p2_pts = american_safe_float(m["LPts"] if p1 == winner else m["WPts"])
            s1, s2 = st(p1), st(p2)
            a, b = sorted([p1, p2])
            key12 = (a, b)
            prior_h = h2h.get(key12, [0, 0])  # [sorted[0] wins, sorted[1] wins]
            if key12[0] == p1:
                h2h_p1, h2h_p2 = prior_h[0], prior_h[1]
            else:
                h2h_p1, h2h_p2 = prior_h[1], prior_h[0]
            imp1_raw = 1.0 / p1_odds
            imp2_raw = 1.0 / p2_odds
            overround = imp1_raw + imp2_raw
            implied_p1_no_vig = imp1_raw / overround if overround else np.nan
            rows.append({
                "date": m["Date"],
                "tournament": m.get("Tournament"),
                "surface": surface,
                "series": m.get("Series"),
                "round": m.get("Round"),
                "player1": p1,
                "player2": p2,
                "p1_odds": p1_odds,
                "p2_odds": p2_odds,
                "implied_p1_no_vig": implied_p1_no_vig,
                "result": p1_is_winner,
                "rank_diff": p1_rank - p2_rank,
                "p1_rank": p1_rank,
                "p2_rank": p2_rank,
                "p1_pts": p1_pts,
                "p2_pts": p2_pts,
                "p1_top10": int(not pd.isna(p1_rank) and p1_rank <= 10),
                "p2_top10": int(not pd.isna(p2_rank) and p2_rank <= 10),
                "any_top10": int((not pd.isna(p1_rank) and p1_rank <= 10) or (not pd.isna(p2_rank) and p2_rank <= 10)),
                "both_top10": int((not pd.isna(p1_rank) and p1_rank <= 10) and (not pd.isna(p2_rank) and p2_rank <= 10)),
                "log_rank_diff": math.log1p(abs(p1_rank - p2_rank)) if not pd.isna(p1_rank) and not pd.isna(p2_rank) else np.nan,
                "points_ratio": math.log1p(p1_pts) - math.log1p(p2_pts) if not pd.isna(p1_pts) and not pd.isna(p2_pts) else np.nan,
                "p1_prior_win_pct": s1.win_pct(),
                "p2_prior_win_pct": s2.win_pct(),
                "prior_win_pct_diff": s1.win_pct() - s2.win_pct(),
                "p1_last5_pct": s1.last5_pct(),
                "p2_last5_pct": s2.last5_pct(),
                "last5_diff": s1.last5_pct() - s2.last5_pct(),
                "p1_surface_pct": s1.surface_pct(surface),
                "p2_surface_pct": s2.surface_pct(surface),
                "surface_pct_diff": s1.surface_pct(surface) - s2.surface_pct(surface),
                "h2h_diff": h2h_p1 - h2h_p2,
                "p1_prior_matches": s1.matches,
                "p2_prior_matches": s2.matches,
            })
            pending_updates.append((winner, loser, surface))
        # Update only after every feature row for this date has been captured.
        for winner, loser, surface in pending_updates:
            st(winner).update(1, surface)
            st(loser).update(0, surface)
            a, b = sorted([winner, loser])
            sorted_key = (a, b)
            h = h2h.setdefault(sorted_key, [0, 0])
            if winner == sorted_key[0]:
                h[0] += 1
            else:
                h[1] += 1
    return pd.DataFrame(rows)


def feature_columns() -> tuple[list[str], list[str]]:
    numeric = [
        "rank_diff", "log_rank_diff", "points_ratio", "p1_top10", "p2_top10", "any_top10", "both_top10",
        "prior_win_pct_diff", "last5_diff", "surface_pct_diff", "h2h_diff", "p1_prior_matches", "p2_prior_matches",
        "p1_prior_win_pct", "p2_prior_win_pct", "p1_last5_pct", "p2_last5_pct",
        "p1_surface_pct", "p2_surface_pct",
    ]
    categorical = ["surface", "series", "round"]
    return numeric, categorical


def build_model() -> Pipeline:
    numeric, categorical = feature_columns()
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    base = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
            ("hgb", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, l2_regularization=0.05, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=8, random_state=42, n_jobs=-1)),
        ],
        voting="soft",
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    return Pipeline([("features", pre), ("model", calibrated)])


def evaluate_betting(test: pd.DataFrame, prob_col: str, threshold: float, stake: float = 1.0) -> dict:
    bets = test[test[prob_col] - test["implied_p1_no_vig"] >= threshold].copy()
    if bets.empty:
        return {"threshold": threshold, "bets": 0, "profit": 0.0, "roi": 0.0, "hit_rate": None, "avg_edge": None}
    bets["profit"] = np.where(bets["result"] == 1, (bets["p1_odds"] - 1.0) * stake, -stake)
    return {
        "threshold": threshold,
        "bets": int(len(bets)),
        "profit": float(bets["profit"].sum()),
        "roi": float(bets["profit"].sum() / (len(bets) * stake)),
        "hit_rate": float(bets["result"].mean()),
        "avg_edge": float((bets[prob_col] - bets["implied_p1_no_vig"]).mean()),
        "avg_odds": float(bets["p1_odds"].mean()),
    }


def train_and_backtest(years: list[int], test_year: int, force_download: bool = False) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_odds_data(years, force_download=force_download)
    data = make_side_dataset(raw)
    data = data.dropna(subset=["result", "p1_odds", "implied_p1_no_vig"])
    train = data[data["date"].dt.year < test_year].copy()
    test = data[data["date"].dt.year == test_year].copy()
    if len(train) < 1000 or len(test) < 100:
        raise RuntimeError(f"Not enough train/test rows: train={len(train)} test={len(test)}")
    numeric, categorical = feature_columns()
    features = numeric + categorical
    model = build_model()
    model.fit(train[features], train["result"].astype(int))
    test["model_p1"] = model.predict_proba(test[features])[:, 1]
    pred = (test["model_p1"] >= 0.5).astype(int)
    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_year": int(test_year),
        "accuracy": float(accuracy_score(test["result"], pred)),
        "roc_auc": float(roc_auc_score(test["result"], test["model_p1"])),
        "log_loss": float(log_loss(test["result"], test["model_p1"])),
        "brier": float(brier_score_loss(test["result"], test["model_p1"])),
        "market_brier": float(brier_score_loss(test["result"], test["implied_p1_no_vig"])),
        "market_log_loss": float(log_loss(test["result"], test["implied_p1_no_vig"])),
    }
    strategies = [evaluate_betting(test, "model_p1", t) for t in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]]
    best = max(strategies, key=lambda x: x["profit"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pred_path = OUT_DIR / f"backtest_predictions_{test_year}_{stamp}.csv"
    latest_pred_path = OUT_DIR / "latest_backtest_predictions.csv"
    metrics_path = OUT_DIR / f"metrics_{test_year}_{stamp}.json"
    latest_metrics_path = OUT_DIR / "latest_metrics.json"
    test.to_csv(pred_path, index=False)
    test.to_csv(latest_pred_path, index=False)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics, "strategies": strategies, "best_strategy_by_profit": best, "disclaimer": "Research only; no autonomous betting or financial execution."}
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Keep the evaluated model outputs above for honest backtesting, but save a
    # separate production model fit on every available row so daily retrains can
    # learn from the newest completed matches after the backtest window.
    production_model = build_model()
    production_model.fit(data[features], data["result"].astype(int))
    with open(MODEL_DIR / "latest_model.pkl", "wb") as f:
        pickle.dump({
            "model": production_model,
            "features": features,
            "trained_at": payload["generated_at"],
            "metrics": metrics,
            "evaluation_model_note": f"Backtest metrics are from training before {test_year}; saved model is refit on all {len(data)} rows.",
        }, f)
    payload["production_train_rows"] = int(len(data))
    latest_metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description="Research-only tennis betting model/backtester")
    p.add_argument("--years", nargs="+", type=int, default=list(range(2021, 2027)))
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--force-download", action="store_true")
    args = p.parse_args()
    payload = train_and_backtest(args.years, args.test_year, force_download=args.force_download)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
