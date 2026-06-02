#!/usr/bin/env python3
"""Track no-lookahead model picks for this year's French Open.

Research-only. This script does not place bets, connect to sportsbooks, or execute
financial transactions. It records model-vs-result accuracy for completed Roland
Garros matches in the local enriched match export.

Current tracker model: a transparent Elo-style baseline using only matches before
Roland Garros match date. Matches on the same date are predicted from the state
ending before that date, then the state is updated after all same-day predictions
are emitted to avoid same-day leakage.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = ROOT / "data" / "exports" / "atp_matches_enriched_current.csv"
OUT_DIR = ROOT / "data" / "french_open_tracking"
PICKS_CSV = OUT_DIR / "roland_garros_2026_model_picks.csv"
METRICS_JSON = OUT_DIR / "roland_garros_2026_accuracy.json"
REPORT_MD = OUT_DIR / "roland_garros_2026_report.md"

K_OVERALL = 24.0
K_SURFACE = 18.0
START_ELO = 1500.0
SURFACE_WEIGHT = 0.45


def norm_name(x: object) -> str:
    return str(x).strip()


def logistic_prob(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / 400.0))


@dataclass
class EloState:
    overall: dict[str, float] = field(default_factory=dict)
    surface: dict[tuple[str, str], float] = field(default_factory=dict)

    def overall_rating(self, player: str) -> float:
        return self.overall.get(player, START_ELO)

    def surface_rating(self, player: str, surface: str) -> float:
        return self.surface.get((player, surface), START_ELO)

    def blended_rating(self, player: str, surface: str) -> float:
        return (1.0 - SURFACE_WEIGHT) * self.overall_rating(player) + SURFACE_WEIGHT * self.surface_rating(player, surface)

    def update(self, winner: str, loser: str, surface: str) -> None:
        winner = norm_name(winner)
        loser = norm_name(loser)
        surface = str(surface or "Unknown")

        rw = self.overall_rating(winner)
        rl = self.overall_rating(loser)
        pw = logistic_prob(rw, rl)
        self.overall[winner] = rw + K_OVERALL * (1.0 - pw)
        self.overall[loser] = rl + K_OVERALL * (0.0 - (1.0 - pw))

        sw = self.surface_rating(winner, surface)
        sl = self.surface_rating(loser, surface)
        psw = logistic_prob(sw, sl)
        self.surface[(winner, surface)] = sw + K_SURFACE * (1.0 - psw)
        self.surface[(loser, surface)] = sl + K_SURFACE * (0.0 - (1.0 - psw))


def load_matches(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["match_date"])
    required = ["match_date", "tourney_name", "surface", "round", "winner_name", "loser_name", "score", "match_key"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in {path}: {missing}")
    df = df.dropna(subset=["match_date", "winner_name", "loser_name"])
    df["tourney_name_norm"] = df["tourney_name"].astype(str).str.lower()
    df = df.sort_values(["match_date", "tourney_name", "round", "winner_name", "loser_name", "match_key"]).reset_index(drop=True)
    return df


def is_rg_2026(df: pd.DataFrame) -> pd.Series:
    return (
        (df["match_date"].dt.year == 2026)
        & df["tourney_name"].astype(str).str.contains("Roland Garros|French Open", case=False, regex=True, na=False)
    )


def build_tracking(df: pd.DataFrame) -> pd.DataFrame:
    state = EloState()
    picks: list[dict] = []

    for day_value, day in df.groupby(df["match_date"].dt.date, sort=True):
        # Emit all picks for the date before applying same-day updates.
        for _, row in day.iterrows():
            winner = norm_name(row["winner_name"])
            loser = norm_name(row["loser_name"])
            surface = str(row.get("surface") or "Unknown")
            r_w = state.blended_rating(winner, surface)
            r_l = state.blended_rating(loser, surface)
            p_w = logistic_prob(r_w, r_l)

            # Stable side orientation: alphabetic, so the row is not always winner-first.
            p1, p2 = sorted([winner, loser])
            p1_rating = state.blended_rating(p1, surface)
            p2_rating = state.blended_rating(p2, surface)
            p1_prob = logistic_prob(p1_rating, p2_rating)
            picked = p1 if p1_prob >= 0.5 else p2
            picked_prob = p1_prob if picked == p1 else 1.0 - p1_prob
            actual = winner

            if str(row.get("tourney_name", "")).lower().find("roland garros") >= 0 or str(row.get("tourney_name", "")).lower().find("french open") >= 0:
                if pd.Timestamp(row["match_date"]).year == 2026:
                    picks.append({
                        "match_date": pd.Timestamp(row["match_date"]).date().isoformat(),
                        "tournament": row.get("tourney_name"),
                        "round": row.get("round"),
                        "surface": surface,
                        "player1": p1,
                        "player2": p2,
                        "model_pick": picked,
                        "model_pick_probability": round(float(picked_prob), 6),
                        "player1_probability": round(float(p1_prob), 6),
                        "actual_winner": actual,
                        "correct": int(picked == actual),
                        "score": row.get("score"),
                        "match_key": row.get("match_key"),
                        "model_version": "elo_blend_overall_surface_v1_no_same_day_leakage",
                        "rating_player1": round(float(p1_rating), 3),
                        "rating_player2": round(float(p2_rating), 3),
                    })

        # Now update state with all matches from this date.
        for _, row in day.iterrows():
            state.update(row["winner_name"], row["loser_name"], str(row.get("surface") or "Unknown"))

    return pd.DataFrame(picks)


def metrics(picks: pd.DataFrame) -> dict:
    if picks.empty:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": 0,
            "accuracy": None,
            "correct": 0,
            "incorrect": 0,
            "model_version": "elo_blend_overall_surface_v1_no_same_day_leakage",
            "disclaimer": "Research only; no betting or financial execution.",
        }
    by_round = []
    for rnd, g in picks.groupby("round", dropna=False):
        by_round.append({
            "round": str(rnd),
            "rows": int(len(g)),
            "correct": int(g["correct"].sum()),
            "accuracy": float(g["correct"].mean()),
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(picks)),
        "correct": int(picks["correct"].sum()),
        "incorrect": int(len(picks) - picks["correct"].sum()),
        "accuracy": float(picks["correct"].mean()),
        "by_round": sorted(by_round, key=lambda x: x["round"]),
        "model_version": "elo_blend_overall_surface_v1_no_same_day_leakage",
        "source_export": str(DEFAULT_EXPORT),
        "disclaimer": "Research only; no betting or financial execution.",
    }


def write_report(picks: pd.DataFrame, payload: dict) -> None:
    lines = [
        "# Roland Garros 2026 model pick tracker",
        "",
        "Research only. No betting or financial execution.",
        "",
        "## Current accuracy",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Model: `{payload['model_version']}`",
        f"- Completed tracked matches: `{payload['rows']}`",
        f"- Correct: `{payload['correct']}`",
        f"- Incorrect: `{payload['incorrect']}`",
        f"- Accuracy: `{payload['accuracy']:.4f}`" if payload.get("accuracy") is not None else "- Accuracy: `n/a`",
        "",
        "## Notes",
        "- Picks are emitted from pre-date Elo state, then same-day results are applied after all picks for that date. This avoids same-day leakage when match start times are unavailable.",
        "- This is a transparent current-tournament tracker. It should be replaced or compared against the advanced calibrated model once live 2026 French Open odds/prematch rows are available.",
    ]
    if not picks.empty:
        unique_dates = sorted(picks["match_date"].astype(str).unique())
        if len(unique_dates) == 1 and len(picks) > 20:
            lines.append(
                f"- Data caveat: current fallback source has all Roland Garros rows dated `{unique_dates[0]}`. "
                "So this tracker is effectively pre-tournament/prior-state Elo picks until better per-match dates are added."
            )
    lines.extend(["", "## Recent tracked picks"])
    if picks.empty:
        lines.append("No Roland Garros 2026 rows found yet.")
    else:
        round_order = {"F": 7, "SF": 6, "QF": 5, "R16": 4, "R32": 3, "R64": 2, "R128": 1}
        recent = picks.assign(_round_order=picks["round"].map(round_order).fillna(0)).sort_values(
            ["match_date", "_round_order", "player1", "player2"], ascending=[False, False, True, True]
        ).head(20)
        for _, r in recent.iterrows():
            mark = "✅" if int(r["correct"]) else "❌"
            lines.append(
                f"- {mark} {r['match_date']} {r['round']}: picked **{r['model_pick']}** "
                f"({r['model_pick_probability']:.3f}) vs {r['player1']}/{r['player2']}; winner **{r['actual_winner']}**; score {r.get('score','')}"
            )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Track Roland Garros 2026 model picks and accuracy.")
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_matches(args.export)
    picks = build_tracking(df)
    picks.to_csv(PICKS_CSV, index=False)
    payload = metrics(picks)
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(picks, payload)

    print(json.dumps({
        "picks_csv": str(PICKS_CSV),
        "metrics_json": str(METRICS_JSON),
        "report_md": str(REPORT_MD),
        "rows": payload["rows"],
        "correct": payload["correct"],
        "incorrect": payload["incorrect"],
        "accuracy": payload["accuracy"],
        "model_version": payload["model_version"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
