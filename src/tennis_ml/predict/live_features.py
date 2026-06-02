"""Live/current matchup feature construction.

The functions here are deterministic and no-lookahead by design: a feature
snapshot only uses rows strictly before the requested match date. They provide the
infrastructure for real schedules, current-tournament fatigue, rolling
serve/return stats, surface/BO5 Elo, entry context, and calibrated routing.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any
import math
import re

import pandas as pd

SLAM_NAMES = {"australian open", "roland garros", "french open", "wimbledon", "us open", "u.s. open"}
ENTRY_CODES = {
    "Q": "qualifier",
    "QUALIFIER": "qualifier",
    "WC": "wildcard",
    "WILD CARD": "wildcard",
    "WILDCARD": "wildcard",
    "LL": "lucky_loser",
    "LUCKY LOSER": "lucky_loser",
    "PR": "protected_ranking",
    "PROTECTED RANKING": "protected_ranking",
}


@dataclass
class PlayerState:
    elo: float = 1500.0
    surface_elo: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    bo5_elo: float = 1500.0
    matches: int = 0
    wins: int = 0
    surface_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    serve_first_won: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    return_points_won: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    total_points_won: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    last_dates: deque[pd.Timestamp] = field(default_factory=lambda: deque(maxlen=20))
    tournament_load: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: {"matches": 0, "minutes": 0.0, "sets": 0.0}))

    def recent_win_rate(self) -> float:
        if not self.recent_results:
            return 0.5
        return sum(self.recent_results) / len(self.recent_results)

    def mean(self, attr: str, default: float = 50.0) -> float:
        values = getattr(self, attr)
        if not values:
            return default
        clean = [float(v) for v in values if not pd.isna(v)]
        return sum(clean) / len(clean) if clean else default

    def days_since_last(self, match_date: pd.Timestamp | None) -> float | None:
        if match_date is None or pd.isna(match_date) or not self.last_dates:
            return None
        return float((match_date - max(self.last_dates)).days)

    def matches_last(self, match_date: pd.Timestamp | None, days: int) -> int:
        if match_date is None or pd.isna(match_date):
            return 0
        cutoff = match_date - pd.Timedelta(days=days)
        return sum(1 for d in self.last_dates if cutoff <= d < match_date)


def normalize_name(name: Any) -> str:
    return " ".join(str(name).lower().strip().split())


def normalize_tournament(name: Any) -> str:
    return " ".join(str(name).lower().strip().split())


def is_slam_context(context: dict[str, Any] | None) -> bool:
    ctx = context or {}
    tournament = normalize_tournament(ctx.get("tournament") or ctx.get("tourney_name") or "")
    return bool(ctx.get("best_of_5")) or tournament in SLAM_NAMES


def parse_entry_context(player1_entry: Any = None, player2_entry: Any = None) -> dict[str, int]:
    def flags(value: Any) -> dict[str, int]:
        raw = str(value or "").upper().strip()
        tokens = {raw}
        tokens.update(t.strip() for t in re.split(r"[,/;|()\s]+", raw) if t.strip())
        kinds = {ENTRY_CODES[t] for t in tokens if t in ENTRY_CODES}
        return {
            "qualifier": int("qualifier" in kinds),
            "wildcard": int("wildcard" in kinds),
            "lucky_loser": int("lucky_loser" in kinds),
            "protected_ranking": int("protected_ranking" in kinds),
        }

    p1 = flags(player1_entry)
    p2 = flags(player2_entry)
    out = {
        "p1_qualifier": p1["qualifier"],
        "p2_qualifier": p2["qualifier"],
        "qualifier_diff": p1["qualifier"] - p2["qualifier"],
        "p1_wildcard": p1["wildcard"],
        "p2_wildcard": p2["wildcard"],
        "wildcard_diff": p1["wildcard"] - p2["wildcard"],
        "p1_lucky_loser": p1["lucky_loser"],
        "p2_lucky_loser": p2["lucky_loser"],
        "lucky_loser_diff": p1["lucky_loser"] - p2["lucky_loser"],
        "p1_protected_ranking": p1["protected_ranking"],
        "p2_protected_ranking": p2["protected_ranking"],
        "protected_ranking_diff": p1["protected_ranking"] - p2["protected_ranking"],
    }
    return out


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def update_elo(ra: float, rb: float, score_a: float, k: float = 28.0) -> tuple[float, float]:
    ea = expected_score(ra, rb)
    delta = k * (score_a - ea)
    return ra + delta, rb - delta


def _sets_played(score: Any) -> int:
    text = str(score or "")
    if not text or text.lower() == "nan":
        return 0
    # Count set-like chunks such as 6-4, 7-6(5), 10-8. Ignore RET/W/O text.
    return len(re.findall(r"\b\d{1,2}-\d{1,2}(?:\([^)]*\))?", text))


def _side_stat(row: pd.Series, prefix: str, stat: str) -> float | None:
    col = f"{prefix}_{stat}"
    if col not in row or pd.isna(row[col]):
        return None
    return float(row[col])


def _ensure_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "match_date" in out.columns:
        out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    else:
        out["match_date"] = pd.NaT
    return out


def _chronological_prior_rows(df: pd.DataFrame, match_date: pd.Timestamp | None) -> pd.DataFrame:
    dated = _ensure_dates(df)
    if match_date is None or pd.isna(match_date):
        return dated.sort_values("match_date", na_position="first")
    return dated[dated["match_date"] < match_date].sort_values("match_date", na_position="first")


def _update_state_for_row(states: dict[str, PlayerState], row: pd.Series) -> None:
    winner = normalize_name(row.get("winner_name", ""))
    loser = normalize_name(row.get("loser_name", ""))
    if not winner or not loser:
        return
    surface = str(row.get("surface") or "Unknown")
    date = row.get("match_date")
    tournament = normalize_tournament(row.get("tourney_name", ""))
    is_bo5 = normalize_tournament(row.get("tourney_name", "")) in SLAM_NAMES or _sets_played(row.get("score")) > 3

    sw = states[winner]
    sl = states[loser]
    sw.elo, sl.elo = update_elo(sw.elo, sl.elo, 1.0)
    sw.surface_elo[surface], sl.surface_elo[surface] = update_elo(sw.surface_elo[surface], sl.surface_elo[surface], 1.0, k=30.0)
    if is_bo5:
        sw.bo5_elo, sl.bo5_elo = update_elo(sw.bo5_elo, sl.bo5_elo, 1.0, k=24.0)

    for state, won, prefix in [(sw, 1, "winner"), (sl, 0, "loser")]:
        state.matches += 1
        state.wins += won
        state.surface_matches[surface] += 1
        state.surface_wins[surface] += won
        state.recent_results.append(won)
        for stat, attr in [
            ("serve_first_won_pct", "serve_first_won"),
            ("return_points_won_pct", "return_points_won"),
            ("total_points_won_pct", "total_points_won"),
        ]:
            value = _side_stat(row, prefix, stat)
            if value is not None:
                getattr(state, attr).append(value)
        if date is not None and not pd.isna(date):
            state.last_dates.append(date)
        load = state.tournament_load[tournament]
        load["matches"] += 1
        load["minutes"] += float(row.get("minutes") or 0.0) if not pd.isna(row.get("minutes")) else 0.0
        load["sets"] += float(_sets_played(row.get("score")))


def build_player_states(df: pd.DataFrame, match_date: pd.Timestamp | str | None = None) -> dict[str, PlayerState]:
    target_date = pd.to_datetime(match_date, errors="coerce") if match_date is not None else None
    rows = _chronological_prior_rows(df, target_date)
    states: dict[str, PlayerState] = defaultdict(PlayerState)
    for _, row in rows.iterrows():
        _update_state_for_row(states, row)
    return states


def _state_snapshot(state: PlayerState, surface: str, tournament: str, match_date: pd.Timestamp | None) -> dict[str, float | None]:
    load = state.tournament_load[normalize_tournament(tournament)]
    return {
        "overall_elo": state.elo,
        "surface_elo": state.surface_elo[surface],
        "bo5_elo": state.bo5_elo,
        "matches": float(state.matches),
        "win_rate": state.wins / state.matches if state.matches else 0.5,
        "surface_matches": float(state.surface_matches[surface]),
        "surface_win_rate": state.surface_wins[surface] / state.surface_matches[surface] if state.surface_matches[surface] else 0.5,
        "recent_win_rate": state.recent_win_rate(),
        "first_won_pct": state.mean("serve_first_won"),
        "return_points_won_pct": state.mean("return_points_won"),
        "total_points_won_pct": state.mean("total_points_won"),
        "days_since_last": state.days_since_last(match_date),
        "matches_last_7": float(state.matches_last(match_date, 7)),
        "matches_last_14": float(state.matches_last(match_date, 14)),
        "current_tournament_matches": float(load["matches"]),
        "current_tournament_minutes": float(load["minutes"]),
        "current_tournament_sets": float(load["sets"]),
    }


def build_matchup_feature_snapshot(
    df: pd.DataFrame,
    player1: str,
    player2: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx = dict(context or {})
    match_date = pd.to_datetime(ctx.get("match_date"), errors="coerce") if ctx.get("match_date") else None
    surface = str(ctx.get("surface") or "Unknown")
    tournament = str(ctx.get("tournament") or ctx.get("tourney_name") or "")
    states = build_player_states(df, match_date)
    s1 = states[normalize_name(player1)]
    s2 = states[normalize_name(player2)]
    p1 = _state_snapshot(s1, surface, tournament, match_date)
    p2 = _state_snapshot(s2, surface, tournament, match_date)

    entry = parse_entry_context(ctx.get("player1_entry"), ctx.get("player2_entry"))
    fatigue_minutes_diff = (p1["current_tournament_minutes"] or 0.0) - (p2["current_tournament_minutes"] or 0.0)
    fatigue_match_diff = (p1["matches_last_14"] or 0.0) - (p2["matches_last_14"] or 0.0)
    elo_diff = (p1["overall_elo"] or 1500.0) - (p2["overall_elo"] or 1500.0)
    surface_elo_diff = (p1["surface_elo"] or 1500.0) - (p2["surface_elo"] or 1500.0)
    bo5_elo_diff = (p1["bo5_elo"] or 1500.0) - (p2["bo5_elo"] or 1500.0)
    serve_return_diff = ((p1["first_won_pct"] or 50.0) - (p2["first_won_pct"] or 50.0)) + 0.8 * ((p1["return_points_won_pct"] or 50.0) - (p2["return_points_won_pct"] or 50.0))
    recent_diff = (p1["recent_win_rate"] or 0.5) - (p2["recent_win_rate"] or 0.5)

    score_components = {
        "elo_score": 0.0040 * elo_diff,
        "surface_elo_score": 0.0045 * surface_elo_diff,
        "bo5_elo_score": 0.0025 * bo5_elo_diff if is_slam_context(ctx) else 0.0,
        "serve_return_score": 0.030 * serve_return_diff,
        "recent_form_score": 0.70 * recent_diff,
        "fatigue_score": -0.004 * fatigue_match_diff - 0.0008 * fatigue_minutes_diff,
        "entry_context_score": -0.05 * entry["qualifier_diff"] - 0.035 * entry["wildcard_diff"] - 0.04 * entry["lucky_loser_diff"],
    }

    return {
        "player1": player1,
        "player2": player2,
        "context": ctx,
        "elo": {
            "p1_overall": p1["overall_elo"],
            "p2_overall": p2["overall_elo"],
            "overall_diff": elo_diff,
            "p1_surface": p1["surface_elo"],
            "p2_surface": p2["surface_elo"],
            "surface_diff": surface_elo_diff,
            "p1_bo5": p1["bo5_elo"],
            "p2_bo5": p2["bo5_elo"],
            "bo5_diff": bo5_elo_diff,
        },
        "rolling_stats": {
            "p1_first_won_pct": p1["first_won_pct"],
            "p2_first_won_pct": p2["first_won_pct"],
            "p1_return_points_won_pct": p1["return_points_won_pct"],
            "p2_return_points_won_pct": p2["return_points_won_pct"],
            "p1_total_points_won_pct": p1["total_points_won_pct"],
            "p2_total_points_won_pct": p2["total_points_won_pct"],
            "serve_return_diff": serve_return_diff,
        },
        "fatigue": {
            "p1_days_since_last": p1["days_since_last"],
            "p2_days_since_last": p2["days_since_last"],
            "p1_matches_last_7": p1["matches_last_7"],
            "p2_matches_last_7": p2["matches_last_7"],
            "p1_matches_last_14": p1["matches_last_14"],
            "p2_matches_last_14": p2["matches_last_14"],
            "p1_current_tournament_matches": int(p1["current_tournament_matches"] or 0),
            "p2_current_tournament_matches": int(p2["current_tournament_matches"] or 0),
            "p1_current_tournament_minutes": p1["current_tournament_minutes"],
            "p2_current_tournament_minutes": p2["current_tournament_minutes"],
            "p1_current_tournament_sets": p1["current_tournament_sets"],
            "p2_current_tournament_sets": p2["current_tournament_sets"],
        },
        "entry_context": entry,
        "score_components": score_components,
        "feature_score": sum(score_components.values()),
    }


def route_model_source(context: dict[str, Any] | None, has_market: bool, has_advanced_report: bool) -> dict[str, Any]:
    ctx = context or {}
    if is_slam_context(ctx) and has_market and has_advanced_report:
        return {"model_source": "slam_market_calibrated_route_v1", "fallback_shrink": 0.12, "market_shrink": 0.35}
    if has_market and has_advanced_report:
        return {"model_source": "market_residual_calibrated_route_v1", "fallback_shrink": 0.15, "market_shrink": 0.45}
    if is_slam_context(ctx):
        return {"model_source": "slam_surface_bo5_feature_route_v1", "fallback_shrink": 0.18, "market_shrink": None}
    return {"model_source": "surface_elo_fatigue_feature_route_v1", "fallback_shrink": 0.22, "market_shrink": None}


def audit_schedule_quality(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "match_date" not in df.columns or "tourney_name" not in df.columns:
        return {"collapsed_tournaments": [], "caveats": ["No match_date/tourney_name columns available for schedule audit."]}
    dated = _ensure_dates(df)
    collapsed: list[dict[str, Any]] = []
    caveats: list[str] = []
    for tournament, group in dated.groupby("tourney_name", dropna=False):
        rows = int(len(group))
        unique_dates = int(group["match_date"].nunique(dropna=True))
        if rows >= 5 and unique_dates <= 1:
            collapsed.append({"tournament": str(tournament), "rows": rows, "unique_dates": unique_dates})
    if collapsed:
        caveats.append("Collapsed or low-granularity match dates detected; current-tournament fatigue may be understated until real schedule dates are acquired.")
    return {"collapsed_tournaments": collapsed, "caveats": caveats}


__all__ = [
    "PlayerState",
    "audit_schedule_quality",
    "build_matchup_feature_snapshot",
    "build_player_states",
    "expected_score",
    "is_slam_context",
    "parse_entry_context",
    "route_model_source",
    "update_elo",
]
