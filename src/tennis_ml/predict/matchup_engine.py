"""Transparent matchup prediction engine for the Streamlit research app.

This module is intentionally lightweight: it turns the current enriched export and
research artifacts into a deterministic, auditable matchup breakdown. It does not
place bets, submit orders, or connect to financial execution systems.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math

import pandas as pd

from .live_features import audit_schedule_quality, build_matchup_feature_snapshot, route_model_source

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPORT_PATH = ROOT / "data" / "exports" / "atp_matches_enriched_current.csv"
DEFAULT_METRICS_PATH = ROOT / "data" / "betting_research" / "latest_metrics.json"
DEFAULT_ADVANCED_REPORT_PATH = ROOT / "data" / "betting_research" / "latest_advanced_feature_model_research.json"


@dataclass(frozen=True)
class PlayerComparison:
    name: str
    matches: int = 0
    wins: int = 0
    win_rate: float | None = None
    surface_matches: int = 0
    surface_wins: int = 0
    surface_win_rate: float | None = None
    recent_matches: int = 0
    recent_wins: int = 0
    recent_win_rate: float | None = None
    avg_total_points_won_pct: float | None = None
    avg_return_points_won_pct: float | None = None
    avg_first_won_pct: float | None = None


@dataclass(frozen=True)
class MatchupPrediction:
    player1: str
    player2: str
    p1_probability: float
    p2_probability: float
    pick: str
    confidence: str
    model_source: str
    market_p1_probability: float | None = None
    market_p2_probability: float | None = None
    edge_p1: float | None = None
    reasons: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    player_comparison: dict[str, PlayerComparison] = field(default_factory=dict)
    feature_snapshot: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["player_comparison"] = {
            key: asdict(value) if hasattr(value, "__dataclass_fields__") else value
            for key, value in self.player_comparison.items()
        }
        return payload


def clamp_probability(value: float, lower: float = 0.03, upper: float = 0.97) -> float:
    """Clamp a probability to avoid fake certainty in a lightweight fallback model."""
    if pd.isna(value) or math.isinf(float(value)):
        return 0.5
    return max(lower, min(upper, float(value)))


def confidence_label(probability: float) -> str:
    """Return a deterministic human-readable confidence bucket."""
    margin = abs(float(probability) - 0.5)
    if margin >= 0.22:
        return "high"
    if margin >= 0.12:
        return "medium"
    if margin >= 0.06:
        return "lean"
    return "low"


def conservative_calibrate(probability: float, shrink_to_market: float | None = None, shrink: float = 0.15) -> float:
    """Conservatively shrink model probability toward market if supplied, else toward 0.50.

    This is not a trained calibration model. It is a transparent guardrail used by
    the product MVP until the heavier no-lookahead recalibration experiments are
    wired into live matchup inference.
    """
    target = 0.5 if shrink_to_market is None else float(shrink_to_market)
    return clamp_probability((1.0 - shrink) * float(probability) + shrink * target)


def _read_json(path: Path | str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _load_export(path: Path | str | None) -> pd.DataFrame:
    p = Path(path or DEFAULT_EXPORT_PATH)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


def _norm_name(name: str) -> str:
    return " ".join(str(name).lower().strip().split())


def _safe_rate(num: int | float, den: int | float) -> float | None:
    if not den:
        return None
    return float(num) / float(den)


def _mean_or_none(values: list[float]) -> float | None:
    clean = [float(v) for v in values if not pd.isna(v)]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _player_rows(df: pd.DataFrame, player: str) -> pd.DataFrame:
    name = _norm_name(player)
    if df.empty:
        return df
    winner = df.get("winner_name", pd.Series(dtype=str)).astype(str).map(_norm_name) == name
    loser = df.get("loser_name", pd.Series(dtype=str)).astype(str).map(_norm_name) == name
    return df[winner | loser].copy()


def _player_summary(df: pd.DataFrame, player: str, surface: str | None = None, recent_n: int = 10) -> PlayerComparison:
    rows = _player_rows(df, player)
    if rows.empty:
        return PlayerComparison(name=player)

    player_norm = _norm_name(player)
    is_win = rows["winner_name"].astype(str).map(_norm_name) == player_norm
    rows = rows.assign(_is_win=is_win.astype(int)).sort_values("match_date" if "match_date" in rows.columns else rows.index.name or "_is_win")

    surface_rows = rows
    if surface and "surface" in rows.columns:
        surface_rows = rows[rows["surface"].astype(str).str.lower() == str(surface).lower()]

    recent = rows.tail(recent_n)

    total_points: list[float] = []
    return_points: list[float] = []
    first_won: list[float] = []
    for _, row in rows.iterrows():
        prefix = "winner" if _norm_name(row.get("winner_name", "")) == player_norm else "loser"
        for col, bucket in [
            (f"{prefix}_total_points_won_pct", total_points),
            (f"{prefix}_return_points_won_pct", return_points),
            (f"{prefix}_serve_first_won_pct", first_won),
        ]:
            if col in row and not pd.isna(row[col]):
                bucket.append(float(row[col]))

    return PlayerComparison(
        name=player,
        matches=int(len(rows)),
        wins=int(rows["_is_win"].sum()),
        win_rate=_safe_rate(int(rows["_is_win"].sum()), len(rows)),
        surface_matches=int(len(surface_rows)),
        surface_wins=int(surface_rows["_is_win"].sum()) if not surface_rows.empty else 0,
        surface_win_rate=_safe_rate(int(surface_rows["_is_win"].sum()), len(surface_rows)) if not surface_rows.empty else None,
        recent_matches=int(len(recent)),
        recent_wins=int(recent["_is_win"].sum()) if not recent.empty else 0,
        recent_win_rate=_safe_rate(int(recent["_is_win"].sum()), len(recent)) if not recent.empty else None,
        avg_total_points_won_pct=_mean_or_none(total_points),
        avg_return_points_won_pct=_mean_or_none(return_points),
        avg_first_won_pct=_mean_or_none(first_won),
    )


def _diff(a: float | None, b: float | None, default: float = 0.0) -> float:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return default
    return float(a) - float(b)


def _market_probability(odds: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if not odds:
        return None, None
    if "p1_probability" in odds:
        p1 = clamp_probability(float(odds["p1_probability"]), 0.001, 0.999)
        p2 = 1.0 - p1
        return p1, p2
    if "p1_prob" in odds:
        p1 = clamp_probability(float(odds["p1_prob"]), 0.001, 0.999)
        return p1, 1.0 - p1
    if "p1_odds" in odds and "p2_odds" in odds:
        p1_raw = 1.0 / float(odds["p1_odds"])
        p2_raw = 1.0 / float(odds["p2_odds"])
        total = p1_raw + p2_raw
        if total > 0:
            return p1_raw / total, p2_raw / total
    return None, None


def _artifact_summary(metrics: dict[str, Any], advanced: dict[str, Any]) -> dict[str, Any]:
    metrics_block = metrics.get("metrics", metrics) if isinstance(metrics, dict) else {}
    top_models = advanced.get("top_models") or advanced.get("overall_metrics") or advanced.get("models") or []
    return {
        "daily_accuracy": metrics_block.get("accuracy") if isinstance(metrics_block, dict) else None,
        "daily_log_loss": metrics_block.get("log_loss") if isinstance(metrics_block, dict) else None,
        "advanced_generated_at": advanced.get("generated_at"),
        "advanced_top_model_count": len(top_models) if isinstance(top_models, list) else None,
    }


def predict_matchup(
    player1: str,
    player2: str,
    context: dict[str, Any] | None = None,
    export_path: Path | str | None = None,
    metrics_path: Path | str | None = None,
    advanced_report_path: Path | str | None = None,
    odds: dict[str, Any] | None = None,
) -> MatchupPrediction:
    """Predict a matchup using available current-export summaries and research artifacts.

    The output is product-ready for Streamlit, but the model source is a transparent
    fallback, not a claim of deployable betting edge.
    """
    ctx = dict(context or {})
    surface = ctx.get("surface")
    df = _load_export(export_path)
    metrics = _read_json(metrics_path or DEFAULT_METRICS_PATH)
    advanced = _read_json(advanced_report_path or DEFAULT_ADVANCED_REPORT_PATH)
    caveats: list[str] = []
    risk_flags: list[str] = []
    reasons: list[str] = []

    if not player1 or not player2:
        caveats.append("Both player names are required; returning neutral probability.")
    if _norm_name(player1) == _norm_name(player2):
        caveats.append("Player names are identical; returning neutral probability.")

    if df.empty:
        caveats.append("Current enriched export is missing; used neutral fallback only.")
        p1_raw = 0.5
        feature_snapshot: dict[str, Any] = {}
        route = route_model_source(ctx, has_market=bool(odds), has_advanced_report=bool(advanced))
        p1_summary = PlayerComparison(name=player1)
        p2_summary = PlayerComparison(name=player2)
    else:
        schedule_audit = audit_schedule_quality(df)
        caveats.extend(schedule_audit.get("caveats", []))
        feature_snapshot = build_matchup_feature_snapshot(df, player1, player2, ctx)
        route = route_model_source(ctx, has_market=bool(odds), has_advanced_report=bool(advanced))
        p1_summary = _player_summary(df, player1, surface=surface)
        p2_summary = _player_summary(df, player2, surface=surface)
        if p1_summary.matches == 0:
            caveats.append(f"No current-export history found for {player1}.")
        if p2_summary.matches == 0:
            caveats.append(f"No current-export history found for {player2}.")

        score = 0.0
        overall_diff = _diff(p1_summary.win_rate, p2_summary.win_rate)
        surface_diff = _diff(p1_summary.surface_win_rate, p2_summary.surface_win_rate)
        recent_diff = _diff(p1_summary.recent_win_rate, p2_summary.recent_win_rate)
        tpw_diff = _diff(p1_summary.avg_total_points_won_pct, p2_summary.avg_total_points_won_pct) / 100.0
        rpw_diff = _diff(p1_summary.avg_return_points_won_pct, p2_summary.avg_return_points_won_pct) / 100.0
        first_won_diff = _diff(p1_summary.avg_first_won_pct, p2_summary.avg_first_won_pct) / 100.0

        score += 1.15 * overall_diff
        score += 0.85 * surface_diff
        score += 0.70 * recent_diff
        score += 1.10 * tpw_diff
        score += 0.65 * rpw_diff
        score += 0.45 * first_won_diff
        feature_score = float(feature_snapshot.get("feature_score", 0.0))
        score += 0.45 * feature_score

        if surface and (p1_summary.surface_matches < 3 or p2_summary.surface_matches < 3):
            risk_flags.append("Low surface-specific sample for at least one player.")
        if p1_summary.matches < 5 or p2_summary.matches < 5:
            risk_flags.append("Low current-export sample for at least one player.")
        if ctx.get("best_of_5") and str(ctx.get("tournament", "")).strip():
            reasons.append("Best-of-five/Slam context captured as context only; dedicated BO5 model is a next refinement.")

        p1_raw = 1.0 / (1.0 + math.exp(-score))
        reasons.extend([
            f"Overall win-rate diff: {overall_diff:+.3f}",
            f"{surface or 'Selected'} surface win-rate diff: {surface_diff:+.3f}",
            f"Recent-form diff: {recent_diff:+.3f}",
            f"Total-points-won profile diff: {tpw_diff:+.3f}",
            f"Return-points-won profile diff: {rpw_diff:+.3f}",
            f"Live feature score from Elo/fatigue/serve-return/entry routing: {feature_score:+.3f}",
        ])
        components = feature_snapshot.get("score_components", {})
        if components:
            top_components = sorted(components.items(), key=lambda item: abs(float(item[1])), reverse=True)[:4]
            reasons.extend([f"{name.replace('_', ' ')}: {float(value):+.3f}" for name, value in top_components])

    market_p1, market_p2 = _market_probability(odds)
    if market_p1 is not None:
        p1_prob = conservative_calibrate(p1_raw, shrink_to_market=market_p1, shrink=float(route.get("market_shrink") or 0.35))
        reasons.append("Probability routed through calibrated market/residual shrinkage using supplied no-vig market probability.")
    else:
        p1_prob = conservative_calibrate(p1_raw, shrink_to_market=None, shrink=float(route.get("fallback_shrink") or 0.22))
        caveats.append("No live odds supplied; model uses transparent historical/player-profile fallback.")

    p1_prob = clamp_probability(p1_prob)
    p2_prob = 1.0 - p1_prob
    edge = None if market_p1 is None else p1_prob - market_p1
    pick = player1 if p1_prob >= 0.5 else player2

    artifacts = _artifact_summary(metrics, advanced)
    artifacts.update({
        "export_path": str(Path(export_path or DEFAULT_EXPORT_PATH)),
        "metrics_path": str(Path(metrics_path or DEFAULT_METRICS_PATH)),
        "advanced_report_path": str(Path(advanced_report_path or DEFAULT_ADVANCED_REPORT_PATH)),
    })

    return MatchupPrediction(
        player1=player1,
        player2=player2,
        p1_probability=round(float(p1_prob), 6),
        p2_probability=round(float(p2_prob), 6),
        pick=pick,
        confidence=confidence_label(p1_prob),
        model_source=str(route.get("model_source", "surface_elo_fatigue_feature_route_v1")),
        market_p1_probability=round(float(market_p1), 6) if market_p1 is not None else None,
        market_p2_probability=round(float(market_p2), 6) if market_p2 is not None else None,
        edge_p1=round(float(edge), 6) if edge is not None else None,
        reasons=reasons,
        risk_flags=risk_flags,
        caveats=caveats,
        player_comparison={"player1": p1_summary, "player2": p2_summary},
        feature_snapshot=feature_snapshot,
        context=ctx,
        artifacts=artifacts,
    )


__all__ = [
    "MatchupPrediction",
    "PlayerComparison",
    "clamp_probability",
    "confidence_label",
    "conservative_calibrate",
    "predict_matchup",
]
