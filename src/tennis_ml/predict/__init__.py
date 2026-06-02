"""Matchup prediction API."""
from .matchup_engine import (
    MatchupPrediction,
    PlayerComparison,
    clamp_probability,
    confidence_label,
    conservative_calibrate,
    predict_matchup,
)
from .live_features import (
    audit_schedule_quality,
    build_matchup_feature_snapshot,
    parse_entry_context,
    route_model_source,
)

__all__ = [
    "MatchupPrediction",
    "PlayerComparison",
    "audit_schedule_quality",
    "build_matchup_feature_snapshot",
    "clamp_probability",
    "confidence_label",
    "conservative_calibrate",
    "parse_entry_context",
    "predict_matchup",
    "route_model_source",
]
