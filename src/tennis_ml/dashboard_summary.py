"""Dashboard-facing summaries for tennis model research artifacts.

The helpers in this module intentionally compress large diagnostic JSON files into
small decision-oriented payloads for the Streamlit app.  They do not create new
model claims; they surface whether the latest artifacts beat, trail, or merely
risk-control the calibrated-market baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CALIBRATED_MARKET_MODEL = "market_bin_recalibrated"
RAW_MARKET_MODEL = "market_no_vig"
RESIDUAL_FILTERED_MODEL = "residual_overlay_filtered"


def _round_metric(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _model_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("overall_model_comparison") or []
    return [row for row in rows if isinstance(row, dict) and row.get("model")]


def _by_model(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["model"]): row for row in _model_rows(report)}


def _best_by_proper_scores(report: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        row
        for row in _model_rows(report)
        if row.get("log_loss") is not None and row.get("brier") is not None
    ]
    if not candidates:
        return {}
    return min(candidates, key=lambda row: (float(row["log_loss"]), float(row["brier"])))


def _compact_model(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "model": row.get("model"),
        "rows": row.get("rows"),
        "accuracy": _round_metric(row.get("accuracy")),
        "log_loss": _round_metric(row.get("log_loss")),
        "brier": _round_metric(row.get("brier")),
    }


def _ability_routing_summary(ability_report: dict[str, Any]) -> dict[str, Any]:
    diagnostics = ability_report.get("routing_diagnostics") or {}
    best: dict[str, Any] | None = None
    best_name = None
    for name, payload in diagnostics.items():
        if not isinstance(payload, dict):
            continue
        delta = payload.get("routed_minus_baseline_log_loss")
        if delta is None:
            continue
        if best is None or float(delta) < float(best.get("routed_minus_baseline_log_loss", float("inf"))):
            best = payload
            best_name = name
    if best is None:
        return {
            "available": False,
            "best_routing_name": None,
            "best_routed_minus_baseline_log_loss": None,
            "best_routed_minus_baseline_brier": None,
            "best_routed_rows": 0,
            "beats_calibrated_market": False,
        }
    log_delta = _round_metric(best.get("routed_minus_baseline_log_loss"))
    brier_delta = _round_metric(best.get("routed_minus_baseline_brier"))
    return {
        "available": True,
        "best_routing_name": best_name,
        "best_routed_minus_baseline_log_loss": log_delta,
        "best_routed_minus_baseline_brier": brier_delta,
        "best_routed_rows": int(best.get("routed_rows") or 0),
        "beats_calibrated_market": bool(
            log_delta is not None and brier_delta is not None and log_delta < 0 and brier_delta < 0
        ),
    }


def _material_calibration_gaps(row: dict[str, Any] | None, limit: int = 3) -> list[dict[str, Any]]:
    bins = [] if not row else row.get("calibration_bins") or []
    material = []
    for item in bins:
        if not isinstance(item, dict) or item.get("rows", 0) < 250:
            continue
        error = item.get("calibration_error")
        if error is None:
            continue
        material.append(
            {
                "bin": item.get("bin"),
                "rows": item.get("rows"),
                "mean_prob": _round_metric(item.get("mean_prob")),
                "actual_rate": _round_metric(item.get("actual_rate")),
                "calibration_error": _round_metric(error),
                "direction": "overpredicts_p1" if float(error) > 0 else "underpredicts_p1",
                "weighted_abs_error": _round_metric(abs(float(error)) * int(item.get("rows") or 0)),
            }
        )
    return sorted(material, key=lambda item: item["weighted_abs_error"] or 0, reverse=True)[:limit]


def _artifact_alignment(best: dict[str, Any], advanced_report: dict[str, Any], ability_report: dict[str, Any]) -> dict[str, Any]:
    advanced_rows = best.get("rows")
    ability_rows = ability_report.get("rows")
    advanced_years = advanced_report.get("test_years")
    ability_years = ability_report.get("years")
    warnings = []
    if advanced_rows is not None and ability_rows is not None and int(advanced_rows) != int(ability_rows):
        warnings.append("Advanced and ability artifacts cover different row counts.")
    if advanced_years is not None and ability_years is not None and list(advanced_years) != list(ability_years):
        warnings.append("Advanced test years and ability years differ.")
    baseline_col = ability_report.get("baseline_probability_col")
    if baseline_col is not None and baseline_col != "market_bin_recalibrated_p1":
        warnings.append("Ability diagnostics are not benchmarked against market_bin_recalibrated_p1.")
    return {
        "advanced_rows": advanced_rows,
        "ability_rows": ability_rows,
        "advanced_test_years": advanced_years,
        "ability_years": ability_years,
        "baseline_probability_col": baseline_col,
        "warnings": warnings,
    }


def build_dashboard_summary(
    advanced_report: dict[str, Any], ability_report: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a concise decision summary from current research artifacts."""
    ability_report = ability_report or {}
    models = _by_model(advanced_report)
    best = _best_by_proper_scores(advanced_report)
    calibrated = models.get(CALIBRATED_MARKET_MODEL, {})
    raw_market = models.get(RAW_MARKET_MODEL, {})
    residual = models.get(RESIDUAL_FILTERED_MODEL, {})

    best_log_loss = float(best.get("log_loss", float("nan"))) if best else None
    best_brier = float(best.get("brier", float("nan"))) if best else None
    cal_log_loss = float(calibrated.get("log_loss", float("nan"))) if calibrated else None
    cal_brier = float(calibrated.get("brier", float("nan"))) if calibrated else None

    best_summary = _compact_model(best)
    if calibrated and raw_market:
        best_summary["beats_market_no_vig_by_log_loss"] = _round_metric(
            float(raw_market["log_loss"]) - float(calibrated["log_loss"]), 6
        )
        best_summary["beats_market_no_vig_by_brier"] = _round_metric(
            float(raw_market["brier"]) - float(calibrated["brier"]), 6
        )

    residual_summary = _compact_model(residual)
    if residual and best:
        residual_summary["log_loss_delta_vs_best"] = _round_metric(float(residual["log_loss"]) - float(best["log_loss"]), 6)
        residual_summary["brier_delta_vs_best"] = _round_metric(float(residual["brier"]) - float(best["brier"]), 6)

    ability_summary = _ability_routing_summary(ability_report)
    calibrated_beaten = bool(
        best
        and calibrated
        and best.get("model") != CALIBRATED_MARKET_MODEL
        and best_log_loss is not None
        and best_brier is not None
        and cal_log_loss is not None
        and cal_brier is not None
        and best_log_loss < cal_log_loss
        and best_brier < cal_brier
    )

    if calibrated_beaten or ability_summary["beats_calibrated_market"]:
        decision = "continue"
        reason = "A model/policy currently beats calibrated market on both log_loss and Brier."
    else:
        decision = "pause_or_change_scope"
        reason = (
            "Current automated overlays/routing do not beat calibrated market; next useful run needs a "
            "non-placeholder data source or a product/reporting requirement, not another segment scan."
        )

    return {
        "generated_at": advanced_report.get("generated_at"),
        "test_years": advanced_report.get("test_years"),
        "current_best": best_summary,
        "calibrated_market": _compact_model(calibrated),
        "market_no_vig": _compact_model(raw_market),
        "residual_overlay_filtered": residual_summary,
        "ability_routing": ability_summary,
        "artifact_alignment": _artifact_alignment(best, advanced_report, ability_report),
        "top_calibration_gaps": _material_calibration_gaps(best),
        "stable_failure_counts": {
            "filtered_overlay_lags_calibrated_market": len(
                advanced_report.get("where_filtered_overlay_stably_lags_calibrated_market") or []
            ),
            "filtered_overlay_beats_calibrated_market": len(
                advanced_report.get("where_filtered_overlay_beats_calibrated_market") or []
            ),
        },
        "automation_recommendation": {"decision": decision, "reason": reason},
        "research_only_guardrail": "No betting execution; model-quality diagnostics and paper tracking only.",
    }


def write_dashboard_summary_file(
    advanced_path: str | Path,
    ability_path: str | Path | None,
    output_path: str | Path,
) -> dict[str, Any]:
    """Read current artifacts, write the dashboard decision summary JSON, and return it."""
    advanced_payload = json.loads(Path(advanced_path).read_text(encoding="utf-8"))
    ability_payload: dict[str, Any] = {}
    if ability_path is not None and Path(ability_path).exists():
        ability_payload = json.loads(Path(ability_path).read_text(encoding="utf-8"))
    summary = build_dashboard_summary(advanced_payload, ability_payload)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
