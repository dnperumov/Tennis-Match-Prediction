from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ml.dashboard_summary import build_dashboard_summary, write_dashboard_summary_file  # noqa: E402


def test_dashboard_summary_names_current_best_and_recommends_pause_when_no_model_beats_it():
    advanced_report = {
        "generated_at": "2026-06-05T00:00:00+00:00",
        "test_years": [2022, 2023, 2024, 2025, 2026],
        "overall_model_comparison": [
            {"model": "market_bin_recalibrated", "rows": 11801, "accuracy": 0.6782, "log_loss": 0.5886, "brier": 0.2025},
            {"model": "market_no_vig", "rows": 11801, "accuracy": 0.6789, "log_loss": 0.5888, "brier": 0.2026},
            {"model": "residual_overlay_filtered", "rows": 11801, "accuracy": 0.6787, "log_loss": 0.5896, "brier": 0.2030},
        ],
        "where_filtered_overlay_stably_lags_calibrated_market": [{"segment": "heavy favorite underpriced"}],
        "where_filtered_overlay_beats_calibrated_market": [],
    }
    ability_report = {
        "overall_model_vs_baseline": {
            "log_loss": 0.5896,
            "brier": 0.2030,
            "market_log_loss": 0.5886,
            "market_brier": 0.2025,
        },
        "routing_diagnostics": {
            "fallback": {
                "routed_rows": 1590,
                "routed_minus_baseline_log_loss": 0.0008,
                "routed_minus_baseline_brier": 0.0004,
            }
        },
    }

    summary = build_dashboard_summary(advanced_report, ability_report)

    assert summary["current_best"]["model"] == "market_bin_recalibrated"
    assert summary["current_best"]["beats_market_no_vig_by_log_loss"] == 0.0002
    assert summary["residual_overlay_filtered"]["log_loss_delta_vs_best"] == 0.001
    assert summary["ability_routing"]["best_routed_minus_baseline_log_loss"] == 0.0008
    assert summary["artifact_alignment"] == {
        "advanced_rows": 11801,
        "ability_rows": None,
        "advanced_test_years": [2022, 2023, 2024, 2025, 2026],
        "ability_years": None,
        "baseline_probability_col": None,
        "aligned": True,
        "status": "aligned",
        "warnings": [],
    }
    assert summary["automation_recommendation"]["decision"] == "pause_or_change_scope"
    assert "non-placeholder data source" in summary["automation_recommendation"]["reason"]


def test_dashboard_summary_warns_when_artifact_scopes_do_not_align():
    advanced_report = {
        "test_years": [2022, 2023],
        "overall_model_comparison": [
            {"model": "market_bin_recalibrated", "rows": 500, "accuracy": 0.65, "log_loss": 0.59, "brier": 0.20},
            {"model": "market_no_vig", "rows": 500, "accuracy": 0.64, "log_loss": 0.591, "brier": 0.201},
        ],
    }
    ability_report = {
        "rows": 11801,
        "years": [2022, 2023, 2024, 2025, 2026],
        "baseline_probability_col": "market_bin_recalibrated_p1",
    }

    summary = build_dashboard_summary(advanced_report, ability_report)

    assert summary["artifact_alignment"]["advanced_rows"] == 500
    assert summary["artifact_alignment"]["ability_rows"] == 11801
    assert summary["artifact_alignment"]["aligned"] is False
    assert summary["artifact_alignment"]["status"] == "scope_mismatch"
    assert "Advanced and ability artifacts cover different row counts." in summary["artifact_alignment"]["warnings"]
    assert "Advanced test years and ability years differ." in summary["artifact_alignment"]["warnings"]
    assert summary["automation_recommendation"]["decision"] == "pause_or_change_scope"


def test_dashboard_summary_allows_continue_when_policy_beats_calibrated_market():
    advanced_report = {
        "overall_model_comparison": [
            {"model": "market_bin_recalibrated", "rows": 100, "accuracy": 0.65, "log_loss": 0.59, "brier": 0.20},
            {"model": "new_policy", "rows": 100, "accuracy": 0.66, "log_loss": 0.5893, "brier": 0.1995},
        ]
    }

    summary = build_dashboard_summary(advanced_report, ability_report={})

    assert summary["current_best"]["model"] == "new_policy"
    assert summary["automation_recommendation"]["decision"] == "continue"
    assert summary["automation_recommendation"]["reason"] == "A model/policy currently beats calibrated market on both log_loss and Brier."


def test_write_dashboard_summary_file_creates_decision_artifact(tmp_path):
    advanced_path = tmp_path / "advanced.json"
    ability_path = tmp_path / "ability.json"
    output_path = tmp_path / "latest_dashboard_decision_summary.json"
    advanced_path.write_text(
        '{"generated_at":"2026-06-06T00:00:00+00:00","test_years":[2026],'
        '"overall_model_comparison":[{"model":"market_bin_recalibrated","rows":10,'
        '"accuracy":0.7,"log_loss":0.58,"brier":0.20},{"model":"market_no_vig",'
        '"rows":10,"accuracy":0.7,"log_loss":0.59,"brier":0.21}]}',
        encoding="utf-8",
    )
    ability_path.write_text(
        '{"rows":10,"years":[2026],"baseline_probability_col":"market_bin_recalibrated_p1",'
        '"routing_diagnostics":{}}',
        encoding="utf-8",
    )

    summary = write_dashboard_summary_file(advanced_path, ability_path, output_path)

    assert output_path.exists()
    assert summary["current_best"]["model"] == "market_bin_recalibrated"
    assert '"research_only_guardrail": "No betting execution; model-quality diagnostics and paper tracking only."' in output_path.read_text(encoding="utf-8")


def test_generate_dashboard_decision_summary_cli_writes_output(tmp_path):
    advanced_path = tmp_path / "advanced.json"
    ability_path = tmp_path / "ability.json"
    output_path = tmp_path / "decision.json"
    advanced_path.write_text(
        '{"generated_at":"2026-06-06T00:00:00+00:00","overall_model_comparison":['
        '{"model":"market_bin_recalibrated","rows":5,"accuracy":0.6,"log_loss":0.6,"brier":0.22}]}',
        encoding="utf-8",
    )
    ability_path.write_text('{"rows":5,"years":[],"routing_diagnostics":{}}', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_dashboard_decision_summary.py"),
            "--advanced",
            str(advanced_path),
            "--ability",
            str(ability_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "pause_or_change_scope" in result.stdout
    assert output_path.exists()
