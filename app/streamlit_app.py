from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tennis_ml.dashboard_summary import build_dashboard_summary  # noqa: E402
from tennis_ml.predict import predict_matchup  # noqa: E402

EXPORT_PATH = ROOT / "data" / "exports" / "atp_matches_enriched_current.csv"
DB_PATH = ROOT / "data" / "tennis_matches.sqlite"
METRICS_PATH = ROOT / "data" / "betting_research" / "latest_metrics.json"
ADVANCED_REPORT_PATH = ROOT / "data" / "betting_research" / "latest_advanced_feature_model_research.json"
ABILITY_REPORT_PATH = ROOT / "data" / "betting_research" / "latest_ability_pressure_diagnostic.json"
PRED_PATH = ROOT / "data" / "betting_research" / "latest_backtest_predictions.csv"
NICHE_REPORT_PATH = ROOT / "data" / "betting_research" / "latest_niche_research.json"
NICHE_SEGMENTS_PATH = ROOT / "data" / "betting_research" / "latest_niche_segments.csv"
MODEL_PATH = ROOT / "models" / "betting_research" / "latest_model.pkl"

st.set_page_config(page_title="Tennis ML Research", layout="wide")
st.title("Tennis ML + Betting Research Dashboard")
st.caption("Research only. This app does not place bets or connect to sportsbooks.")


@st.cache_data(ttl=60)
def load_current_export() -> pd.DataFrame:
    if not EXPORT_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(EXPORT_PATH)


@st.cache_data(ttl=60)
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())


@st.cache_data(ttl=60)
def load_json_artifact(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    if not PRED_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PRED_PATH, parse_dates=["date"])
    if "model_p1" in df.columns and "implied_p1_no_vig" in df.columns:
        df["edge"] = df["model_p1"] - df["implied_p1_no_vig"]
    return df


@st.cache_data(ttl=60)
def load_niche_report() -> dict:
    if not NICHE_REPORT_PATH.exists():
        return {}
    return json.loads(NICHE_REPORT_PATH.read_text())


@st.cache_data(ttl=60)
def load_niche_segments() -> pd.DataFrame:
    if not NICHE_SEGMENTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(NICHE_SEGMENTS_PATH)


def db_counts() -> dict:
    if not DB_PATH.exists():
        return {}
    con = sqlite3.connect(DB_PATH)
    try:
        return {
            "matches": con.execute("select count(*) from matches").fetchone()[0],
            "player_stat_rows": con.execute("select count(*) from player_match_stats").fetchone()[0],
            "date_range": con.execute("select min(match_date), max(match_date) from matches").fetchone(),
            "latest_tournaments": con.execute(
                "select tourney_name, min(match_date), max(match_date), count(*) from matches group by tourney_name order by max(match_date) desc limit 10"
            ).fetchall(),
        }
    finally:
        con.close()


current = load_current_export()
metrics_payload = load_metrics()
advanced_payload = load_json_artifact(ADVANCED_REPORT_PATH)
ability_payload = load_json_artifact(ABILITY_REPORT_PATH)
decision_summary = build_dashboard_summary(advanced_payload, ability_payload) if advanced_payload else {}
preds = load_predictions()
counts = db_counts()

niche_report = load_niche_report()
niche_segments = load_niche_segments()

tab_summary, tab_matchup, tab_data, tab_model, tab_edges, tab_niches, tab_rows = st.tabs([
    "Research summary",
    "Matchup Breakdown",
    "Data",
    "Model",
    "Edge backtest",
    "Niche research",
    "Rows",
])

with tab_summary:
    st.subheader("Current all-data decision summary")
    st.info("Research only. These metrics benchmark prediction quality and paper tracking; no betting execution.")
    if not decision_summary:
        st.warning("No advanced research artifact found. Run: `.venv/bin/python scripts/advanced_feature_model_research.py`")
    else:
        rec = decision_summary.get("automation_recommendation", {})
        if rec.get("decision") == "continue":
            st.success(rec.get("reason"))
        else:
            st.warning(rec.get("reason"))

        best = decision_summary.get("current_best", {})
        residual = decision_summary.get("residual_overlay_filtered", {})
        ability = decision_summary.get("ability_routing", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best proper-score model", best.get("model", "n/a"))
        c2.metric("Best log loss", best.get("log_loss", "n/a"))
        c3.metric("Best Brier", best.get("brier", "n/a"))
        c4.metric("Rows", best.get("rows", "n/a"))

        m1, m2, m3 = st.columns(3)
        m1.metric("Calibrated vs raw market log-loss gain", best.get("beats_market_no_vig_by_log_loss", "n/a"))
        m2.metric("Filtered overlay log-loss delta vs best", residual.get("log_loss_delta_vs_best", "n/a"))
        m3.metric("Best ability routing delta vs calibrated", ability.get("best_routed_minus_baseline_log_loss", "n/a"))

        st.subheader("Model comparison contract")
        st.dataframe(
            pd.DataFrame(
                [
                    decision_summary.get("calibrated_market", {}),
                    decision_summary.get("market_no_vig", {}),
                    residual,
                ]
            ),
            use_container_width=True,
        )

        gaps = decision_summary.get("top_calibration_gaps", [])
        alignment = decision_summary.get("artifact_alignment", {})
        if alignment.get("warnings"):
            st.subheader("Artifact alignment warnings")
            for warning in alignment.get("warnings", []):
                st.warning(warning)
            with st.expander("Artifact alignment details"):
                st.json(alignment)
        elif alignment:
            st.caption("Artifact alignment: advanced model and ability/routing diagnostics cover the same rows, years, and calibrated-market baseline.")

        if gaps:
            st.subheader("Largest material calibration gaps for current best")
            st.dataframe(pd.DataFrame(gaps), use_container_width=True)

        st.subheader("Known failure / success counts")
        st.json(decision_summary.get("stable_failure_counts", {}))
        with st.expander("Raw decision summary"):
            st.json(decision_summary)

with tab_matchup:
    st.subheader("Matchup Breakdown MVP")
    st.info("Research only. This tool explains model probabilities; it does not place bets or execute trades.")

    if current.empty:
        player_options: list[str] = []
    else:
        player_options = sorted(
            set(current.get("winner_name", pd.Series(dtype=str)).dropna().astype(str))
            | set(current.get("loser_name", pd.Series(dtype=str)).dropna().astype(str))
        )

    c1, c2 = st.columns(2)
    if player_options:
        default_p2 = 1 if len(player_options) > 1 else 0
        player1 = c1.selectbox("Player 1", player_options, index=0)
        player2 = c2.selectbox("Player 2", player_options, index=default_p2)
    else:
        player1 = c1.text_input("Player 1")
        player2 = c2.text_input("Player 2")

    c3, c4, c5, c6 = st.columns(4)
    tournament_default = "Roland Garros" if current.empty or "tourney_name" not in current.columns else str(current["tourney_name"].dropna().tail(1).iloc[0])
    tournament = c3.text_input("Tournament", value=tournament_default)
    surfaces = ["", "Hard", "Clay", "Grass", "Carpet"]
    surface = c4.selectbox("Surface", surfaces, index=2 if tournament_default == "Roland Garros" else 0)
    round_name = c5.text_input("Round", value="")
    best_of_5 = c6.checkbox("Best of 5 / Slam context", value=bool(tournament.lower() in {"roland garros", "wimbledon", "us open", "australian open"}))
    c7, c8, c9 = st.columns(3)
    match_date = c7.text_input("Match date for no-lookahead features", value="")
    player1_entry = c8.selectbox("Player 1 entry", ["", "Q", "WC", "LL", "PR"])
    player2_entry = c9.selectbox("Player 2 entry", ["", "Q", "WC", "LL", "PR"])

    st.markdown("**Optional market input** — decimal odds are converted to no-vig implied probability.")
    m1, m2 = st.columns(2)
    p1_odds = m1.number_input("Player 1 decimal odds", min_value=1.01, value=1.90, step=0.01)
    p2_odds = m2.number_input("Player 2 decimal odds", min_value=1.01, value=1.90, step=0.01)
    use_market = st.checkbox("Use these market odds in the breakdown", value=False)

    if player1 and player2:
        odds_payload = {"p1_odds": p1_odds, "p2_odds": p2_odds} if use_market else None
        prediction = predict_matchup(
            player1,
            player2,
            context={
                "tournament": tournament,
                "surface": surface or None,
                "round": round_name or None,
                "match_date": match_date or None,
                "best_of_5": best_of_5,
                "player1_entry": player1_entry or None,
                "player2_entry": player2_entry or None,
            },
            export_path=EXPORT_PATH,
            metrics_path=METRICS_PATH,
            advanced_report_path=ADVANCED_REPORT_PATH,
            odds=odds_payload,
        ).to_dict()

        p1_pct = prediction["p1_probability"] * 100
        p2_pct = prediction["p2_probability"] * 100
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"{player1} win probability", f"{p1_pct:.1f}%")
        k2.metric(f"{player2} win probability", f"{p2_pct:.1f}%")
        k3.metric("Pick", prediction["pick"])
        k4.metric("Confidence", prediction["confidence"])

        if prediction.get("market_p1_probability") is not None:
            st.metric("Model edge on Player 1 vs no-vig market", f"{prediction['edge_p1'] * 100:+.2f}%")

        left, right = st.columns(2)
        with left:
            st.markdown("**Reason breakdown**")
            for reason in prediction.get("reasons", []):
                st.write("-", reason)
            if prediction.get("risk_flags"):
                st.markdown("**Risk flags**")
                for flag in prediction["risk_flags"]:
                    st.warning(flag)
        with right:
            st.markdown("**Caveats / data coverage**")
            for caveat in prediction.get("caveats", []):
                st.write("-", caveat)
            st.caption(f"Model source: {prediction['model_source']}")

        comparison = prediction.get("player_comparison", {})
        if comparison:
            st.subheader("Player comparison")
            st.dataframe(pd.DataFrame(comparison).T, use_container_width=True)
        feature_snapshot = prediction.get("feature_snapshot", {})
        if feature_snapshot:
            st.subheader("Live feature snapshot")
            f1, f2, f3 = st.columns(3)
            f1.json(feature_snapshot.get("elo", {}))
            f2.json(feature_snapshot.get("rolling_stats", {}))
            f3.json(feature_snapshot.get("fatigue", {}))
            st.caption("Entry context and calibrated routing are included in the raw payload below.")
        with st.expander("Raw prediction payload"):
            st.json(prediction)

with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current export rows", len(current))
    c2.metric("DB matches", counts.get("matches", 0))
    c3.metric("Player stat rows", counts.get("player_stat_rows", 0))
    date_range = counts.get("date_range") or (None, None)
    c4.metric("DB max date", date_range[1] or "n/a")

    if not current.empty:
        st.subheader("Coverage by tournament")
        show = current.groupby(["tourney_name", "match_date"], dropna=False).size().reset_index(name="rows")
        show = show.sort_values(["match_date", "tourney_name"], ascending=[False, True]).head(25)
        st.dataframe(show, use_container_width=True)
        st.subheader("Stat non-null coverage")
        stat_cols = [c for c in current.columns if any(x in c for x in ["serve", "return", "points", "minutes"])]
        coverage = pd.DataFrame({"column": stat_cols, "non_null": [int(current[c].notna().sum()) for c in stat_cols]})
        coverage["pct"] = (coverage["non_null"] / len(current) * 100).round(1)
        st.dataframe(coverage, use_container_width=True)

with tab_model:
    st.subheader("Latest daily research model")
    if not metrics_payload:
        st.warning("No model metrics yet. Run: `.venv/bin/python scripts/betting_research_pipeline.py`")
    else:
        st.json(metrics_payload.get("metrics", {}))
        st.write("Model artifact:", str(MODEL_PATH), "exists=" + str(MODEL_PATH.exists()))
        disclaimer = metrics_payload.get("disclaimer")
        if disclaimer:
            st.info(disclaimer)

with tab_edges:
    st.subheader("Flat-stake edge backtest vs Bet365 closing odds")
    strategies = metrics_payload.get("strategies", []) if metrics_payload else []
    if strategies:
        st.dataframe(pd.DataFrame(strategies), use_container_width=True)
        best = metrics_payload.get("best_strategy_by_profit", {})
        st.warning(
            "Current test result: best threshold still lost money "
            f"(profit={best.get('profit')}, ROI={best.get('roi')}). Do not deploy this as a betting system."
        )
    if not preds.empty:
        edge_min = st.slider("Minimum model edge", 0.0, 0.30, 0.04, 0.01)
        subset = preds[preds["edge"] >= edge_min].copy()
        st.metric("Qualifying historical bets", len(subset))
        if not subset.empty:
            subset["flat_profit"] = subset.apply(lambda r: r["p1_odds"] - 1 if r["result"] == 1 else -1, axis=1)
            st.metric("Flat-stake ROI", f"{subset['flat_profit'].sum() / len(subset):.2%}")
            st.dataframe(
                subset[["date", "tournament", "surface", "round", "player1", "player2", "p1_odds", "model_p1", "implied_p1_no_vig", "edge", "result", "flat_profit"]]
                .sort_values("edge", ascending=False)
                .head(200),
                use_container_width=True,
            )

with tab_niches:
    st.subheader("Walk-forward niche / submarket research")
    if not niche_report:
        st.warning("No niche research output yet. Run: `.venv/bin/python scripts/niche_submarket_research.py`")
    else:
        st.write("Generated:", niche_report.get("generated_at"))
        st.write("Test years:", niche_report.get("test_years"))
        robust = niche_report.get("robust_positive_segments", [])
        if robust:
            st.success(f"Found {len(robust)} robust-positive candidate segment(s). Treat as hypotheses only.")
            st.dataframe(pd.DataFrame(robust), use_container_width=True)
        else:
            st.warning("No robust positive segment passed the current filters. That is useful evidence: broad edge is not confirmed yet.")
        yearly = niche_report.get("yearly_metrics", [])
        if yearly:
            st.subheader("Yearly walk-forward model vs market")
            st.dataframe(pd.DataFrame(yearly), use_container_width=True)
        overall = niche_report.get("overall_threshold_results", [])
        if overall:
            st.subheader("Overall candidate-side edge thresholds")
            st.dataframe(pd.DataFrame(overall), use_container_width=True)
        if not niche_segments.empty:
            st.subheader("Top scanned submarkets")
            st.dataframe(niche_segments.head(200), use_container_width=True)
        disclaimer = niche_report.get("disclaimer")
        if disclaimer:
            st.info(disclaimer)

with tab_rows:
    st.subheader("Current enriched 2026 export")
    if current.empty:
        st.warning("No current export found.")
    else:
        tournaments = ["All"] + sorted(current["tourney_name"].dropna().unique().tolist())
        selected = st.selectbox("Tournament", tournaments)
        table = current if selected == "All" else current[current["tourney_name"] == selected]
        st.dataframe(table.tail(500), use_container_width=True)
