#!/usr/bin/env python3
"""Advanced no-lookahead tennis model research.

Implements the requested feature/model experiments:
1. Elo family features: overall, surface, Slam/best-of-five, recent-ish form via current Elo deltas.
2. Market-aware model: includes no-vig market probability as a feature.
3. Grand Slam context: best-of-five/Slam experience and Slam Elo.
4. Surface skill: surface Elo and rolling surface records.
5. Top-player context: top-10/top-20 flags and rolling records vs top players.
6. Early-round weakness: prior-week title/final, rest, recent match volume, tournament fatigue, surface switch.
7. Clustering: KMeans clusters of OOS rows where global advanced model underperforms market.

Research only. No betting execution.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from betting_research_pipeline import american_safe_float, load_odds_data  # noqa: E402
from niche_submarket_research import add_candidate_columns  # noqa: E402

OUT_DIR = ROOT / "data" / "betting_research"


def safe_logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(ra: float, rb: float, score_a: float, k: float = 28.0) -> tuple[float, float]:
    ea = expected_score(ra, rb)
    eb = 1.0 - ea
    return ra + k * (score_a - ea), rb + k * ((1.0 - score_a) - eb)


def is_final(round_value: str) -> bool:
    return str(round_value).lower() in {"the final", "final", "f"}


def round_group(r: str) -> str:
    r = str(r)
    if r in {"1st Round", "Round Robin", "R128", "R64", "R32"}:
        return "early"
    if r in {"2nd Round", "3rd Round", "4th Round"}:
        return "middle"
    if r in {"Quarterfinals", "Semifinals", "The Final", "QF", "SF", "F"}:
        return "late"
    return r


def sum_games(m: pd.Series) -> float:
    total = 0.0
    for i in range(1, 6):
        w = pd.to_numeric(m.get(f"W{i}"), errors="coerce")
        l = pd.to_numeric(m.get(f"L{i}"), errors="coerce")
        if pd.notna(w):
            total += float(w)
        if pd.notna(l):
            total += float(l)
    return total if total > 0 else np.nan


def game_totals(m: pd.Series) -> tuple[float, float, int]:
    """Winner games, loser games, and tiebreak-like set count from completed score columns."""
    wg = lg = 0.0
    tb = 0
    for i in range(1, 6):
        w = pd.to_numeric(m.get(f"W{i}"), errors="coerce")
        l = pd.to_numeric(m.get(f"L{i}"), errors="coerce")
        if pd.notna(w) and pd.notna(l):
            wg += float(w)
            lg += float(l)
            if max(float(w), float(l)) >= 7 and min(float(w), float(l)) >= 5:
                tb += 1
    return (wg if wg else np.nan, lg if lg else np.nan, tb)


# Small built-in location map for schedule/travel features. Unknown events safely fall back to 0 distance.
LOCATION_META = {
    "Melbourne": (-37.8136, 144.9631, "Australia", "Oceania"), "Paris": (48.8566, 2.3522, "France", "Europe"),
    "London": (51.5072, -0.1276, "United Kingdom", "Europe"), "New York": (40.7128, -74.0060, "United States", "North America"),
    "Indian Wells": (33.7175, -116.3408, "United States", "North America"), "Miami": (25.7617, -80.1918, "United States", "North America"),
    "Monte Carlo": (43.7384, 7.4246, "Monaco", "Europe"), "Madrid": (40.4168, -3.7038, "Spain", "Europe"),
    "Rome": (41.9028, 12.4964, "Italy", "Europe"), "Cincinnati": (39.1031, -84.5120, "United States", "North America"),
    "Toronto": (43.6532, -79.3832, "Canada", "North America"), "Montreal": (45.5019, -73.5674, "Canada", "North America"),
    "Shanghai": (31.2304, 121.4737, "China", "Asia"), "Doha": (25.2854, 51.5310, "Qatar", "Asia"),
    "Dubai": (25.2048, 55.2708, "United Arab Emirates", "Asia"), "Acapulco": (16.8531, -99.8237, "Mexico", "North America"),
    "Barcelona": (41.3874, 2.1686, "Spain", "Europe"), "Halle": (51.4969, 11.9688, "Germany", "Europe"),
    "Queens Club": (51.4876, -0.2107, "United Kingdom", "Europe"), "Washington": (38.9072, -77.0369, "United States", "North America"),
    "Tokyo": (35.6762, 139.6503, "Japan", "Asia"), "Beijing": (39.9042, 116.4074, "China", "Asia"),
}


def location_meta(location: str) -> tuple[float, float, str, str]:
    return LOCATION_META.get(str(location), (np.nan, np.nan, "Unknown", "Unknown"))


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    if any(pd.isna(v) for v in [a_lat, a_lon, b_lat, b_lon]):
        return 0.0
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(x))


@dataclass
class AdvState:
    wins: int = 0
    matches: int = 0
    last5: deque = field(default_factory=lambda: deque(maxlen=5))
    match_dates: deque = field(default_factory=lambda: deque(maxlen=30))
    surface_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    slam_wins: int = 0
    slam_matches: int = 0
    bo5_wins: int = 0
    bo5_matches: int = 0
    vs_top10_wins: int = 0
    vs_top10_matches: int = 0
    vs_top20_wins: int = 0
    vs_top20_matches: int = 0
    elo: float = 1500.0
    surface_elo: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    slam_elo: float = 1500.0
    last_match_date: pd.Timestamp | None = None
    last_surface: str | None = None
    last_tourney: str | None = None
    last_location: str | None = None
    last_country: str | None = None
    last_continent: str | None = None
    last_title_date: pd.Timestamp | None = None
    last_final_date: pd.Timestamp | None = None
    last_retired_date: pd.Timestamp | None = None
    sets_won: float = 0.0
    sets_lost: float = 0.0
    games_won: float = 0.0
    games_lost: float = 0.0
    tiebreak_sets: int = 0
    straight_set_wins: int = 0

    def pct(self, wins: int, matches: int, default: float = 0.5) -> float:
        return wins / matches if matches else default

    def overall_pct(self) -> float:
        return self.pct(self.wins, self.matches)

    def last5_pct(self) -> float:
        return sum(self.last5) / len(self.last5) if self.last5 else 0.5

    def surface_pct(self, surface: str) -> float:
        return self.pct(self.surface_wins[surface], self.surface_matches[surface])

    def slam_pct(self) -> float:
        return self.pct(self.slam_wins, self.slam_matches)

    def bo5_pct(self) -> float:
        return self.pct(self.bo5_wins, self.bo5_matches)

    def vs_top10_pct(self) -> float:
        return self.pct(self.vs_top10_wins, self.vs_top10_matches)

    def vs_top20_pct(self) -> float:
        return self.pct(self.vs_top20_wins, self.vs_top20_matches)

    def days_since(self, date: pd.Timestamp) -> float:
        if self.last_match_date is None:
            return np.nan
        return float((date - self.last_match_date).days)

    def matches_last(self, date: pd.Timestamp, days: int) -> int:
        cutoff = date - pd.Timedelta(days=days)
        return int(sum(d >= cutoff for d in self.match_dates))

    def days_since_title(self, date: pd.Timestamp) -> float:
        if self.last_title_date is None:
            return np.nan
        return float((date - self.last_title_date).days)

    def days_since_final(self, date: pd.Timestamp) -> float:
        if self.last_final_date is None:
            return np.nan
        return float((date - self.last_final_date).days)

    def days_since_retirement(self, date: pd.Timestamp) -> float:
        if self.last_retired_date is None:
            return np.nan
        return float((date - self.last_retired_date).days)

    def set_win_pct(self) -> float:
        total = self.sets_won + self.sets_lost
        return self.sets_won / total if total else 0.5

    def game_win_pct(self) -> float:
        total = self.games_won + self.games_lost
        return self.games_won / total if total else 0.5

    def tiebreak_rate(self) -> float:
        return self.tiebreak_sets / self.matches if self.matches else 0.0

    def straight_set_win_rate(self) -> float:
        return self.straight_set_wins / self.matches if self.matches else 0.0

    def travel_from_last(self, location: str) -> tuple[float, int, int]:
        if self.last_location is None:
            return 0.0, 0, 0
        lat1, lon1, country1, cont1 = location_meta(self.last_location)
        lat2, lon2, country2, cont2 = location_meta(location)
        return haversine_km(lat1, lon1, lat2, lon2), int(country1 != country2), int(cont1 != cont2)

    def update(self, won: int, date: pd.Timestamp, surface: str, series: str, best_of: float, opp_rank: float, round_value: str, tourney: str, location: str, sets_for: float, sets_against: float, games_for: float, games_against: float, tiebreaks: int, retired: bool):
        won = int(won)
        is_slam = str(series) == "Grand Slam"
        is_bo5 = bool(pd.notna(best_of) and float(best_of) >= 5)
        self.matches += 1
        self.wins += won
        self.last5.append(won)
        self.match_dates.append(date)
        self.surface_matches[surface] += 1
        self.surface_wins[surface] += won
        if is_slam:
            self.slam_matches += 1
            self.slam_wins += won
        if is_bo5:
            self.bo5_matches += 1
            self.bo5_wins += won
        if pd.notna(opp_rank) and opp_rank <= 10:
            self.vs_top10_matches += 1
            self.vs_top10_wins += won
        if pd.notna(opp_rank) and opp_rank <= 20:
            self.vs_top20_matches += 1
            self.vs_top20_wins += won
        if is_final(round_value):
            self.last_final_date = date
            if won:
                self.last_title_date = date
        if retired:
            self.last_retired_date = date
        if pd.notna(sets_for):
            self.sets_won += float(sets_for)
        if pd.notna(sets_against):
            self.sets_lost += float(sets_against)
        if pd.notna(games_for):
            self.games_won += float(games_for)
        if pd.notna(games_against):
            self.games_lost += float(games_against)
        self.tiebreak_sets += int(tiebreaks or 0)
        if won and pd.notna(sets_against) and float(sets_against) == 0:
            self.straight_set_wins += 1
        _, _, country, continent = location_meta(location)
        self.last_match_date = date
        self.last_surface = surface
        self.last_tourney = tourney
        self.last_location = location
        self.last_country = country
        self.last_continent = continent


def add_advanced_side_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.sort_values(["Date", "Tournament", "Round", "Winner", "Loser"]).reset_index(drop=True)
    states: dict[str, AdvState] = {}
    h2h: dict[tuple[str, str], list[int]] = {}
    tournament_load: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    tournament_history: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: {"matches": 0.0, "favorite_wins": 0.0, "upsets": 0.0})
    rows = []

    def st(name: str) -> AdvState:
        if name not in states:
            states[name] = AdvState()
        return states[name]

    def tload(player: str, tourney: str) -> dict[str, float]:
        return tournament_load[(player, tourney)]

    for date, day in df.groupby("Date", sort=True):
        pending = []
        date = pd.Timestamp(date)
        for _, m in day.iterrows():
            winner, loser = str(m["Winner"]), str(m["Loser"])
            surface = str(m.get("Surface", "Unknown"))
            series = str(m.get("Series", "Unknown"))
            court = str(m.get("Court", "Unknown"))
            round_value = str(m.get("Round", "Unknown"))
            tourney = str(m.get("Tournament", "Unknown"))
            location = str(m.get("Location", "Unknown"))
            best_of = american_safe_float(m.get("Best of"))
            w_rank = american_safe_float(m.get("WRank"))
            l_rank = american_safe_float(m.get("LRank"))
            w_pts = american_safe_float(m.get("WPts"))
            l_pts = american_safe_float(m.get("LPts"))
            wsets = american_safe_float(m.get("Wsets"))
            lsets = american_safe_float(m.get("Lsets"))
            games = sum_games(m)
            w_games, l_games, tiebreaks = game_totals(m)
            comment = str(m.get("Comment", ""))
            retired = "retired" in comment.lower() or "walkover" in comment.lower()
            loc_lat, loc_lon, loc_country, loc_continent = location_meta(location)
            th = tournament_history[(tourney, surface, round_group(round_value))]
            tournament_favorite_win_rate = th["favorite_wins"] / th["matches"] if th["matches"] else 0.5
            tournament_upset_rate = th["upsets"] / th["matches"] if th["matches"] else 0.5
            swap = bool(rng.random() < 0.5)
            p1, p2 = (winner, loser) if not swap else (loser, winner)
            p1_is_winner = int(p1 == winner)
            p1_rank = w_rank if p1 == winner else l_rank
            p2_rank = l_rank if p1 == winner else w_rank
            p1_pts = w_pts if p1 == winner else l_pts
            p2_pts = l_pts if p1 == winner else w_pts
            p1_odds = float(m["B365W"] if p1 == winner else m["B365L"])
            p2_odds = float(m["B365L"] if p1 == winner else m["B365W"])
            imp1_raw, imp2_raw = 1.0 / p1_odds, 1.0 / p2_odds
            implied_p1_no_vig = imp1_raw / (imp1_raw + imp2_raw)
            s1, s2 = st(p1), st(p2)
            a, b = sorted([p1, p2])
            hh = h2h.get((a, b), [0, 0])
            h2h_p1, h2h_p2 = (hh[0], hh[1]) if a == p1 else (hh[1], hh[0])
            tl1, tl2 = tload(p1, tourney), tload(p2, tourney)
            p1_days_title = s1.days_since_title(date)
            p2_days_title = s2.days_since_title(date)
            p1_days_final = s1.days_since_final(date)
            p2_days_final = s2.days_since_final(date)
            p1_days_rest = s1.days_since(date)
            p2_days_rest = s2.days_since(date)
            p1_travel_km, p1_country_switch, p1_continent_switch = s1.travel_from_last(location)
            p2_travel_km, p2_country_switch, p2_continent_switch = s2.travel_from_last(location)
            p1_days_retired = s1.days_since_retirement(date)
            p2_days_retired = s2.days_since_retirement(date)
            is_early = int(round_group(round_value) == "early")
            row = {
                "date": date,
                "year": int(date.year),
                "tournament": tourney,
                "location": location,
                "location_country": loc_country,
                "location_continent": loc_continent,
                "location_lat": loc_lat,
                "location_lon": loc_lon,
                "surface": surface,
                "series": series,
                "court": court,
                "round": round_value,
                "round_group": round_group(round_value),
                "best_of": best_of,
                "is_grand_slam": int(series == "Grand Slam"),
                "is_best_of_5": int(pd.notna(best_of) and float(best_of) >= 5),
                "is_early_round": is_early,
                "player1": p1,
                "player2": p2,
                "p1_odds": p1_odds,
                "p2_odds": p2_odds,
                "implied_p1_no_vig": implied_p1_no_vig,
                "market_logit_p1": safe_logit(implied_p1_no_vig),
                "result": p1_is_winner,
                "p1_rank": p1_rank,
                "p2_rank": p2_rank,
                "p1_pts": p1_pts,
                "p2_pts": p2_pts,
                "rank_diff": p1_rank - p2_rank,
                "log_rank_diff": math.log1p(abs(p1_rank - p2_rank)) if pd.notna(p1_rank) and pd.notna(p2_rank) else np.nan,
                "points_ratio": math.log1p(p1_pts) - math.log1p(p2_pts) if pd.notna(p1_pts) and pd.notna(p2_pts) else np.nan,
                "p1_top10": int(pd.notna(p1_rank) and p1_rank <= 10),
                "p2_top10": int(pd.notna(p2_rank) and p2_rank <= 10),
                "p1_top20": int(pd.notna(p1_rank) and p1_rank <= 20),
                "p2_top20": int(pd.notna(p2_rank) and p2_rank <= 20),
                "any_top10": int((pd.notna(p1_rank) and p1_rank <= 10) or (pd.notna(p2_rank) and p2_rank <= 10)),
                "both_top10": int((pd.notna(p1_rank) and p1_rank <= 10) and (pd.notna(p2_rank) and p2_rank <= 10)),
                "p1_prior_matches": s1.matches,
                "p2_prior_matches": s2.matches,
                "prior_match_diff": s1.matches - s2.matches,
                "p1_prior_win_pct": s1.overall_pct(),
                "p2_prior_win_pct": s2.overall_pct(),
                "prior_win_pct_diff": s1.overall_pct() - s2.overall_pct(),
                "p1_last5_pct": s1.last5_pct(),
                "p2_last5_pct": s2.last5_pct(),
                "last5_diff": s1.last5_pct() - s2.last5_pct(),
                "p1_surface_pct": s1.surface_pct(surface),
                "p2_surface_pct": s2.surface_pct(surface),
                "surface_pct_diff": s1.surface_pct(surface) - s2.surface_pct(surface),
                "p1_slam_pct": s1.slam_pct(),
                "p2_slam_pct": s2.slam_pct(),
                "slam_pct_diff": s1.slam_pct() - s2.slam_pct(),
                "p1_bo5_pct": s1.bo5_pct(),
                "p2_bo5_pct": s2.bo5_pct(),
                "bo5_pct_diff": s1.bo5_pct() - s2.bo5_pct(),
                "p1_vs_top10_pct": s1.vs_top10_pct(),
                "p2_vs_top10_pct": s2.vs_top10_pct(),
                "vs_top10_pct_diff": s1.vs_top10_pct() - s2.vs_top10_pct(),
                "p1_vs_top20_pct": s1.vs_top20_pct(),
                "p2_vs_top20_pct": s2.vs_top20_pct(),
                "vs_top20_pct_diff": s1.vs_top20_pct() - s2.vs_top20_pct(),
                "p1_elo": s1.elo,
                "p2_elo": s2.elo,
                "elo_diff": s1.elo - s2.elo,
                "p1_surface_elo": s1.surface_elo[surface],
                "p2_surface_elo": s2.surface_elo[surface],
                "surface_elo_diff": s1.surface_elo[surface] - s2.surface_elo[surface],
                "p1_slam_elo": s1.slam_elo,
                "p2_slam_elo": s2.slam_elo,
                "slam_elo_diff": s1.slam_elo - s2.slam_elo,
                # Statistically enhanced ability covariates: pre-match estimates from historical state only.
                "surface_ability_diff": (s1.surface_elo[surface] - s2.surface_elo[surface]) + 250.0 * (s1.surface_pct(surface) - s2.surface_pct(surface)),
                "serve_return_ability_diff": (s1.game_win_pct() - s2.game_win_pct()) * 400.0,
                "bo5_ability_diff": (s1.slam_elo - s2.slam_elo) + 250.0 * (s1.bo5_pct() - s2.bo5_pct()),
                "recent_form_ability_diff": (s1.last5_pct() - s2.last5_pct()) * 250.0 + 0.35 * (s1.elo - s2.elo),
                "fatigue_adjusted_ability_diff": (s1.elo - s2.elo) - 12.0 * (s1.matches_last(date, 14) - s2.matches_last(date, 14)) - 0.003 * (p1_travel_km - p2_travel_km),
                "h2h_diff": h2h_p1 - h2h_p2,
                "p1_days_rest": p1_days_rest,
                "p2_days_rest": p2_days_rest,
                "rest_diff": (p1_days_rest if pd.notna(p1_days_rest) else 30) - (p2_days_rest if pd.notna(p2_days_rest) else 30),
                "p1_matches_last7": s1.matches_last(date, 7),
                "p2_matches_last7": s2.matches_last(date, 7),
                "matches_last7_diff": s1.matches_last(date, 7) - s2.matches_last(date, 7),
                "p1_matches_last14": s1.matches_last(date, 14),
                "p2_matches_last14": s2.matches_last(date, 14),
                "matches_last14_diff": s1.matches_last(date, 14) - s2.matches_last(date, 14),
                "p1_days_since_title": p1_days_title,
                "p2_days_since_title": p2_days_title,
                "days_since_title_diff": (p1_days_title if pd.notna(p1_days_title) else 90) - (p2_days_title if pd.notna(p2_days_title) else 90),
                "p1_title_within_14": int(pd.notna(p1_days_title) and 0 < p1_days_title <= 14),
                "p2_title_within_14": int(pd.notna(p2_days_title) and 0 < p2_days_title <= 14),
                "title_within_14_diff": int(pd.notna(p1_days_title) and 0 < p1_days_title <= 14) - int(pd.notna(p2_days_title) and 0 < p2_days_title <= 14),
                "p1_final_within_7": int(pd.notna(p1_days_final) and 0 < p1_days_final <= 7),
                "p2_final_within_7": int(pd.notna(p2_days_final) and 0 < p2_days_final <= 7),
                "final_within_7_diff": int(pd.notna(p1_days_final) and 0 < p1_days_final <= 7) - int(pd.notna(p2_days_final) and 0 < p2_days_final <= 7),
                "p1_surface_switch": int(s1.last_surface is not None and s1.last_surface != surface),
                "p2_surface_switch": int(s2.last_surface is not None and s2.last_surface != surface),
                "surface_switch_diff": int(s1.last_surface is not None and s1.last_surface != surface) - int(s2.last_surface is not None and s2.last_surface != surface),
                "p1_tourney_sets_before": tl1.get("sets", 0.0),
                "p2_tourney_sets_before": tl2.get("sets", 0.0),
                "tourney_sets_before_diff": tl1.get("sets", 0.0) - tl2.get("sets", 0.0),
                "p1_tourney_games_before": tl1.get("games", 0.0),
                "p2_tourney_games_before": tl2.get("games", 0.0),
                "tourney_games_before_diff": tl1.get("games", 0.0) - tl2.get("games", 0.0),
                "p1_tourney_matches_before": tl1.get("matches", 0.0),
                "p2_tourney_matches_before": tl2.get("matches", 0.0),
                "tourney_matches_before_diff": tl1.get("matches", 0.0) - tl2.get("matches", 0.0),
                "early_after_title_p1": int(is_early and pd.notna(p1_days_title) and 0 < p1_days_title <= 14),
                "early_after_title_p2": int(is_early and pd.notna(p2_days_title) and 0 < p2_days_title <= 14),
                # Travel/schedule features.
                "p1_travel_km": p1_travel_km,
                "p2_travel_km": p2_travel_km,
                "travel_km_diff": p1_travel_km - p2_travel_km,
                "p1_country_switch": p1_country_switch,
                "p2_country_switch": p2_country_switch,
                "country_switch_diff": p1_country_switch - p2_country_switch,
                "p1_continent_switch": p1_continent_switch,
                "p2_continent_switch": p2_continent_switch,
                "continent_switch_diff": p1_continent_switch - p2_continent_switch,
                "p1_title_and_continent_switch": int(pd.notna(p1_days_title) and 0 < p1_days_title <= 14 and p1_continent_switch),
                "p2_title_and_continent_switch": int(pd.notna(p2_days_title) and 0 < p2_days_title <= 14 and p2_continent_switch),
                # Entry-context placeholders. Fill from external draw/entry data when available.
                "p1_qualifier": 0, "p2_qualifier": 0, "qualifier_diff": 0,
                "p1_wildcard": 0, "p2_wildcard": 0, "wildcard_diff": 0,
                "p1_lucky_loser": 0, "p2_lucky_loser": 0, "lucky_loser_diff": 0,
                "p1_protected_ranking": 0, "p2_protected_ranking": 0, "protected_ranking_diff": 0,
                # Challenger/lower-tour placeholders. Fill from Challenger ETL when available.
                "p1_challenger_form": 0.5, "p2_challenger_form": 0.5, "challenger_form_diff": 0.0,
                "p1_challenger_title_30d": 0, "p2_challenger_title_30d": 0, "challenger_title_30d_diff": 0,
                # Injury/withdrawal proxy features from prior comments and layoffs.
                "p1_retired_within_30": int(pd.notna(p1_days_retired) and 0 < p1_days_retired <= 30),
                "p2_retired_within_30": int(pd.notna(p2_days_retired) and 0 < p2_days_retired <= 30),
                "retired_within_30_diff": int(pd.notna(p1_days_retired) and 0 < p1_days_retired <= 30) - int(pd.notna(p2_days_retired) and 0 < p2_days_retired <= 30),
                "p1_long_layoff_45": int(pd.notna(p1_days_rest) and p1_days_rest >= 45),
                "p2_long_layoff_45": int(pd.notna(p2_days_rest) and p2_days_rest >= 45),
                "long_layoff_45_diff": int(pd.notna(p1_days_rest) and p1_days_rest >= 45) - int(pd.notna(p2_days_rest) and p2_days_rest >= 45),
                # Style/matchup proxies from rolling score outcomes.
                "set_win_pct_diff": s1.set_win_pct() - s2.set_win_pct(),
                "game_win_pct_diff": s1.game_win_pct() - s2.game_win_pct(),
                "tiebreak_rate_diff": s1.tiebreak_rate() - s2.tiebreak_rate(),
                "straight_set_win_rate_diff": s1.straight_set_win_rate() - s2.straight_set_win_rate(),
                # Tournament-specific history.
                "tournament_favorite_win_rate": tournament_favorite_win_rate,
                "tournament_upset_rate": tournament_upset_rate,
            }
            rows.append(row)
            pending.append((winner, loser, surface, series, best_of, w_rank, l_rank, round_value, tourney, location, wsets, lsets, games, w_games, l_games, tiebreaks, retired, p1_odds, p2_odds, p1_is_winner, swap))
        # End-of-date updates: avoids same-day leakage.
        for winner, loser, surface, series, best_of, w_rank, l_rank, round_value, tourney, location, wsets, lsets, games, w_games, l_games, tiebreaks, retired, p1_odds, p2_odds, p1_is_winner, swap in pending:
            sw, sl = st(winner), st(loser)
            # Elo updates read pre-update ratings.
            sw.elo, sl.elo = update_elo(sw.elo, sl.elo, 1.0)
            sw.surface_elo[surface], sl.surface_elo[surface] = update_elo(sw.surface_elo[surface], sl.surface_elo[surface], 1.0, k=30.0)
            if str(series) == "Grand Slam" or (pd.notna(best_of) and float(best_of) >= 5):
                sw.slam_elo, sl.slam_elo = update_elo(sw.slam_elo, sl.slam_elo, 1.0, k=24.0)
            sw.update(1, date, surface, series, best_of, l_rank, round_value, tourney, location, wsets, lsets, w_games, l_games, tiebreaks, retired)
            sl.update(0, date, surface, series, best_of, w_rank, round_value, tourney, location, lsets, wsets, l_games, w_games, tiebreaks, retired)
            a, b = sorted([winner, loser])
            hh = h2h.setdefault((a, b), [0, 0])
            if winner == a:
                hh[0] += 1
            else:
                hh[1] += 1
            hist = tournament_history[(tourney, surface, round_group(round_value))]
            p1_favorite = p1_odds < p2_odds
            favorite_won = (p1_favorite and p1_is_winner == 1) or ((not p1_favorite) and p1_is_winner == 0)
            hist["matches"] += 1.0
            hist["favorite_wins"] += float(favorite_won)
            hist["upsets"] += float(not favorite_won)
            # Tournament cumulative fatigue/load is only for future dates.
            for player, sets in [(winner, wsets), (loser, lsets)]:
                load = tload(player, tourney)
                load["matches"] += 1.0
                if pd.notna(sets):
                    load["sets"] += float(sets)
                if pd.notna(games):
                    load["games"] += float(games)
    return pd.DataFrame(rows)


BASE_NUMERIC = [
    "rank_diff", "log_rank_diff", "points_ratio", "p1_top10", "p2_top10", "any_top10", "both_top10",
    "prior_win_pct_diff", "last5_diff", "surface_pct_diff", "h2h_diff", "p1_prior_matches", "p2_prior_matches",
    "p1_prior_win_pct", "p2_prior_win_pct", "p1_last5_pct", "p2_last5_pct", "p1_surface_pct", "p2_surface_pct",
]
ADV_NUMERIC = BASE_NUMERIC + [
    "p1_top20", "p2_top20", "is_grand_slam", "is_best_of_5", "is_early_round", "best_of",
    "prior_match_diff", "slam_pct_diff", "bo5_pct_diff", "vs_top10_pct_diff", "vs_top20_pct_diff",
    "p1_slam_pct", "p2_slam_pct", "p1_bo5_pct", "p2_bo5_pct", "p1_vs_top10_pct", "p2_vs_top10_pct",
    "elo_diff", "surface_elo_diff", "slam_elo_diff", "surface_ability_diff", "serve_return_ability_diff", "bo5_ability_diff",
    "recent_form_ability_diff", "fatigue_adjusted_ability_diff", "p1_elo", "p2_elo", "p1_surface_elo", "p2_surface_elo",
    "rest_diff", "matches_last7_diff", "matches_last14_diff", "p1_matches_last7", "p2_matches_last7",
    "p1_matches_last14", "p2_matches_last14", "days_since_title_diff", "title_within_14_diff", "final_within_7_diff",
    "p1_title_within_14", "p2_title_within_14", "p1_final_within_7", "p2_final_within_7",
    "surface_switch_diff", "p1_surface_switch", "p2_surface_switch", "tourney_sets_before_diff",
    "tourney_games_before_diff", "tourney_matches_before_diff", "p1_tourney_matches_before", "p2_tourney_matches_before",
    "early_after_title_p1", "early_after_title_p2",
    "p1_travel_km", "p2_travel_km", "travel_km_diff", "p1_country_switch", "p2_country_switch", "country_switch_diff",
    "p1_continent_switch", "p2_continent_switch", "continent_switch_diff", "p1_title_and_continent_switch", "p2_title_and_continent_switch",
    "p1_qualifier", "p2_qualifier", "qualifier_diff", "p1_wildcard", "p2_wildcard", "wildcard_diff",
    "p1_lucky_loser", "p2_lucky_loser", "lucky_loser_diff", "p1_protected_ranking", "p2_protected_ranking", "protected_ranking_diff",
    "p1_challenger_form", "p2_challenger_form", "challenger_form_diff", "p1_challenger_title_30d", "p2_challenger_title_30d", "challenger_title_30d_diff",
    "p1_retired_within_30", "p2_retired_within_30", "retired_within_30_diff", "p1_long_layoff_45", "p2_long_layoff_45", "long_layoff_45_diff",
    "set_win_pct_diff", "game_win_pct_diff", "tiebreak_rate_diff", "straight_set_win_rate_diff", "tournament_favorite_win_rate", "tournament_upset_rate",
]
MARKET_NUMERIC = ADV_NUMERIC + ["implied_p1_no_vig", "market_logit_p1"]
CATEGORICAL = ["surface", "series", "court", "round", "round_group"]


def build_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    base = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, C=0.7, random_state=42)),
            ("hgb", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.045, l2_regularization=0.08, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=220, max_depth=9, min_samples_leaf=8, random_state=42, n_jobs=-1)),
        ],
        voting="soft",
    )
    return Pipeline([("features", pre), ("model", base)])


def build_residual_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    base = VotingRegressor(
        estimators=[
            ("enet", ElasticNet(alpha=0.002, l1_ratio=0.08, random_state=42, max_iter=5000)),
            ("hgb", HistGradientBoostingRegressor(max_iter=180, learning_rate=0.035, l2_regularization=0.12, random_state=42)),
            ("rf", RandomForestRegressor(n_estimators=180, max_depth=7, min_samples_leaf=12, random_state=42, n_jobs=-1)),
        ],
    )
    return Pipeline([("features", pre), ("model", base)])


def specialist_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    title_or_final = (
        df["p1_title_within_14"].eq(1)
        | df["p2_title_within_14"].eq(1)
        | df["p1_final_within_7"].eq(1)
        | df["p2_final_within_7"].eq(1)
    )
    surface_switch = df["p1_surface_switch"].eq(1) | df["p2_surface_switch"].eq(1)
    return {
        "early_atp250": df["round_group"].eq("early") & df["series"].eq("ATP250"),
        "post_title_or_final": title_or_final,
        "surface_switch": surface_switch,
        "grand_slam": df["is_grand_slam"].eq(1),
        "top10_match": df["any_top10"].eq(1),
        "early_surface_switch": df["round_group"].eq("early") & surface_switch,
    }


def fit_residual_overlay(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    base_col: str = "implied_p1_no_vig",
    shrink: float = 0.55,
    min_specialist_rows: int = 350,
    segment_shrink: dict[str, float] | None = None,
) -> tuple[pd.Series, dict[str, dict]]:
    """Predict residuals vs market, with specialist models replacing global residuals in known weak buckets."""
    train = train.copy()
    test = test.copy()
    train["market_residual"] = train["result"].astype(float) - train[base_col].astype(float)
    global_model = build_residual_model(MARKET_NUMERIC, CATEGORICAL)
    global_model.fit(train[features], train["market_residual"])
    global_resid = pd.Series(global_model.predict(test[features]), index=test.index).clip(-0.20, 0.20)
    final_resid = global_resid.copy()
    shrink_series = pd.Series(shrink, index=test.index, dtype=float)
    segment_shrink = segment_shrink or {}
    diagnostics: dict[str, dict] = {
        "global": {"train_rows": int(len(train)), "test_rows": int(len(test)), "shrink": shrink}
    }
    train_masks = specialist_masks(train)
    test_masks = specialist_masks(test)
    for name, tr_mask in train_masks.items():
        te_mask = test_masks[name]
        n_train = int(tr_mask.sum())
        n_test = int(te_mask.sum())
        if n_train < min_specialist_rows or n_test == 0:
            diagnostics[name] = {"used": False, "train_rows": n_train, "test_rows": n_test}
            continue
        specialist = build_residual_model(MARKET_NUMERIC, CATEGORICAL)
        specialist.fit(train.loc[tr_mask, features], train.loc[tr_mask, "market_residual"])
        spec_resid = pd.Series(specialist.predict(test.loc[te_mask, features]), index=test.loc[te_mask].index).clip(-0.20, 0.20)
        # Weight grows with segment sample size; still shrink to avoid threshold-mined overfit.
        segment_weight = min(0.80, n_train / (n_train + 1200.0))
        final_resid.loc[te_mask] = (1 - segment_weight) * final_resid.loc[te_mask] + segment_weight * spec_resid
        if name in segment_shrink:
            shrink_series.loc[te_mask] = float(segment_shrink[name])
        diagnostics[name] = {
            "used": True,
            "train_rows": n_train,
            "test_rows": n_test,
            "segment_weight": float(segment_weight),
            "segment_shrink": float(segment_shrink.get(name, shrink)),
            "avg_predicted_residual": float(spec_resid.mean()),
        }
    pred = (test[base_col].astype(float) + shrink_series * final_resid).clip(0.03, 0.97)
    return pred, diagnostics


def metrics_for(df: pd.DataFrame, prob_col: str) -> dict:
    y = df["result"].astype(int)
    p = df[prob_col].clip(1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    return {
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "market_log_loss": float(log_loss(y, df["implied_p1_no_vig"].clip(1e-6, 1 - 1e-6), labels=[0, 1])),
        "market_brier": float(brier_score_loss(y, df["implied_p1_no_vig"].clip(1e-6, 1 - 1e-6))),
        "mean_prob": float(p.mean()),
        "actual_rate": float(y.mean()),
    }


def calibration_bins(df: pd.DataFrame, prob_col: str, bins: int = 10) -> list[dict]:
    """Return reliability-bin diagnostics for a probability column.

    Positive calibration_error means the model over-predicted p1 wins in that bin;
    negative means the observed p1 win rate was higher than predicted.
    """
    tmp = df[["result", prob_col]].copy()
    tmp[prob_col] = tmp[prob_col].clip(0.0, 1.0)
    tmp["bin"] = np.minimum((tmp[prob_col] * bins).astype(int), bins - 1)
    rows = []
    for bin_id, g in tmp.groupby("bin", sort=True):
        mean_prob = float(g[prob_col].mean())
        actual_rate = float(g["result"].astype(float).mean())
        rows.append({
            "bin": int(bin_id),
            "prob_min": float(bin_id / bins),
            "prob_max": float((bin_id + 1) / bins),
            "rows": int(len(g)),
            "mean_prob": mean_prob,
            "actual_rate": actual_rate,
            "calibration_error": float(mean_prob - actual_rate),
        })
    return rows


def calibration_error_metrics(bin_rows: list[dict]) -> dict:
    """Return compact weighted ECE/MCE metrics from reliability-bin rows.

    ECE is the row-weighted mean absolute calibration error across non-empty bins;
    MCE is the largest absolute bin error. These are diagnostics only and should be
    read beside proper scoring rules such as log loss and Brier.
    """
    total_rows = int(sum(int(row.get("rows", 0)) for row in bin_rows))
    if total_rows <= 0:
        return {
            "rows": 0,
            "expected_calibration_error": None,
            "maximum_calibration_error": None,
        }
    weighted_abs_error = 0.0
    max_abs_error = 0.0
    for row in bin_rows:
        rows = int(row.get("rows", 0))
        if rows <= 0:
            continue
        abs_error = abs(float(row.get("calibration_error", 0.0)))
        weighted_abs_error += rows * abs_error
        max_abs_error = max(max_abs_error, abs_error)
    return {
        "rows": total_rows,
        "expected_calibration_error": float(weighted_abs_error / total_rows),
        "maximum_calibration_error": float(max_abs_error),
    }


def fit_bin_recalibration(
    train: pd.DataFrame,
    prob_col: str,
    bins: int = 10,
    min_rows: int = 80,
    shrink: float = 0.35,
) -> dict[int, dict]:
    """Fit a conservative no-lookahead bin-offset recalibrator on training rows.

    The learned offset is ``actual_rate - mean_prob`` per probability bin. It is
    intentionally simple and auditable for reliability diagnostics; test rows in
    bins without enough training evidence keep their original probabilities.
    """
    tmp = train[["result", prob_col]].dropna().copy()
    tmp[prob_col] = tmp[prob_col].clip(0.0, 1.0)
    tmp["bin"] = np.minimum((tmp[prob_col] * bins).astype(int), bins - 1)
    table: dict[int, dict] = {}
    for bin_id, g in tmp.groupby("bin", sort=True):
        rows = int(len(g))
        mean_prob = float(g[prob_col].mean())
        actual_rate = float(g["result"].astype(float).mean())
        raw_offset = actual_rate - mean_prob
        table[int(bin_id)] = {
            "bin": int(bin_id),
            "prob_min": float(bin_id / bins),
            "prob_max": float((bin_id + 1) / bins),
            "rows": rows,
            "mean_prob": mean_prob,
            "actual_rate": actual_rate,
            "raw_offset": float(raw_offset),
            "applied_offset": float(shrink * raw_offset) if rows >= min_rows else 0.0,
            "used": bool(rows >= min_rows),
        }
    return table


def apply_bin_recalibration(df: pd.DataFrame, prob_col: str, table: dict[int, dict], bins: int = 10) -> pd.Series:
    """Apply a fitted bin-offset recalibration table to a probability column."""
    p = df[prob_col].astype(float).clip(0.0, 1.0)
    bin_ids = np.minimum((p * bins).astype(int), bins - 1)
    offsets = bin_ids.map(lambda b: float(table.get(int(b), {}).get("applied_offset", 0.0)))
    return (p + offsets).clip(0.03, 0.97)


def evaluate_bin_recalibration_shrinkage_sweep(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prob_col: str,
    bins: int = 10,
    min_rows: int = 120,
    shrink_values: list[float] | tuple[float, ...] = (0.0, 0.15, 0.25, 0.35, 0.50, 0.75, 1.0),
) -> dict:
    """Evaluate no-lookahead bin-recalibration shrinkage candidates on one test split.

    Every candidate fits offsets only from ``train`` and scores on ``test``. This is
    a diagnostic sweep, not automatic model selection for future rows; it exposes
    whether the fixed conservative shrink is leaving log-loss/Brier improvement on
    the table or whether stronger offsets overfit.
    """
    candidates = []
    for shrink in shrink_values:
        table = fit_bin_recalibration(train, prob_col, bins=bins, min_rows=min_rows, shrink=float(shrink))
        scored = test[["result", prob_col]].copy()
        scored["recalibrated_p1"] = apply_bin_recalibration(scored, prob_col, table, bins=bins)
        y = scored["result"].astype(int)
        p = scored["recalibrated_p1"].clip(1e-6, 1 - 1e-6)
        candidates.append({
            "shrink": float(shrink),
            "log_loss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "mean_prob": float(p.mean()),
            "actual_rate": float(y.mean()),
            "used_bins": int(sum(1 for row in table.values() if row.get("used"))),
        })
    best_by_log_loss = min(candidates, key=lambda r: r["log_loss"]) if candidates else None
    best_by_brier = min(candidates, key=lambda r: r["brier"]) if candidates else None
    baseline = next((row for row in candidates if abs(row["shrink"]) < 1e-12), None)
    return {
        "source_model": "market_no_vig",
        "prob_col": prob_col,
        "bins": int(bins),
        "min_rows": int(min_rows),
        "fit_scope": "train_rows_only_no_lookahead",
        "baseline": baseline,
        "best_by_log_loss": best_by_log_loss,
        "best_by_brier": best_by_brier,
        "candidates": candidates,
    }


def summarize_shrinkage_sweeps(sweeps: list[dict]) -> dict:
    """Summarize multi-year stability of no-lookahead recalibration shrinkage sweeps.

    The sweep is diagnostic only: it should help decide whether a fixed conservative
    shrinkage is worth changing after several walk-forward years, not auto-tune the
    live/current-year model from test results.
    """
    years: list[int] = []
    log_loss_deltas: list[float] = []
    brier_deltas: list[float] = []
    log_loss_counts: dict[str, int] = {}
    brier_counts: dict[str, int] = {}
    years_improved_log_loss = 0
    years_improved_brier = 0

    for sweep in sweeps:
        if sweep.get("year") is not None:
            years.append(int(sweep["year"]))
        baseline = sweep.get("baseline") or {}
        best_ll = sweep.get("best_by_log_loss") or {}
        best_brier = sweep.get("best_by_brier") or {}
        if baseline and best_ll:
            shrink_key = str(float(best_ll.get("shrink", 0.0)))
            log_loss_counts[shrink_key] = log_loss_counts.get(shrink_key, 0) + 1
            delta = float(best_ll["log_loss"]) - float(baseline["log_loss"])
            log_loss_deltas.append(delta)
            if delta < 0:
                years_improved_log_loss += 1
        if baseline and best_brier:
            shrink_key = str(float(best_brier.get("shrink", 0.0)))
            brier_counts[shrink_key] = brier_counts.get(shrink_key, 0) + 1
            delta = float(best_brier["brier"]) - float(baseline["brier"])
            brier_deltas.append(delta)
            if delta < 0:
                years_improved_brier += 1

    evaluated_years = len(log_loss_deltas)
    avg_log_loss_delta = float(np.mean(log_loss_deltas)) if log_loss_deltas else None
    avg_brier_delta = float(np.mean(brier_deltas)) if brier_deltas else None
    best_shrink_share = max(log_loss_counts.values()) / evaluated_years if evaluated_years and log_loss_counts else 0.0
    if evaluated_years >= 3 and years_improved_log_loss == evaluated_years and best_shrink_share >= 0.67:
        recommendation = "consider_conservative_default_increase_after_review"
    else:
        recommendation = "diagnostic_only_mixed_or_insufficient_years"

    return {
        "years": sorted(years),
        "evaluated_years": int(evaluated_years),
        "best_log_loss_shrink_counts": log_loss_counts,
        "best_brier_shrink_counts": brier_counts,
        "avg_log_loss_delta_vs_baseline": avg_log_loss_delta,
        "avg_brier_delta_vs_baseline": avg_brier_delta,
        "years_improved_log_loss": int(years_improved_log_loss),
        "years_improved_brier": int(years_improved_brier),
        "recommendation": recommendation,
    }


def _metric_leader(model_rows: list[dict], metric: str, higher_is_better: bool, baseline: dict | None) -> dict | None:
    candidates = [row for row in model_rows if row.get(metric) is not None]
    if not candidates:
        return None
    leader = max(candidates, key=lambda r: float(r[metric])) if higher_is_better else min(candidates, key=lambda r: float(r[metric]))
    out = {
        "model": str(leader.get("model", "unknown")),
        metric: float(leader[metric]),
    }
    if baseline and baseline.get(metric) is not None:
        out["delta_vs_baseline"] = float(leader[metric]) - float(baseline[metric])
    return out


def summarize_probability_quality_tradeoffs(model_rows: list[dict], baseline_model: str = "market_no_vig") -> dict:
    """Summarize model leaders without letting accuracy hide poor probabilities.

    Dennis's target is higher match-outcome accuracy only when calibration/log loss/
    Brier remain strong. This compact report makes the tradeoff explicit in the JSON
    artifact so hourly runs can spot accuracy-only regressions quickly.
    """
    rows = list(model_rows)
    baseline = next((row for row in rows if row.get("model") == baseline_model), None)
    ece_rows = []
    for row in rows:
        ece = (row.get("calibration_error_metrics") or {}).get("expected_calibration_error")
        if ece is not None:
            copied = dict(row)
            copied["expected_calibration_error"] = float(ece)
            ece_rows.append(copied)

    best_by_log_loss = _metric_leader(rows, "log_loss", higher_is_better=False, baseline=baseline)
    best_by_accuracy = _metric_leader(rows, "accuracy", higher_is_better=True, baseline=baseline)
    warning = None
    if best_by_log_loss and best_by_accuracy and best_by_log_loss["model"] != best_by_accuracy["model"]:
        log_loss_leader_row = next((row for row in rows if row.get("model") == best_by_log_loss["model"]), {})
        accuracy_gap = float(best_by_accuracy.get("accuracy", 0.0)) - float(log_loss_leader_row.get("accuracy", 0.0))
        if accuracy_gap > 1e-12:
            warning = "accuracy leader is not the log-loss leader; prioritize calibrated probability quality over accuracy-only gains"

    return {
        "baseline_model": baseline_model,
        "best_by_log_loss": best_by_log_loss,
        "best_by_brier": _metric_leader(rows, "brier", higher_is_better=False, baseline=baseline),
        "best_by_accuracy": best_by_accuracy,
        "best_by_ece": _metric_leader(ece_rows, "expected_calibration_error", higher_is_better=False, baseline=(dict(baseline, expected_calibration_error=(baseline.get("calibration_error_metrics") or {}).get("expected_calibration_error")) if baseline else None)),
        "warning": warning,
    }


def summarize_calibration_diagnostics(model_rows: list[dict], min_rows: int = 50, top_n: int = 12) -> dict:
    """Summarize the largest material reliability-bin errors across model rows.

    The existing per-model calibration bins are verbose. This compact summary makes
    the next recalibration target obvious while filtering tiny bins that can dominate
    raw absolute error but are not yet stable enough to tune against.
    """
    worst_bins = []
    excluded_low_sample_bins = 0
    for model_row in model_rows:
        model = str(model_row.get("model", "unknown"))
        for bin_row in model_row.get("calibration_bins", []):
            rows = int(bin_row.get("rows", 0))
            if rows < min_rows:
                excluded_low_sample_bins += 1
                continue
            error = float(bin_row.get("calibration_error", 0.0))
            bin_id = int(bin_row.get("bin", 0))
            prob_min = float(bin_row.get("prob_min", bin_id / 10))
            prob_max = float(bin_row.get("prob_max", (bin_id + 1) / 10))
            abs_error = abs(error)
            worst_bins.append({
                "model": model,
                "bin": bin_id,
                "prob_min": prob_min,
                "prob_max": prob_max,
                "rows": rows,
                "mean_prob": float(bin_row.get("mean_prob", 0.0)),
                "actual_rate": float(bin_row.get("actual_rate", 0.0)),
                "calibration_error": error,
                "direction": "overpredicts_p1" if error > 0 else "underpredicts_p1",
                "abs_calibration_error": abs_error,
                "weighted_abs_error": float(abs_error * rows),
            })
    worst_bins.sort(key=lambda r: (r["abs_calibration_error"], r["weighted_abs_error"]), reverse=True)
    return {
        "min_rows": int(min_rows),
        "top_n": int(top_n),
        "excluded_low_sample_bins": int(excluded_low_sample_bins),
        "worst_bins": worst_bins[:top_n],
    }


def build_paper_model(kind: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    """Model zoo for Buhamra-style Grand Slam benchmark tables."""
    if kind == "spline_logistic":
        pre = ColumnTransformer([
            ("num_spline", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ])
        clf = LogisticRegression(max_iter=2500, C=0.35, random_state=42)
    else:
        pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ])
        if kind == "logistic_market":
            clf = LogisticRegression(max_iter=2000, C=0.7, random_state=42)
        elif kind == "random_forest":
            clf = RandomForestClassifier(n_estimators=260, max_depth=9, min_samples_leaf=6, random_state=42, n_jobs=-1)
        elif kind == "xgboost":
            clf = XGBClassifier(
                n_estimators=180,
                max_depth=3,
                learning_rate=0.045,
                subsample=0.90,
                colsample_bytree=0.90,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        elif kind == "svm_linear":
            clf = SVC(kernel="linear", C=0.6, probability=True, random_state=42)
        elif kind == "advanced_voting":
            clf = VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000, C=0.7, random_state=42)),
                    ("hgb", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, l2_regularization=0.08, random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=180, max_depth=9, min_samples_leaf=8, random_state=42, n_jobs=-1)),
                ],
                voting="soft",
            )
        else:
            raise ValueError(f"unknown paper model kind: {kind}")
    return Pipeline([("features", pre), ("model", clf)])


PAPER_EXPECTED_GRAND_SLAM_ROWS = {
    # Buhamra et al. reported 293 Grand Slam validation rows for 2022; use this
    # as an explicit coverage target instead of letting the small repo sample look
    # like a complete reproduction.
    2022: 293,
}


def grand_slam_coverage_diagnostics(
    gs: pd.DataFrame,
    test_years: list[int],
    evaluated_folds: list[dict],
    skipped_folds: list[dict],
) -> dict:
    """Summarize Grand Slam row-count coverage and paper-replication gaps."""
    years = [int(y) for y in test_years]
    gs = gs.copy()
    if "date" in gs.columns:
        gs["date"] = pd.to_datetime(gs["date"], errors="coerce")
    available = gs[gs["year"].isin(years)] if "year" in gs.columns else pd.DataFrame()
    available_by_year = available.groupby("year").size().to_dict() if not available.empty else {}

    evaluated_by_year: dict[int, int] = defaultdict(int)
    evaluated_tournaments: dict[int, set[str]] = defaultdict(set)
    for fold in evaluated_folds:
        year = int(fold["year"])
        evaluated_by_year[year] += int(fold.get("rows", fold.get("test_rows", 0)))
        evaluated_tournaments[year].add(str(fold["tournament"]))

    skipped_tournaments: dict[int, set[str]] = defaultdict(set)
    for fold in skipped_folds:
        skipped_tournaments[int(fold["year"])].add(str(fold["tournament"]))

    tournaments_by_year: dict[str, dict] = {}
    if not available.empty:
        for (year, tournament), grp in available.groupby(["year", "tournament"]):
            year = int(year)
            tournament = str(tournament)
            if tournament in evaluated_tournaments[year]:
                status = "evaluated"
            elif tournament in skipped_tournaments[year]:
                status = "skipped"
            else:
                status = "not_evaluated"
            tournaments_by_year.setdefault(str(year), {})[tournament] = {
                "available_rows": int(len(grp)),
                "date_min": grp["date"].min().date().isoformat() if "date" in grp and pd.notna(grp["date"].min()) else None,
                "date_max": grp["date"].max().date().isoformat() if "date" in grp and pd.notna(grp["date"].max()) else None,
                "status": status,
            }

    expected_by_year = {str(y): int(PAPER_EXPECTED_GRAND_SLAM_ROWS[y]) for y in years if y in PAPER_EXPECTED_GRAND_SLAM_ROWS}
    missing_vs_paper = {
        str(y): max(0, int(PAPER_EXPECTED_GRAND_SLAM_ROWS[y]) - int(evaluated_by_year.get(y, 0)))
        for y in years
        if y in PAPER_EXPECTED_GRAND_SLAM_ROWS
    }
    available_missing_vs_paper = {
        str(y): max(0, int(PAPER_EXPECTED_GRAND_SLAM_ROWS[y]) - int(available_by_year.get(y, 0)))
        for y in years
        if y in PAPER_EXPECTED_GRAND_SLAM_ROWS
    }
    coverage_ratio = {
        str(y): float(int(evaluated_by_year.get(y, 0)) / PAPER_EXPECTED_GRAND_SLAM_ROWS[y])
        for y in years
        if y in PAPER_EXPECTED_GRAND_SLAM_ROWS and PAPER_EXPECTED_GRAND_SLAM_ROWS[y]
    }
    available_coverage_ratio = {
        str(y): float(int(available_by_year.get(y, 0)) / PAPER_EXPECTED_GRAND_SLAM_ROWS[y])
        for y in years
        if y in PAPER_EXPECTED_GRAND_SLAM_ROWS and PAPER_EXPECTED_GRAND_SLAM_ROWS[y]
    }
    return {
        "available_rows_by_year": {str(y): int(available_by_year.get(y, 0)) for y in years},
        "evaluated_rows_by_year": {str(y): int(evaluated_by_year.get(y, 0)) for y in years},
        "paper_expected_rows_by_year": expected_by_year,
        "missing_vs_paper_by_year": missing_vs_paper,
        "available_missing_vs_paper_by_year": available_missing_vs_paper,
        "coverage_ratio_vs_paper_by_year": coverage_ratio,
        "available_coverage_ratio_vs_paper_by_year": available_coverage_ratio,
        "evaluated_tournament_count_by_year": {str(y): len(evaluated_tournaments.get(y, set())) for y in years},
        "skipped_tournament_count_by_year": {str(y): len(skipped_tournaments.get(y, set())) for y in years},
        "tournaments_by_year": tournaments_by_year,
        "note": "Coverage diagnostics compare evaluated benchmark rows against known paper targets where available; available rows expose source coverage, while evaluated rows expose benchmark coverage.",
    }


def grand_slam_tournament_folds(gs: pd.DataFrame, test_years: list[int]) -> list[dict]:
    """Return chronological train-before-tournament Grand Slam folds."""
    gs = gs.copy()
    gs["date"] = pd.to_datetime(gs["date"], errors="coerce")
    test_tournaments = (
        gs[gs["year"].isin(test_years)]
        .groupby(["year", "tournament"], as_index=False)["date"]
        .min()
        .sort_values(["date", "tournament"])
        .to_dict("records")
    )
    folds = []
    for t in test_tournaments:
        tournament = str(t["tournament"])
        test = gs[(gs["year"].eq(int(t["year"]))) & (gs["tournament"].eq(tournament))].copy()
        train = gs[gs["date"] < test["date"].min()].copy()
        folds.append({"year": int(t["year"]), "tournament": tournament, "train": train, "test": test})
    return folds


def run_grand_slam_tournament_benchmark(data: pd.DataFrame, test_years: list[int]) -> dict:
    """Replicate the paper's spirit: Grand Slam only, market feature included, tournament-level expanding window."""
    gs = data[data["is_grand_slam"].eq(1)].sort_values(["date", "tournament", "round", "player1", "player2"]).copy()
    if gs.empty:
        return {"error": "no Grand Slam rows available"}
    model_defs = {
        "market_no_vig": None,
        "logistic_market": (MARKET_NUMERIC, CATEGORICAL),
        "spline_logistic": (["market_logit_p1", "rank_diff", "points_ratio", "elo_diff", "surface_elo_diff", "slam_elo_diff", "surface_ability_diff", "recent_form_ability_diff", "fatigue_adjusted_ability_diff"], CATEGORICAL),
        "random_forest": (MARKET_NUMERIC, CATEGORICAL),
        "xgboost": (MARKET_NUMERIC, CATEGORICAL),
        "svm_linear": (["market_logit_p1", "rank_diff", "points_ratio", "elo_diff", "surface_elo_diff", "slam_elo_diff", "surface_ability_diff", "bo5_ability_diff", "recent_form_ability_diff", "fatigue_adjusted_ability_diff"], CATEGORICAL),
        "advanced_voting": (MARKET_NUMERIC, CATEGORICAL),
    }
    test_tournaments = (
        gs[gs["year"].isin(test_years)]
        .groupby(["year", "tournament"], as_index=False)["date"]
        .min()
        .sort_values(["date", "tournament"])
        .to_dict("records")
    )
    preds_all = []
    tournament_rows = []
    evaluated_folds = []
    skipped = []
    for fold in grand_slam_tournament_folds(gs, test_years):
        tournament = fold["tournament"]
        train = fold["train"].copy()
        test = fold["test"].copy()
        if len(train) < 300 or len(test) < 20 or train["result"].nunique() < 2:
            skipped.append({"year": int(fold["year"]), "tournament": tournament, "train_rows": int(len(train)), "test_rows": int(len(test))})
            continue
        out = test.copy()
        out["benchmark_scope"] = "grand_slam_tournament_expanding"
        out["market_no_vig_p1"] = out["implied_p1_no_vig"]
        for name, spec in model_defs.items():
            if spec is None:
                continue
            num, cat = spec
            features = num + cat
            model = build_paper_model(name, num, cat)
            model.fit(train[features], train["result"].astype(int))
            out[f"{name}_p1"] = model.predict_proba(test[features])[:, 1]
        residual_features = MARKET_NUMERIC + CATEGORICAL
        out["residual_overlay_segment_tuned_p1"], _ = fit_residual_overlay(
            train=train,
            test=test,
            features=residual_features,
            shrink=0.55,
            min_specialist_rows=150,
            segment_shrink={"grand_slam": 0.55, "top10_match": 0.65, "surface_switch": 0.45},
        )
        for name in list(model_defs) + ["residual_overlay_segment_tuned"]:
            tournament_rows.append({
                "scope": "grand_slam_tournament_expanding",
                "year": int(fold["year"]),
                "tournament": tournament,
                "model": name,
                **metrics_for(out, f"{name}_p1"),
            })
        evaluated_folds.append({"year": int(fold["year"]), "tournament": tournament, "rows": int(len(out))})
        preds_all.append(out)
    if not preds_all:
        return {
            "error": "no Grand Slam tournament folds could be evaluated",
            "skipped": skipped,
            "coverage_diagnostics": grand_slam_coverage_diagnostics(gs, test_years, evaluated_folds, skipped),
        }
    preds = pd.concat(preds_all, ignore_index=True)
    overall = []
    for name in list(model_defs) + ["residual_overlay_segment_tuned"]:
        m = metrics_for(preds, f"{name}_p1")
        bins = calibration_bins(preds, f"{name}_p1")
        overall.append({
            "scope": "grand_slam_tournament_expanding",
            "model": name,
            **m,
            "calibration_error_metrics": calibration_error_metrics(bins),
            "calibration_bins": bins,
        })
    return {
        "scope": "grand_slam_tournament_expanding",
        "description": "Grand Slam-only benchmark using chronological train-before-tournament folds, bookmaker no-vig probability included where applicable.",
        "test_years": test_years,
        "rows": int(len(preds)),
        "folds": int(len(preds_all)),
        "skipped_folds": skipped,
        "coverage_diagnostics": grand_slam_coverage_diagnostics(gs, test_years, evaluated_folds, skipped),
        "overall_model_comparison": sorted(overall, key=lambda r: (r["log_loss"], -r["accuracy"])),
        "tournament_model_comparison": tournament_rows,
    }


def candidate_betting(df: pd.DataFrame, prob_col: str, threshold: float) -> dict:
    tmp = df.copy()
    tmp["model_p1"] = tmp[prob_col]
    tmp = add_candidate_columns(tmp)
    bets = tmp[tmp["candidate_edge"] >= threshold]
    if bets.empty:
        return {"threshold": threshold, "bets": 0, "profit": 0.0, "roi": 0.0}
    return {
        "threshold": threshold,
        "bets": int(len(bets)),
        "profit": float(bets["candidate_profit"].sum()),
        "roi": float(bets["candidate_profit"].mean()),
        "hit_rate": float(bets["candidate_result"].mean()),
        "avg_odds": float(bets["candidate_odds"].mean()),
        "avg_edge": float(bets["candidate_edge"].mean()),
    }


def _bucket_segment_diagnostics(preds: pd.DataFrame) -> pd.DataFrame:
    """Add coarse all-data diagnostic buckets without affecting model features."""
    df = preds.copy()
    if "rank_diff" in df:
        df["rank_diff_bucket"] = pd.cut(
            df["rank_diff"],
            bins=[-np.inf, -50, -15, 15, 50, np.inf],
            labels=["p1_much_higher_rank", "p1_higher_rank", "similar_rank", "p1_lower_rank", "p1_much_lower_rank"],
        ).astype(str)
    if "implied_p1_no_vig" in df:
        df["market_prob_bucket"] = pd.cut(
            df["implied_p1_no_vig"],
            bins=[-np.inf, 0.30, 0.45, 0.55, 0.70, np.inf],
            labels=["heavy_p2_favorite", "p2_favorite", "near_pickem", "p1_favorite", "heavy_p1_favorite"],
        ).astype(str)
    if "rest_diff" in df:
        df["rest_diff_bucket"] = pd.cut(
            df["rest_diff"],
            bins=[-np.inf, -3, -1, 1, 3, np.inf],
            labels=["p1_less_rest", "p1_slightly_less_rest", "similar_rest", "p1_slightly_more_rest", "p1_more_rest"],
        ).astype(str)
    if "matches_last7_diff" in df:
        df["matches_last7_diff_bucket"] = pd.cut(
            df["matches_last7_diff"],
            bins=[-np.inf, -2, -1, 1, 2, np.inf],
            labels=["p2_heavier_load", "p2_slightly_heavier_load", "similar_load", "p1_slightly_heavier_load", "p1_heavier_load"],
        ).astype(str)
    if {"p1_surface_switch", "p2_surface_switch"}.issubset(df.columns):
        df["surface_switch_any"] = (df["p1_surface_switch"].eq(1) | df["p2_surface_switch"].eq(1))
    title_final_cols = ["p1_title_within_14", "p2_title_within_14", "p1_final_within_7", "p2_final_within_7"]
    if set(title_final_cols).issubset(df.columns):
        df["post_title_or_final_any"] = df[title_final_cols].eq(1).any(axis=1)
    return df


DEFAULT_SEGMENT_COLS = [
    "surface", "series", "court", "round_group", "round", "is_early_round", "any_top10", "both_top10",
    "early_after_title_p1", "early_after_title_p2", "rank_diff_bucket", "market_prob_bucket",
    "rest_diff_bucket", "matches_last7_diff_bucket", "surface_switch_any", "post_title_or_final_any",
]


def segment_errors(preds: pd.DataFrame, prob_col: str) -> list[dict]:
    rows = []
    df = _bucket_segment_diagnostics(preds)
    for col in [c for c in DEFAULT_SEGMENT_COLS if c in df.columns]:
        for seg, g in df.groupby(col, dropna=False):
            if len(g) < 80:
                continue
            m = metrics_for(g, prob_col)
            rows.append({
                "segment_col": col,
                "segment": str(seg),
                **m,
                **segment_year_stability(g, prob_col),
                "model_minus_market_log_loss": float(m["log_loss"] - m["market_log_loss"]),
                "model_minus_market_brier": float(m["brier"] - m["market_brier"]),
            })
    return sorted(rows, key=lambda r: r["model_minus_market_log_loss"], reverse=True)


def model_market_disagreement_segments(
    preds: pd.DataFrame,
    prob_col: str,
    segment_cols: list[str] | None = None,
    min_rows: int = 120,
) -> list[dict]:
    """Score contexts where a model flips the no-vig market favorite.

    Broad segment scans can be dominated by rows where model and market choose the
    same side. This diagnostic isolates true model-vs-market pick disagreements,
    then reports whether those overrides improve or damage accuracy/proper scores
    with the same multi-year stability fields as the broader segment scans.
    """
    rows = []
    df = _bucket_segment_diagnostics(preds)
    if prob_col not in df.columns or "implied_p1_no_vig" not in df.columns:
        return rows
    df = df.copy()
    df["_model_pick"] = (df[prob_col] >= 0.5).astype(int)
    df["_market_pick"] = (df["implied_p1_no_vig"] >= 0.5).astype(int)
    df["_model_market_disagree"] = df["_model_pick"] != df["_market_pick"]
    cols = segment_cols or DEFAULT_SEGMENT_COLS
    for col in [c for c in cols if c in df.columns]:
        for seg, all_g in df.groupby(col, dropna=False):
            g = all_g[all_g["_model_market_disagree"]].copy()
            if len(g) < min_rows:
                continue
            m = metrics_for(g, prob_col)
            y = g["result"].astype(int)
            model_pick_accuracy = float((g["_model_pick"] == y).mean())
            market_pick_accuracy = float((g["_market_pick"] == y).mean())
            rows.append({
                "segment_col": col,
                "segment": str(seg),
                **m,
                **segment_year_stability(g, prob_col),
                "disagreement_rows": int(len(g)),
                "agreement_rows_excluded": int(len(all_g) - len(g)),
                "model_pick_accuracy": model_pick_accuracy,
                "market_pick_accuracy": market_pick_accuracy,
                "model_minus_market_pick_accuracy": float(model_pick_accuracy - market_pick_accuracy),
                "model_minus_market_log_loss": float(m["log_loss"] - m["market_log_loss"]),
                "model_minus_market_brier": float(m["brier"] - m["market_brier"]),
            })
    return sorted(rows, key=lambda r: r["model_minus_market_log_loss"], reverse=True)


DEFAULT_INTERACTION_SEGMENT_PAIRS = [
    ("series", "rank_diff_bucket"),
    ("series", "market_prob_bucket"),
    ("series", "round_group"),
    ("series", "surface_switch_any"),
    ("surface", "market_prob_bucket"),
    ("surface", "round_group"),
    ("surface", "rank_diff_bucket"),
    ("round_group", "market_prob_bucket"),
    ("round_group", "rank_diff_bucket"),
    ("market_prob_bucket", "matches_last7_diff_bucket"),
    ("market_prob_bucket", "rest_diff_bucket"),
    ("market_prob_bucket", "surface_switch_any"),
    ("rank_diff_bucket", "matches_last7_diff_bucket"),
    ("rank_diff_bucket", "rest_diff_bucket"),
    ("surface_switch_any", "post_title_or_final_any"),
]


def interaction_segment_errors(
    preds: pd.DataFrame,
    prob_col: str,
    interaction_pairs: list[tuple[str, str]] | None = None,
    min_rows: int = 120,
) -> list[dict]:
    """Score material two-way diagnostic buckets against the no-vig market.

    Single-column segment scans hide many actionable patterns. This helper keeps
    the same no-lookahead OOS rows but intersects coarse diagnostic buckets (for
    example series x rank bucket) so stable weak/strength clusters can surface.
    """
    rows = []
    df = _bucket_segment_diagnostics(preds)
    pairs = interaction_pairs or DEFAULT_INTERACTION_SEGMENT_PAIRS
    for left, right in pairs:
        if left not in df.columns or right not in df.columns:
            continue
        tmp = df.copy()
        segment_col = f"{left}__{right}"
        tmp[segment_col] = tmp[left].astype(str) + " | " + tmp[right].astype(str)
        for seg, g in tmp.groupby(segment_col, dropna=False):
            if len(g) < min_rows:
                continue
            m = metrics_for(g, prob_col)
            rows.append({
                "segment_col": segment_col,
                "segment": str(seg),
                "left_segment_col": left,
                "right_segment_col": right,
                **m,
                **segment_year_stability(g, prob_col),
                "model_minus_market_log_loss": float(m["log_loss"] - m["market_log_loss"]),
                "model_minus_market_brier": float(m["brier"] - m["market_brier"]),
            })
    return sorted(rows, key=lambda r: r["model_minus_market_log_loss"], reverse=True)


DEFAULT_MULTIVARIATE_SEGMENT_GROUPS = [
    ("series", "round_group", "rank_diff_bucket"),
    ("series", "round_group", "market_prob_bucket"),
    ("series", "market_prob_bucket", "surface_switch_any"),
    ("series", "market_prob_bucket", "rest_diff_bucket"),
    ("surface", "round_group", "market_prob_bucket"),
    ("surface", "round_group", "rank_diff_bucket"),
    ("surface", "market_prob_bucket", "surface_switch_any"),
    ("market_prob_bucket", "rank_diff_bucket", "matches_last7_diff_bucket"),
    ("market_prob_bucket", "rank_diff_bucket", "rest_diff_bucket"),
    ("market_prob_bucket", "surface_switch_any", "post_title_or_final_any"),
]


def multivariate_segment_errors(
    preds: pd.DataFrame,
    prob_col: str,
    segment_groups: list[tuple[str, ...]] | None = None,
    min_rows: int = 120,
) -> list[dict]:
    """Score material three-way-plus diagnostic buckets against the no-vig market.

    Two-way scans are useful but can still smear together distinct contexts. This
    keeps the same OOS prediction rows and only creates reporting-only composite
    buckets, with yearly stability fields to avoid surfacing one-year noise.
    """
    rows = []
    df = _bucket_segment_diagnostics(preds)
    groups = segment_groups or DEFAULT_MULTIVARIATE_SEGMENT_GROUPS
    for group in groups:
        if not set(group).issubset(df.columns):
            continue
        tmp = df.copy()
        segment_col = "__".join(group)
        segment_values = tmp[list(group)].astype(object).where(tmp[list(group)].notna(), "nan").astype(str)
        tmp[segment_col] = segment_values.agg(" | ".join, axis=1)
        for seg, g in tmp.groupby(segment_col, dropna=False):
            if len(g) < min_rows:
                continue
            m = metrics_for(g, prob_col)
            rows.append({
                "segment_col": segment_col,
                "segment": str(seg),
                "segment_columns": list(group),
                **m,
                **segment_year_stability(g, prob_col),
                "model_minus_market_log_loss": float(m["log_loss"] - m["market_log_loss"]),
                "model_minus_market_brier": float(m["brier"] - m["market_brier"]),
            })
    return sorted(rows, key=lambda r: r["model_minus_market_log_loss"], reverse=True)


def segment_year_stability(g: pd.DataFrame, prob_col: str) -> dict:
    """Summarize whether a segment's model-vs-market result persists across years."""
    empty = {
        "years": [],
        "year_count": 0,
        "min_year_rows": 0,
        "years_model_beats_market_log_loss": 0,
        "years_model_beats_market_brier": 0,
        "years_model_lags_market_log_loss": 0,
        "years_model_lags_market_brier": 0,
        "yearly_model_minus_market": [],
    }
    if "date" not in g.columns:
        return empty
    tmp = g.copy()
    tmp["_year"] = pd.to_datetime(tmp["date"], errors="coerce").dt.year
    tmp = tmp.dropna(subset=["_year"])
    if tmp.empty:
        return empty
    yearly = []
    for year, yg in tmp.groupby("_year"):
        m = metrics_for(yg, prob_col)
        yearly.append({
            "year": int(year),
            "rows": int(len(yg)),
            "model_minus_market_log_loss": float(m["log_loss"] - m["market_log_loss"]),
            "model_minus_market_brier": float(m["brier"] - m["market_brier"]),
        })
    return {
        "years": [row["year"] for row in yearly],
        "year_count": int(len(yearly)),
        "min_year_rows": int(min(row["rows"] for row in yearly)),
        "years_model_beats_market_log_loss": int(sum(row["model_minus_market_log_loss"] < 0 for row in yearly)),
        "years_model_beats_market_brier": int(sum(row["model_minus_market_brier"] < 0 for row in yearly)),
        "years_model_lags_market_log_loss": int(sum(row["model_minus_market_log_loss"] > 0 for row in yearly)),
        "years_model_lags_market_brier": int(sum(row["model_minus_market_brier"] > 0 for row in yearly)),
        "yearly_model_minus_market": yearly,
    }


def summarize_segment_strengths(
    segment_rows: list[dict],
    min_rows: int = 150,
    top_n: int = 12,
    min_years: int = 3,
    min_stable_year_share: float = 0.60,
) -> dict:
    """Return material segments where a model beats the no-vig market baseline.

    Segment diagnostics are sorted worst-first for debugging. This companion summary
    makes stable-looking strengths visible without changing any model predictions or
    treating broad-scan positives as betting signals.
    """
    candidates = []
    excluded_low_sample_segments = 0
    excluded_unstable_segments = 0
    for row in segment_rows:
        rows = int(row.get("rows", 0))
        log_loss_delta = float(row.get("model_minus_market_log_loss", 0.0))
        brier_delta = float(row.get("model_minus_market_brier", 0.0))
        if log_loss_delta >= 0 or brier_delta >= 0:
            continue
        if rows < min_rows:
            excluded_low_sample_segments += 1
            continue
        year_count = int(row.get("year_count", 0))
        if year_count:
            required_stable_years = max(min_years, int(math.ceil(year_count * min_stable_year_share)))
            if (
                year_count < min_years
                or int(row.get("years_model_beats_market_log_loss", 0)) < required_stable_years
                or int(row.get("years_model_beats_market_brier", 0)) < required_stable_years
            ):
                excluded_unstable_segments += 1
                continue
        enriched = dict(row)
        enriched["weighted_log_loss_improvement"] = float(-log_loss_delta * rows)
        enriched["weighted_brier_improvement"] = float(-brier_delta * rows)
        enriched["stability_rule"] = {
            "min_years": int(min_years),
            "min_stable_year_share": float(min_stable_year_share),
        }
        enriched["hypothesis_label"] = "market_beating_segment_hypothesis"
        candidates.append(enriched)
    candidates.sort(key=lambda r: (r["weighted_log_loss_improvement"], -float(r.get("log_loss", 0.0))), reverse=True)
    return {
        "min_rows": int(min_rows),
        "min_years": int(min_years),
        "min_stable_year_share": float(min_stable_year_share),
        "top_n": int(top_n),
        "candidate_count": int(len(candidates)),
        "excluded_low_sample_segments": int(excluded_low_sample_segments),
        "excluded_unstable_segments": int(excluded_unstable_segments),
        "top_segments": candidates[:top_n],
        "note": "Research-only segment hypotheses from historical walk-forward rows; require multi-year stability and forward CLV/paper tracking before use.",
    }


def summarize_segment_weaknesses(
    segment_rows: list[dict],
    min_rows: int = 150,
    top_n: int = 12,
    min_years: int = 3,
    min_stable_year_share: float = 0.60,
) -> dict:
    """Return material segments where a model persistently lags the no-vig market.

    This companion to the strength summary keeps hourly reports focused on stable,
    high-impact failure modes instead of one-year noise or tiny buckets.
    """
    candidates = []
    excluded_low_sample_segments = 0
    excluded_unstable_segments = 0
    for row in segment_rows:
        rows = int(row.get("rows", 0))
        log_loss_delta = float(row.get("model_minus_market_log_loss", 0.0))
        brier_delta = float(row.get("model_minus_market_brier", 0.0))
        if log_loss_delta <= 0 or brier_delta <= 0:
            continue
        if rows < min_rows:
            excluded_low_sample_segments += 1
            continue
        year_count = int(row.get("year_count", 0))
        if year_count:
            required_stable_years = max(min_years, int(math.ceil(year_count * min_stable_year_share)))
            lag_ll = int(row.get("years_model_lags_market_log_loss", 0))
            lag_brier = int(row.get("years_model_lags_market_brier", 0))
            if year_count < min_years or lag_ll < required_stable_years or lag_brier < required_stable_years:
                excluded_unstable_segments += 1
                continue
        enriched = dict(row)
        enriched["weighted_log_loss_damage"] = float(log_loss_delta * rows)
        enriched["weighted_brier_damage"] = float(brier_delta * rows)
        enriched["stability_rule"] = {
            "min_years": int(min_years),
            "min_stable_year_share": float(min_stable_year_share),
        }
        enriched["hypothesis_label"] = "stable_market_lagging_segment"
        candidates.append(enriched)
    candidates.sort(key=lambda r: (r["weighted_log_loss_damage"], r["weighted_brier_damage"]), reverse=True)
    return {
        "min_rows": int(min_rows),
        "min_years": int(min_years),
        "min_stable_year_share": float(min_stable_year_share),
        "top_n": int(top_n),
        "candidate_count": int(len(candidates)),
        "excluded_low_sample_segments": int(excluded_low_sample_segments),
        "excluded_unstable_segments": int(excluded_unstable_segments),
        "top_segments": candidates[:top_n],
        "note": "Research-only stable weakness hypotheses from historical walk-forward rows; use these to prioritize feature/routing work, not betting execution.",
    }


def cluster_underperformance(preds: pd.DataFrame, prob_col: str, n_clusters: int = 8) -> list[dict]:
    df = preds.copy()
    y = df["result"].astype(int)
    eps = 1e-6
    df["model_loss"] = -(y * np.log(df[prob_col].clip(eps, 1 - eps)) + (1 - y) * np.log((1 - df[prob_col]).clip(eps, 1 - eps)))
    df["market_loss"] = -(y * np.log(df["implied_p1_no_vig"].clip(eps, 1 - eps)) + (1 - y) * np.log((1 - df["implied_p1_no_vig"]).clip(eps, 1 - eps)))
    df["loss_gap"] = df["model_loss"] - df["market_loss"]
    candidates = df[df["loss_gap"] > 0].copy()
    if len(candidates) < n_clusters * 50:
        candidates = df.copy()
    cluster_features = [
        "implied_p1_no_vig", "market_logit_p1", "rank_diff", "elo_diff", "surface_elo_diff", "slam_elo_diff",
        "last5_diff", "surface_pct_diff", "h2h_diff", "rest_diff", "matches_last7_diff", "matches_last14_diff",
        "title_within_14_diff", "final_within_7_diff", "surface_switch_diff", "is_grand_slam", "is_early_round", "any_top10", "both_top10",
        "tourney_matches_before_diff",
    ]
    X = candidates[cluster_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    candidates["cluster"] = km.fit_predict(Xs)
    rows = []
    for c, g in candidates.groupby("cluster"):
        if len(g) < 50:
            continue
        m = metrics_for(g, prob_col)
        top_surface = g["surface"].mode().iat[0] if not g["surface"].mode().empty else None
        top_series = g["series"].mode().iat[0] if not g["series"].mode().empty else None
        top_round = g["round_group"].mode().iat[0] if not g["round_group"].mode().empty else None
        rows.append({
            "cluster": int(c),
            "rows": int(len(g)),
            "avg_loss_gap": float(g["loss_gap"].mean()),
            "model_log_loss": m["log_loss"],
            "market_log_loss": m["market_log_loss"],
            "model_minus_market_log_loss": float(m["log_loss"] - m["market_log_loss"]),
            "actual_rate": m["actual_rate"],
            "mean_prob": m["mean_prob"],
            "avg_market": float(g["implied_p1_no_vig"].mean()),
            "top_surface": str(top_surface),
            "top_series": str(top_series),
            "top_round_group": str(top_round),
            "early_round_rate": float(g["is_early_round"].mean()),
            "grand_slam_rate": float(g["is_grand_slam"].mean()),
            "any_top10_rate": float(g["any_top10"].mean()),
            "title_within_14_abs_rate": float((g["p1_title_within_14"].eq(1) | g["p2_title_within_14"].eq(1)).mean()),
            "surface_switch_abs_rate": float((g["p1_surface_switch"].eq(1) | g["p2_surface_switch"].eq(1)).mean()),
            "avg_rank_diff": float(g["rank_diff"].mean()),
            "avg_elo_diff": float(g["elo_diff"].mean()),
            "avg_rest_diff": float(g["rest_diff"].mean()),
            "avg_matches_last7_diff": float(g["matches_last7_diff"].mean()),
        })
    return sorted(rows, key=lambda r: r["model_minus_market_log_loss"], reverse=True)


def run(years: list[int], test_years: list[int], paper_test_years: list[int] | None = None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_odds_data(years)
    data = add_advanced_side_dataset(raw, seed=42).dropna(subset=["result", "p1_odds", "p2_odds", "implied_p1_no_vig"])
    preds_all = []
    yearly = []
    recalibration_diagnostics = []
    recalibration_shrinkage_sweeps = []
    model_defs = {
        "base_features": (BASE_NUMERIC, ["surface", "series", "round"]),
        "advanced_features": (ADV_NUMERIC, CATEGORICAL),
        "market_aware": (MARKET_NUMERIC, CATEGORICAL),
    }
    for year in test_years:
        train = data[data["date"].dt.year < year].copy()
        test = data[data["date"].dt.year == year].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        out = test.copy()
        out["market_no_vig_p1"] = out["implied_p1_no_vig"]
        yearly.append({"year": year, "model": "market_no_vig", **metrics_for(out, "market_no_vig_p1")})
        market_recalibration_table = fit_bin_recalibration(
            train,
            "implied_p1_no_vig",
            bins=10,
            min_rows=120,
            shrink=0.25,
        )
        out["market_bin_recalibrated_p1"] = apply_bin_recalibration(out, "implied_p1_no_vig", market_recalibration_table, bins=10)
        yearly.append({"year": year, "model": "market_bin_recalibrated", **metrics_for(out, "market_bin_recalibrated_p1")})
        recalibration_diagnostics.append({
            "year": int(year),
            "model": "market_bin_recalibrated",
            "source_model": "market_no_vig",
            "fit_scope": "train_rows_with_date_year_less_than_test_year",
            "bins": list(market_recalibration_table.values()),
        })
        sweep = evaluate_bin_recalibration_shrinkage_sweep(
            train,
            test,
            prob_col="implied_p1_no_vig",
            bins=10,
            min_rows=120,
            shrink_values=[0.0, 0.15, 0.25, 0.35, 0.50, 0.75, 1.0],
        )
        sweep["year"] = int(year)
        recalibration_shrinkage_sweeps.append(sweep)
        fitted_models = {}
        for name, (num, cat) in model_defs.items():
            model = build_model(num, cat)
            features = num + cat
            model.fit(train[features], train["result"].astype(int))
            fitted_models[name] = (model, features)
            out[f"{name}_p1"] = model.predict_proba(test[features])[:, 1]
            yearly.append({"year": year, "model": name, **metrics_for(out, f"{name}_p1")})
        # Simple market/model blends. Market is hard to beat; use research-only overlay.
        out["blend_market_advanced_25_p1"] = 0.75 * out["implied_p1_no_vig"] + 0.25 * out["advanced_features_p1"]
        out["blend_market_advanced_50_p1"] = 0.50 * out["implied_p1_no_vig"] + 0.50 * out["advanced_features_p1"]
        out["blend_market_aware_advanced_p1"] = 0.50 * out["market_aware_p1"] + 0.50 * out["advanced_features_p1"]
        residual_features = MARKET_NUMERIC + CATEGORICAL
        out["residual_overlay_p1"], residual_diag = fit_residual_overlay(
            train=train,
            test=test,
            features=residual_features,
            shrink=0.55,
            min_specialist_rows=350,
        )
        out["residual_overlay_aggressive_p1"], aggressive_diag = fit_residual_overlay(
            train=train,
            test=test,
            features=residual_features,
            shrink=0.85,
            min_specialist_rows=350,
        )
        segment_shrink = {
            "early_atp250": 0.35,
            "post_title_or_final": 0.40,
            "surface_switch": 0.45,
            "grand_slam": 0.55,
            "top10_match": 0.65,
            "early_surface_switch": 0.35,
        }
        out["residual_overlay_segment_tuned_p1"], tuned_diag = fit_residual_overlay(
            train=train,
            test=test,
            features=residual_features,
            shrink=0.55,
            min_specialist_rows=350,
            segment_shrink=segment_shrink,
        )
        # Underperformance-risk filter: predicts when the advanced model is likely worse than the market.
        adv_model, adv_features = fitted_models["advanced_features"]
        train_adv_p = pd.Series(adv_model.predict_proba(train[adv_features])[:, 1], index=train.index).clip(1e-6, 1 - 1e-6)
        y_train = train["result"].astype(int)
        adv_loss = -(y_train * np.log(train_adv_p) + (1 - y_train) * np.log(1 - train_adv_p))
        market_p = train["implied_p1_no_vig"].clip(1e-6, 1 - 1e-6)
        market_loss = -(y_train * np.log(market_p) + (1 - y_train) * np.log(1 - market_p))
        train_underperf = (adv_loss > market_loss).astype(int)
        if train_underperf.nunique() == 2:
            underperf_model = build_model(MARKET_NUMERIC, CATEGORICAL)
            underperf_model.fit(train[residual_features], train_underperf)
            out["underperformance_risk"] = underperf_model.predict_proba(test[residual_features])[:, 1]
        else:
            out["underperformance_risk"] = float(train_underperf.mean())
        out["residual_overlay_filtered_p1"] = out["residual_overlay_segment_tuned_p1"].where(
            out["underperformance_risk"] < 0.65,
            out["implied_p1_no_vig"],
        )
        out["residual_diag_json"] = json.dumps({"residual_overlay": residual_diag, "residual_overlay_aggressive": aggressive_diag, "residual_overlay_segment_tuned": tuned_diag})
        for name in [
            "blend_market_advanced_25",
            "blend_market_advanced_50",
            "blend_market_aware_advanced",
            "residual_overlay",
            "residual_overlay_aggressive",
            "residual_overlay_segment_tuned",
            "residual_overlay_filtered",
        ]:
            yearly.append({"year": year, "model": name, **metrics_for(out, f"{name}_p1")})
        preds_all.append(out)
    preds = pd.concat(preds_all, ignore_index=True)
    model_names = [
        "market_no_vig",
        "market_bin_recalibrated",
        "base_features",
        "advanced_features",
        "market_aware",
        "blend_market_advanced_25",
        "blend_market_advanced_50",
        "blend_market_aware_advanced",
        "residual_overlay",
        "residual_overlay_aggressive",
        "residual_overlay_segment_tuned",
        "residual_overlay_filtered",
    ]
    overall = []
    for name in model_names:
        col = f"{name}_p1"
        m = metrics_for(preds, col)
        bins = calibration_bins(preds, col)
        bets = [candidate_betting(preds, col, t) for t in [0, 0.02, 0.04, 0.06, 0.08, 0.10]]
        overall.append({
            "model": name,
            **m,
            "calibration_error_metrics": calibration_error_metrics(bins),
            "calibration_bins": bins,
            "best_betting_by_profit": max(bets, key=lambda r: r["profit"]),
            "betting_thresholds": bets,
        })
    base_ll = next(r["log_loss"] for r in overall if r["model"] == "base_features")
    for r in overall:
        r["log_loss_delta_vs_base"] = float(r["log_loss"] - base_ll)
    paper_test_years = paper_test_years or ([min(test_years)] if test_years else [])
    grand_slam_benchmark = run_grand_slam_tournament_benchmark(data, paper_test_years)
    grand_slam_model_rows = grand_slam_benchmark.get("overall_model_comparison", []) if isinstance(grand_slam_benchmark, dict) else []
    advanced_segments = segment_errors(preds, "advanced_features_p1")
    residual_segments = segment_errors(preds, "residual_overlay_p1")
    advanced_interaction_segments = interaction_segment_errors(preds, "advanced_features_p1", min_rows=120)
    residual_interaction_segments = interaction_segment_errors(preds, "residual_overlay_p1", min_rows=120)
    advanced_multivariate_segments = multivariate_segment_errors(preds, "advanced_features_p1", min_rows=120)
    residual_multivariate_segments = multivariate_segment_errors(preds, "residual_overlay_p1", min_rows=120)
    advanced_disagreement_segments = model_market_disagreement_segments(preds, "advanced_features_p1", min_rows=120)
    residual_disagreement_segments = model_market_disagreement_segments(preds, "residual_overlay_p1", min_rows=120)
    filtered_disagreement_segments = model_market_disagreement_segments(preds, "residual_overlay_filtered_p1", min_rows=120)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "years": years,
        "test_years": test_years,
        "overall_model_comparison": sorted(overall, key=lambda r: r["log_loss"]),
        "overall_probability_quality_summary": summarize_probability_quality_tradeoffs(overall, baseline_model="market_no_vig"),
        "overall_calibration_summary": summarize_calibration_diagnostics(overall, min_rows=50, top_n=12),
        "bin_recalibration_diagnostics": recalibration_diagnostics,
        "bin_recalibration_shrinkage_sweeps": recalibration_shrinkage_sweeps,
        "bin_recalibration_shrinkage_summary": summarize_shrinkage_sweeps(recalibration_shrinkage_sweeps),
        "yearly_model_comparison": yearly,
        "grand_slam_tournament_expanding_benchmark": grand_slam_benchmark,
        "grand_slam_calibration_summary": summarize_calibration_diagnostics(grand_slam_model_rows, min_rows=20, top_n=12),
        "where_advanced_underperforms_market": advanced_segments[:40],
        "where_residual_overlay_underperforms_market": residual_segments[:40],
        "where_advanced_interactions_underperform_market": advanced_interaction_segments[:40],
        "where_residual_overlay_interactions_underperform_market": residual_interaction_segments[:40],
        "where_advanced_multivariate_underperform_market": advanced_multivariate_segments[:40],
        "where_residual_overlay_multivariate_underperform_market": residual_multivariate_segments[:40],
        "where_advanced_disagrees_with_market": advanced_disagreement_segments[:40],
        "where_residual_overlay_disagrees_with_market": residual_disagreement_segments[:40],
        "where_filtered_overlay_disagrees_with_market": filtered_disagreement_segments[:40],
        "where_advanced_stably_lags_market": summarize_segment_weaknesses(advanced_segments, min_rows=150, top_n=12),
        "where_residual_overlay_stably_lags_market": summarize_segment_weaknesses(residual_segments, min_rows=150, top_n=12),
        "where_advanced_interactions_stably_lag_market": summarize_segment_weaknesses(advanced_interaction_segments, min_rows=150, top_n=12),
        "where_residual_overlay_interactions_stably_lag_market": summarize_segment_weaknesses(residual_interaction_segments, min_rows=150, top_n=12),
        "where_advanced_multivariate_stably_lag_market": summarize_segment_weaknesses(advanced_multivariate_segments, min_rows=150, top_n=12),
        "where_residual_overlay_multivariate_stably_lag_market": summarize_segment_weaknesses(residual_multivariate_segments, min_rows=150, top_n=12),
        "where_advanced_beats_market": summarize_segment_strengths(advanced_segments, min_rows=150, top_n=12),
        "where_residual_overlay_beats_market": summarize_segment_strengths(residual_segments, min_rows=150, top_n=12),
        "where_advanced_interactions_beat_market": summarize_segment_strengths(advanced_interaction_segments, min_rows=150, top_n=12),
        "where_residual_overlay_interactions_beat_market": summarize_segment_strengths(residual_interaction_segments, min_rows=150, top_n=12),
        "where_advanced_multivariate_beat_market": summarize_segment_strengths(advanced_multivariate_segments, min_rows=150, top_n=12),
        "where_residual_overlay_multivariate_beat_market": summarize_segment_strengths(residual_multivariate_segments, min_rows=150, top_n=12),
        "where_advanced_disagreement_stably_lags_market": summarize_segment_weaknesses(advanced_disagreement_segments, min_rows=150, top_n=12),
        "where_residual_overlay_disagreement_stably_lags_market": summarize_segment_weaknesses(residual_disagreement_segments, min_rows=150, top_n=12),
        "where_filtered_overlay_disagreement_stably_lags_market": summarize_segment_weaknesses(filtered_disagreement_segments, min_rows=150, top_n=12),
        "where_advanced_disagreement_beats_market": summarize_segment_strengths(advanced_disagreement_segments, min_rows=150, top_n=12),
        "where_residual_overlay_disagreement_beats_market": summarize_segment_strengths(residual_disagreement_segments, min_rows=150, top_n=12),
        "where_filtered_overlay_disagreement_beats_market": summarize_segment_strengths(filtered_disagreement_segments, min_rows=150, top_n=12),
        "underperformance_clusters": cluster_underperformance(preds, "advanced_features_p1", n_clusters=8),
        "residual_overlay_underperformance_clusters": cluster_underperformance(preds, "residual_overlay_p1", n_clusters=8),
        "feature_notes": {
            "elo": "overall/surface/Slam Elo updated after each date",
            "early_round": "title/final within 7-14 days, recent matches, rest, surface switch, tournament load",
            "travel_schedule": "location/country/continent switches and haversine travel distance from prior known tournament location; location map is intentionally partial and should be expanded",
            "entry_context": "qualifier/wildcard/lucky-loser/protected-ranking columns are wired as zero-valued placeholders until draw/entry ETL is added",
            "challenger_form": "Challenger momentum columns are wired as neutral placeholders until Challenger results are ingested",
            "injury_absence": "prior retirement/walkover and long-layoff proxy flags from match comments/rest days",
            "style_matchup": "rolling score-derived set/game/tiebreak/straight-set proxies; replace/augment with serve-return stats when available",
            "tournament_context": "rolling tournament/surface/round upset and favorite-win rates from prior matches only",
            "slam": "best-of-five flags, Slam records, Slam Elo",
            "surface": "surface Elo and rolling surface win rate",
            "top_player": "top10/top20 flags and records vs top10/top20",
            "market_aware": "market-aware model includes no-vig probability and logit as inputs; research-only baseline for residual/overlay thinking",
            "market_bin_recalibrated": "conservative no-lookahead reliability-bin offsets learned on prior-year training rows; bins without enough training rows keep the original market probability",
            "paper_benchmark": "Grand Slam-only train-before-tournament expanding-window benchmark compares market, logistic, spline-logistic, random forest, XGBoost, linear SVM, advanced voting, and residual overlay with accuracy/log-loss/Brier",
            "statistically_enhanced_abilities": "pre-match ability covariates are estimated only from prior matches: surface ability, score-derived serve/return proxy, best-of-five/Slam ability, recent form ability, and fatigue-adjusted ability",
            "residual_overlay": "fits result - no-vig-market as target, applies segment-tuned shrinkage, and uses specialist residual models in early ATP250, post-title/final, surface-switch, Grand Slam, top10, and early-surface-switch buckets",
            "underperformance_filter": "predicts rows where advanced features are likely worse than market; filtered overlay falls back to market when risk is high",
            "model_market_disagreement": "diagnostic-only scan of rows where a model flips the no-vig market favorite; useful for separating true model overrides from rows where model and market already agree",
            "clv": "live/pre-match odds snapshots and CLV storage are handled by scripts/odds_snapshot_store.py; not used in historical backtest until real snapshots exist",
        },
        "disclaimer": "Research only. No betting execution. Market-aware models use closing odds and must be adapted carefully for pre-match live odds/CLV tracking.",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pred_path = OUT_DIR / f"advanced_feature_predictions_{stamp}.csv"
    report_path = OUT_DIR / f"advanced_feature_model_research_{stamp}.json"
    preds.to_csv(pred_path, index=False)
    preds.to_csv(OUT_DIR / "latest_advanced_feature_predictions.csv", index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "latest_advanced_feature_model_research.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2027)))
    p.add_argument("--test-years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    p.add_argument("--paper-test-years", nargs="+", type=int, default=None, help="Grand Slam paper-style tournament benchmark years; defaults to first test year for fast apples-to-apples replication.")
    args = p.parse_args()
    report = run(args.years, args.test_years, paper_test_years=args.paper_test_years)
    print(json.dumps({
        "generated_at": report["generated_at"],
        "overall_model_comparison": report["overall_model_comparison"],
        "grand_slam_tournament_expanding_benchmark": report["grand_slam_tournament_expanding_benchmark"].get("overall_model_comparison", report["grand_slam_tournament_expanding_benchmark"]),
        "top_underperformance_clusters": report["underperformance_clusters"][:8],
        "top_underperformance_segments": report["where_advanced_underperforms_market"][:12],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
