"""
Rolling Elo ratings for tennis match features.
"""

import math
from collections import defaultdict

import pandas as pd


class EloTracker:
    """Tracks overall and surface-specific Elo ratings chronologically."""

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.overall = defaultdict(lambda: initial_rating)
        self.surface = defaultdict(lambda: defaultdict(lambda: initial_rating))
        self.recent = defaultdict(lambda: initial_rating)
        self.recent_surface = defaultdict(lambda: defaultdict(lambda: initial_rating))
        self.matches = defaultdict(int)
        self.surface_matches = defaultdict(lambda: defaultdict(int))
        self.last_match_date = {}

    def initialize(self, df: pd.DataFrame):
        """Warm up ratings from historical matches in chronological order."""
        sort_columns = [col for col in ['match_date', 'tourney_date', 'tourney_id', 'match_num'] if col in df.columns]
        for _, row in df.sort_values(by=sort_columns).iterrows():
            surface = row['surface'] if pd.notna(row['surface']) else 'Unknown'
            self.update(row['winner_name'], row['loser_name'], surface, match_date=row.get('match_date', row.get('tourney_date')))

    def features(self, player1: str, player2: str, surface: str) -> dict:
        """Return pre-match Elo features for player1 vs player2."""
        p1_elo = self.overall[player1]
        p2_elo = self.overall[player2]
        p1_surface_elo = self.surface[surface][player1]
        p2_surface_elo = self.surface[surface][player2]
        p1_recent_elo = self.recent[player1]
        p2_recent_elo = self.recent[player2]
        p1_recent_surface_elo = self.recent_surface[surface][player1]
        p2_recent_surface_elo = self.recent_surface[surface][player2]

        return {
            'player1_elo': p1_elo,
            'player2_elo': p2_elo,
            'elo_diff': p1_elo - p2_elo,
            'elo_prob_player1': self.expected_score(p1_elo, p2_elo),
            'player1_surface_elo': p1_surface_elo,
            'player2_surface_elo': p2_surface_elo,
            'surface_elo_diff': p1_surface_elo - p2_surface_elo,
            'surface_elo_prob_player1': self.expected_score(p1_surface_elo, p2_surface_elo),
            'player1_elo_matches': self.matches[player1],
            'player2_elo_matches': self.matches[player2],
            'player1_surface_elo_matches': self.surface_matches[surface][player1],
            'player2_surface_elo_matches': self.surface_matches[surface][player2],
            'player1_recent_elo': p1_recent_elo,
            'player2_recent_elo': p2_recent_elo,
            'recent_elo_diff': p1_recent_elo - p2_recent_elo,
            'recent_elo_prob_player1': self.expected_score(p1_recent_elo, p2_recent_elo),
            'player1_recent_surface_elo': p1_recent_surface_elo,
            'player2_recent_surface_elo': p2_recent_surface_elo,
            'recent_surface_elo_diff': p1_recent_surface_elo - p2_recent_surface_elo,
            'recent_surface_elo_prob_player1': self.expected_score(p1_recent_surface_elo, p2_recent_surface_elo),
            'player1_elo_uncertainty': self._uncertainty(self.matches[player1]),
            'player2_elo_uncertainty': self._uncertainty(self.matches[player2]),
        }

    def update(self, winner: str, loser: str, surface: str, match_date=None):
        """Update ratings after a completed match."""
        self._apply_inactivity_decay(winner, match_date)
        self._apply_inactivity_decay(loser, match_date)
        self._update_rating(self.overall, winner, loser, self._dynamic_k(winner), self._dynamic_k(loser))
        self._update_rating(
            self.surface[surface],
            winner,
            loser,
            self._dynamic_k(winner, surface),
            self._dynamic_k(loser, surface)
        )
        self._update_rating(self.recent, winner, loser, self.k_factor * 1.5, self.k_factor * 1.5)
        self._update_rating(self.recent_surface[surface], winner, loser, self.k_factor * 1.6, self.k_factor * 1.6)
        self.matches[winner] += 1
        self.matches[loser] += 1
        self.surface_matches[surface][winner] += 1
        self.surface_matches[surface][loser] += 1
        parsed_date = pd.to_datetime(match_date, errors='coerce')
        if pd.notna(parsed_date):
            self.last_match_date[winner] = parsed_date
            self.last_match_date[loser] = parsed_date

    def _update_rating(self, ratings, winner: str, loser: str, winner_k: float, loser_k: float):
        winner_rating = ratings[winner]
        loser_rating = ratings[loser]
        expected_winner = self.expected_score(winner_rating, loser_rating)
        ratings[winner] = winner_rating + winner_k * (1 - expected_winner)
        ratings[loser] = loser_rating + loser_k * (0 - (1 - expected_winner))

    def _dynamic_k(self, player: str, surface: str = None) -> float:
        match_count = self.surface_matches[surface][player] if surface is not None else self.matches[player]
        uncertainty_boost = self._uncertainty(match_count)
        return max(12.0, min(48.0, self.k_factor * uncertainty_boost))

    @staticmethod
    def _uncertainty(match_count: int) -> float:
        return 1.0 + 1.0 / math.sqrt(match_count + 1)

    def _apply_inactivity_decay(self, player: str, match_date):
        parsed_date = pd.to_datetime(match_date, errors='coerce')
        if pd.isna(parsed_date) or player not in self.last_match_date:
            return
        days_inactive = max(0, (parsed_date - self.last_match_date[player]).days)
        if days_inactive < 60:
            return
        yearly_decay = 0.08
        factor = math.exp(-yearly_decay * days_inactive / 365)
        for ratings in [self.overall, self.recent]:
            ratings[player] = self.initial_rating + (ratings[player] - self.initial_rating) * factor
        for surface_ratings in self.surface.values():
            surface_ratings[player] = self.initial_rating + (surface_ratings[player] - self.initial_rating) * factor
        for surface_ratings in self.recent_surface.values():
            surface_ratings[player] = self.initial_rating + (surface_ratings[player] - self.initial_rating) * factor

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
