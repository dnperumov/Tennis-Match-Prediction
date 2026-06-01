"""
Player class for tracking player statistics over time.
"""

import math
import pandas as pd


class Player:
    """Tracks comprehensive player statistics for match prediction."""
    
    def __init__(self, name):
        self.name = name
        self.total_matches = 0
        self.total_wins = 0
        self.surface_matches = {'Hard': 0, 'Clay': 0, 'Grass': 0, 'Carpet': 0, 'Unknown': 0}
        self.surface_wins = {'Hard': 0, 'Clay': 0, 'Grass': 0, 'Carpet': 0, 'Unknown': 0}
        self.ranks = []
        self.heights = []
        self.hands = []
        self.seeds = []
        self.ages = []
        self.last_5_matches = []
        self.last_10_matches = []
        self.head_to_head = {}
        self.matches_against_right_handers = 0
        self.wins_against_right_handers = 0
        self.matches_against_left_handers = 0
        self.wins_against_left_handers = 0
        self.win_streak = 0
        self.loss_streak = 0
        self.tourney_performance = {}
        self.surface_performance = {'Hard': [], 'Clay': [], 'Grass': [], 'Carpet': [], 'Unknown': []}
        self.match_dates = []
        self.match_minutes = []
        self.serve_points_won = []
        self.return_points_won = []
        self.ace_rates = []
        self.double_fault_rates = []

    def update_stats(
        self,
        opponent,
        opponent_hand,
        is_winner,
        surface,
        rank,
        height,
        hand,
        seed,
        age,
        tourney_id,
        match_date=None,
        minutes=None,
        serve_points_won_pct=None,
        return_points_won_pct=None,
        ace_rate=None,
        double_fault_rate=None,
    ):
        """Update player statistics after a match."""
        self.total_matches += 1
        self.ranks.append(rank)
        self.heights.append(height)
        self.hands.append(hand)
        self.seeds.append(seed)
        self.ages.append(age)
        parsed_date = pd.to_datetime(match_date, errors='coerce')
        if pd.notna(parsed_date):
            self.match_dates.append(parsed_date)
        if pd.notna(minutes):
            self.match_minutes.append(float(minutes))
        self._append_bounded(self.serve_points_won, serve_points_won_pct, 20)
        self._append_bounded(self.return_points_won, return_points_won_pct, 20)
        self._append_bounded(self.ace_rates, ace_rate, 20)
        self._append_bounded(self.double_fault_rates, double_fault_rate, 20)

        self.last_5_matches.append(is_winner)
        if len(self.last_5_matches) > 5:
            self.last_5_matches.pop(0)

        self.last_10_matches.append(is_winner)
        if len(self.last_10_matches) > 10:
            self.last_10_matches.pop(0)

        if opponent not in self.head_to_head:
            self.head_to_head[opponent] = {'matches': 0, 'wins': 0}
        self.head_to_head[opponent]['matches'] += 1
        if is_winner:
            self.total_wins += 1
            self.surface_wins[surface] += 1
            self.head_to_head[opponent]['wins'] += 1
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.win_streak = 0
            self.loss_streak += 1

        if opponent_hand == 'R':
            self.matches_against_right_handers += 1
            if is_winner:
                self.wins_against_right_handers += 1
        elif opponent_hand == 'L':
            self.matches_against_left_handers += 1
            if is_winner:
                self.wins_against_left_handers += 1

        self.surface_matches[surface] += 1
        self.surface_performance[surface].append(is_winner)
        if len(self.surface_performance[surface]) > 10:
            self.surface_performance[surface].pop(0)

        if tourney_id not in self.tourney_performance:
            self.tourney_performance[tourney_id] = {'matches': 0, 'wins': 0}
        self.tourney_performance[tourney_id]['matches'] += 1
        if is_winner:
            self.tourney_performance[tourney_id]['wins'] += 1

    def win_percentage(self, surface, prior: float = 0.5, prior_matches: int = 5):
        """Calculate shrinkage-adjusted win percentage on a specific surface."""
        if surface not in self.surface_matches or self.surface_matches[surface] == 0:
            return prior
        return (
            self.surface_wins[surface] + prior * prior_matches
        ) / (
            self.surface_matches[surface] + prior_matches
        )

    def overall_win_percentage(self):
        """Calculate overall win percentage."""
        if self.total_matches == 0:
            return 0.5
        return self.total_wins / self.total_matches

    def last_5_win_percentage(self):
        """Calculate win percentage in last 5 matches."""
        if len(self.last_5_matches) == 0:
            return 0.5
        return sum(self.last_5_matches) / len(self.last_5_matches)

    def last_10_win_percentage(self):
        """Calculate win percentage in last 10 matches."""
        if len(self.last_10_matches) == 0:
            return 0.5
        return sum(self.last_10_matches) / len(self.last_10_matches)

    def surface_last_10_win_percentage(self, surface):
        """Calculate win percentage on surface in last 10 matches."""
        surface_mapping = {0: 'Clay', 1: 'Grass', 2: 'Hard'}
        if isinstance(surface, int):
            surface = surface_mapping.get(surface, 'Unknown')

        if surface not in self.surface_performance:
            return 0.5
        if len(self.surface_performance[surface]) == 0:
            return 0.5
        return sum(self.surface_performance[surface]) / len(self.surface_performance[surface])

    def head_to_head_stats(self, opponent):
        """Get head-to-head statistics against an opponent."""
        if opponent not in self.head_to_head:
            return {'matches': 0, 'wins': 0}
        return self.head_to_head[opponent]

    def win_percentage_against_right_handers(self):
        """Calculate win percentage against right-handed players."""
        if self.matches_against_right_handers == 0:
            return 0
        return self.wins_against_right_handers / self.matches_against_right_handers

    def win_percentage_against_left_handers(self):
        """Calculate win percentage against left-handed players."""
        if self.matches_against_left_handers == 0:
            return 0
        return self.wins_against_left_handers / self.matches_against_left_handers

    def is_top_10(self, rank):
        """Check if player is in top 10."""
        return rank <= 10 if rank else False

    def tourney_win_percentage(self, tourney_id):
        """Calculate win percentage at a specific tournament."""
        if tourney_id not in self.tourney_performance:
            return 0
        if self.tourney_performance[tourney_id]['matches'] == 0:
            return 0
        return self.tourney_performance[tourney_id]['wins'] / self.tourney_performance[tourney_id]['matches']

    def preferred_surface(self):
        """Determine player's preferred surface."""
        if sum(self.surface_matches.values()) == 0:
            return 'Unknown'
        best_surface = max(self.surface_wins, key=lambda x: self.win_percentage(x))
        return best_surface

    def days_since_last_match(self, current_date, default: float = 365.0):
        """Days since last known match before current_date."""
        if not self.match_dates:
            return default
        current = pd.to_datetime(current_date, errors='coerce')
        if pd.isna(current):
            return default
        delta = (current - self.match_dates[-1]).days
        if delta < 0:
            return default
        return min(float(delta), default)

    def matches_last_days(self, current_date, days: int):
        """Count prior matches played within a rolling calendar window."""
        current = pd.to_datetime(current_date, errors='coerce')
        if pd.isna(current):
            return 0
        return sum(0 <= (current - match_date).days <= days for match_date in self.match_dates)

    def avg_minutes(self, n: int = 3, default: float = 90.0):
        values = self.match_minutes[-n:]
        if not values:
            return default
        return sum(values) / len(values)

    def rolling_average(self, values, n: int = 10, default: float = 0.5):
        recent = [value for value in values[-n:] if pd.notna(value)]
        if not recent:
            return default
        return sum(recent) / len(recent)

    def recent_serve_points_won(self):
        return self.rolling_average(self.serve_points_won)

    def recent_return_points_won(self):
        return self.rolling_average(self.return_points_won)

    def recent_ace_rate(self):
        return self.rolling_average(self.ace_rates, default=0.05)

    def recent_double_fault_rate(self):
        return self.rolling_average(self.double_fault_rates, default=0.03)

    @staticmethod
    def _append_bounded(values, value, limit: int):
        if value is None or pd.isna(value) or not math.isfinite(float(value)):
            return
        values.append(float(value))
        if len(values) > limit:
            values.pop(0)
