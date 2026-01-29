"""
Player class for tracking player statistics over time.
"""


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

    def update_stats(self, opponent, opponent_hand, is_winner, surface, rank, height, hand, seed, age, tourney_id):
        """Update player statistics after a match."""
        self.total_matches += 1
        self.ranks.append(rank)
        self.heights.append(height)
        self.hands.append(hand)
        self.seeds.append(seed)
        self.ages.append(age)

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

    def win_percentage(self, surface):
        """Calculate win percentage on a specific surface."""
        if self.surface_matches[surface] == 0:
            return 0
        return self.surface_wins[surface] / self.surface_matches[surface]

    def overall_win_percentage(self):
        """Calculate overall win percentage."""
        if self.total_matches == 0:
            return 0
        return self.total_wins / self.total_matches

    def last_5_win_percentage(self):
        """Calculate win percentage in last 5 matches."""
        if len(self.last_5_matches) == 0:
            return 0
        return sum(self.last_5_matches) / len(self.last_5_matches)

    def last_10_win_percentage(self):
        """Calculate win percentage in last 10 matches."""
        if len(self.last_10_matches) == 0:
            return 0
        return sum(self.last_10_matches) / len(self.last_10_matches)

    def surface_last_10_win_percentage(self, surface):
        """Calculate win percentage on surface in last 10 matches."""
        surface_mapping = {0: 'Clay', 1: 'Grass', 2: 'Hard'}
        if isinstance(surface, int):
            surface = surface_mapping.get(surface, 'Unknown')

        if surface not in self.surface_performance:
            return 0
        if len(self.surface_performance[surface]) == 0:
            return 0
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
        best_surface = max(self.surface_wins, key=lambda x: self.win_percentage(x))
        return best_surface

