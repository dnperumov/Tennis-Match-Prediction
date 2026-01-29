"""
Betting-specific feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict
from ..data.player import Player


class BettingFeatureEngineer:
    """Adds betting-specific features for sports betting applications."""
    
    def __init__(self):
        pass
    
    def add_betting_features(self, df: pd.DataFrame, players_dict: Dict) -> pd.DataFrame:
        """Add advanced features for betting applications."""
        df = df.copy()
        
        # Momentum and form features
        df['player1_momentum'] = df.apply(
            lambda row: self._calculate_weighted_form(row['player1'], players_dict), axis=1
        )
        df['player2_momentum'] = df.apply(
            lambda row: self._calculate_weighted_form(row['player2'], players_dict), axis=1
        )
        
        # Fatigue indicators
        df['player1_matches_last_7d'] = df.apply(
            lambda row: self._estimate_recent_matches(row['player1'], players_dict), axis=1
        )
        df['player2_matches_last_7d'] = df.apply(
            lambda row: self._estimate_recent_matches(row['player2'], players_dict), axis=1
        )
        
        # Tournament context
        tournament_importance = {
            'G': 5,  # Grand Slam
            'M': 4,  # Masters 1000
            'A': 3,  # ATP 500
            'C': 2,  # ATP 250
            'F': 1,  # Futures/Challenger
            'D': 1   # Davis Cup
        }
        df['tournament_importance'] = df['tourney_level'].map(tournament_importance).fillna(1)
        
        # Confidence indicators
        df['player1_confidence'] = df.apply(
            lambda row: self._calculate_confidence(row['player1'], players_dict), axis=1
        )
        df['player2_confidence'] = df.apply(
            lambda row: self._calculate_confidence(row['player2'], players_dict), axis=1
        )
        
        # Surface advantages
        df['player1_surface_advantage'] = df.apply(
            lambda row: players_dict.get(row['player1'], Player(row['player1'])).win_percentage(row['surface']), axis=1
        )
        df['player2_surface_advantage'] = df.apply(
            lambda row: players_dict.get(row['player2'], Player(row['player2'])).win_percentage(row['surface']), axis=1
        )
        
        return df
    
    def _calculate_weighted_form(self, player_name: str, players_dict: Dict) -> float:
        """Calculate weighted recent form."""
        if player_name not in players_dict:
            return 0.5
        
        player = players_dict[player_name]
        if len(player.last_5_matches) == 0:
            return 0.5
        
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        recent_matches = player.last_5_matches.copy()
        
        if len(recent_matches) == 0:
            return 0.5
        
        while len(recent_matches) < 5:
            recent_matches = [0] + recent_matches
        
        return np.average(recent_matches, weights=weights[:len(recent_matches)])
    
    def _estimate_recent_matches(self, player_name: str, players_dict: Dict) -> int:
        """Estimate recent match activity."""
        if player_name not in players_dict:
            return 0
        
        player = players_dict[player_name]
        return min(player.total_matches // 30, 10)
    
    def _calculate_confidence(self, player_name: str, players_dict: Dict) -> float:
        """Calculate player confidence."""
        if player_name not in players_dict:
            return 0.5
        
        player = players_dict[player_name]
        recent_wins = np.mean(player.last_5_matches) if len(player.last_5_matches) > 0 else 0.5
        win_streak_bonus = min(player.win_streak * 0.1, 0.3)
        ranking_stability = 1.0 if len(player.ranks) > 0 and player.ranks[-1] <= 50 else 0.5
        
        confidence = (recent_wins + win_streak_bonus + ranking_stability) / 3
        return min(max(confidence, 0), 1)

