"""
Match prediction using trained models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from ..data.player import Player
from ..features.engineering import FeatureEngineer
from ..features.betting_features import BettingFeatureEngineer


class MatchPredictor:
    """Predicts match outcomes using trained models."""
    
    def __init__(self, model: any, scaler: any, features: list, 
                 players: Dict[str, Player], imputer: any = None):
        self.model = model
        self.scaler = scaler
        self.features = features
        self.players = players
        self.imputer = imputer
        self.feature_engineer = FeatureEngineer()
        self.betting_feature_engineer = BettingFeatureEngineer()
    
    def predict(self, player1_name: str, player2_name: str, 
                surface: str, tournament_importance: int = 3) -> Dict:
        """
        Predict match outcome.
        
        Args:
            player1_name: Name of first player
            player2_name: Name of second player
            surface: Surface type (Hard, Clay, Grass)
            tournament_importance: Tournament importance level (1-5)
            
        Returns:
            Dictionary with prediction results
        """
        player1 = self.players.get(player1_name)
        player2 = self.players.get(player2_name)
        
        if not player1 or not player2:
            raise ValueError(f"Player not found: {player1_name} or {player2_name}")
        
        # Create feature vector
        feature_vector = self._create_feature_vector(
            player1, player2, surface, tournament_importance
        )
        
        # Handle missing values
        if self.imputer is not None:
            feature_vector = self.imputer.transform([feature_vector])[0]
        
        # Scale features
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Make prediction
        prediction = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        
        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'player1_probability': probabilities[1] if len(probabilities) > 1 else probabilities[0],
            'player2_probability': probabilities[0] if len(probabilities) > 1 else 1 - probabilities[0],
            'confidence': abs(probabilities[1] - 0.5) * 2 if len(probabilities) > 1 else 0,
            'player1_name': player1_name,
            'player2_name': player2_name,
            'surface': surface
        }
    
    def _create_feature_vector(self, player1: Player, player2: Player,
                              surface: str, tournament_importance: int) -> list:
        """Create feature vector for prediction."""
        # Basic features
        rank_diff = (player1.ranks[-1] if player1.ranks else 999) - \
                   (player2.ranks[-1] if player2.ranks else 999)
        log_rank_diff = np.log(abs(rank_diff) + 1)
        top_10_vs_not = int(
            player1.is_top_10(player1.ranks[-1] if player1.ranks else 999) and
            not player2.is_top_10(player2.ranks[-1] if player2.ranks else 999)
        )
        
        age_diff = (player1.ages[-1] if player1.ages else 25) - \
                  (player2.ages[-1] if player2.ages else 25)
        young_vs_old = int(
            (player1.ages[-1] if player1.ages else 25) <
            (player2.ages[-1] if player2.ages else 25)
        )
        
        seed_diff = (player1.seeds[-1] or 999) - (player2.seeds[-1] or 999)
        seeded_vs_unseeded = int(
            (player1.seeds[-1] is not None) and (player2.seeds[-1] is None)
        )
        
        height_diff = (player1.heights[-1] if player1.heights else 180) - \
                     (player2.heights[-1] if player2.heights else 180)
        tall_vs_short = int(
            (player1.heights[-1] if player1.heights else 180) >
            (player2.heights[-1] if player2.heights else 180)
        )
        
        same_hand = int(
            (player1.hands[-1] if player1.hands else 'R') ==
            (player2.hands[-1] if player2.hands else 'R')
        )
        left_vs_right = int(
            ((player1.hands[-1] if player1.hands else 'R') == 'L' and
             (player2.hands[-1] if player2.hands else 'R') == 'R') or
            ((player2.hands[-1] if player2.hands else 'R') == 'L' and
             (player1.hands[-1] if player1.hands else 'R') == 'R')
        )
        
        # Win percentages
        player1_last_5_win_percentage = player1.last_5_win_percentage()
        player2_last_5_win_percentage = player2.last_5_win_percentage()
        player1_last_10_win_percentage = player1.last_10_win_percentage()
        player2_last_10_win_percentage = player2.last_10_win_percentage()
        
        player1_surface_last_10_win_percentage = player1.surface_last_10_win_percentage(surface)
        player2_surface_last_10_win_percentage = player2.surface_last_10_win_percentage(surface)
        
        # Surface preferences
        player1_preferred_surface = int(player1.preferred_surface() == surface)
        player2_preferred_surface = int(player2.preferred_surface() == surface)
        player1_surface_match = int(player1_preferred_surface)
        player2_surface_match = int(player2_preferred_surface)
        surface_preference_diff = player1_preferred_surface - player2_preferred_surface
        
        # Head-to-head
        head_to_head_wins_p1 = player1.head_to_head_stats(player2.name)['wins']
        head_to_head_wins_p2 = player2.head_to_head_stats(player1.name)['wins']
        
        # Surface encoding
        surface_mapping = {'Clay': 0, 'Grass': 1, 'Hard': 2, 'Unknown': 3}
        surface_encoded = surface_mapping.get(surface, 3)
        
        # Betting features
        player1_momentum = self.betting_feature_engineer._calculate_weighted_form(
            player1.name, self.players
        )
        player2_momentum = self.betting_feature_engineer._calculate_weighted_form(
            player2.name, self.players
        )
        player1_confidence = self.betting_feature_engineer._calculate_confidence(
            player1.name, self.players
        )
        player2_confidence = self.betting_feature_engineer._calculate_confidence(
            player2.name, self.players
        )
        player1_surface_advantage = player1.win_percentage(surface)
        player2_surface_advantage = player2.win_percentage(surface)
        
        # Build feature vector in same order as training
        feature_vector = [
            rank_diff, log_rank_diff, top_10_vs_not,
            age_diff, young_vs_old, seed_diff, seeded_vs_unseeded,
            height_diff, tall_vs_short, same_hand, left_vs_right,
            player1_last_5_win_percentage, player2_last_5_win_percentage,
            player1_last_10_win_percentage, player2_last_10_win_percentage,
            player1_surface_last_10_win_percentage, player2_surface_last_10_win_percentage,
            player1_preferred_surface, player2_preferred_surface,
            player1_surface_match, player2_surface_match,
            surface_preference_diff, head_to_head_wins_p1, head_to_head_wins_p2,
            surface_encoded, tournament_importance,
            player1_momentum, player2_momentum,
            player1_confidence, player2_confidence,
            player1_surface_advantage, player2_surface_advantage
        ]
        
        # Ensure correct length
        while len(feature_vector) < len(self.features):
            feature_vector.append(0)
        
        return feature_vector[:len(self.features)]

