"""
Core feature engineering for tennis match prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import LabelEncoder
from ..data.player import Player


class FeatureEngineer:
    """Creates features for tennis match prediction."""
    
    def __init__(self):
        self.surface_encoder = None
        self.surface_mapping = {
            'Hard': 'Hard', 'hard': 'Hard', 'HARD': 'Hard',
            'Clay': 'Clay', 'clay': 'Clay', 'CLAY': 'Clay',
            'Grass': 'Grass', 'grass': 'Grass', 'GRASS': 'Grass'
        }
    
    def create_features(self, df: pd.DataFrame, players: Dict, encoder: LabelEncoder = None) -> pd.DataFrame:
        """
        Create features for match prediction.
        
        Args:
            df: DataFrame with match data
            players: Dictionary of Player objects
            encoder: Optional pre-fitted surface encoder
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Normalize surface names
        df['surface'] = df['surface'].map(self.surface_mapping).fillna('Unknown')
        
        # Encode surface
        if encoder is None:
            encoder = LabelEncoder()
            surfaces_with_unknown = pd.concat([df['surface'], pd.Series(['Unknown'])])
            encoder.fit(surfaces_with_unknown)
        
        self.surface_encoder = encoder
        df['surface_encoded'] = encoder.transform(df['surface'])
        
        # Basic features
        df['rank_diff'] = df['player1_rank'] - df['player2_rank']
        df['log_rank_diff'] = np.log1p(np.abs(df['player1_rank'] - df['player2_rank']))
        df['top_10_vs_not'] = ((df['player1_rank'] <= 10) & (df['player2_rank'] > 10)) | \
                              ((df['player2_rank'] <= 10) & (df['player1_rank'] > 10))
        
        df['age_diff'] = df['player1_age'] - df['player2_age']
        df['young_vs_old'] = ((df['player1_age'] < 25) & (df['player2_age'] > 30)) | \
                            ((df['player2_age'] < 25) & (df['player1_age'] > 30))
        
        df['player1_seed'] = pd.to_numeric(df['player1_seed'], errors='coerce')
        df['player2_seed'] = pd.to_numeric(df['player2_seed'], errors='coerce')
        df['seed_diff'] = df['player1_seed'] - df['player2_seed']
        df['seeded_vs_unseeded'] = (df['player1_seed'].notna() & df['player2_seed'].isna()) | \
                                   (df['player2_seed'].notna() & df['player1_seed'].isna())
        
        df['height_diff'] = df['player1_ht'] - df['player2_ht']
        df['tall_vs_short'] = ((df['player1_ht'] > 190) & (df['player2_ht'] < 180)) | \
                             ((df['player2_ht'] > 190) & (df['player1_ht'] < 180))
        
        df['same_hand'] = df['player1_hand'] == df['player2_hand']
        df['left_vs_right'] = ((df['player1_hand'] == 'L') & (df['player2_hand'] == 'R')) | \
                             ((df['player2_hand'] == 'L') & (df['player1_hand'] == 'R'))
        
        # Win percentages
        df['player1_last_5_win_percentage'] = df.apply(
            lambda row: players.get(row['player1'], Player(row['player1'])).last_5_win_percentage(), axis=1
        )
        df['player2_last_5_win_percentage'] = df.apply(
            lambda row: players.get(row['player2'], Player(row['player2'])).last_5_win_percentage(), axis=1
        )
        df['player1_last_10_win_percentage'] = df.apply(
            lambda row: players.get(row['player1'], Player(row['player1'])).last_10_win_percentage(), axis=1
        )
        df['player2_last_10_win_percentage'] = df.apply(
            lambda row: players.get(row['player2'], Player(row['player2'])).last_10_win_percentage(), axis=1
        )
        
        df['player1_surface_last_10_win_percentage'] = df.apply(
            lambda row: players.get(row['player1'], Player(row['player1'])).surface_last_10_win_percentage(row['surface']), axis=1
        )
        df['player2_surface_last_10_win_percentage'] = df.apply(
            lambda row: players.get(row['player2'], Player(row['player2'])).surface_last_10_win_percentage(row['surface']), axis=1
        )
        
        # Preferred surface
        df['player1_preferred_surface'] = df.apply(
            lambda row: players.get(row['player1'], Player(row['player1'])).preferred_surface(), axis=1
        )
        df['player2_preferred_surface'] = df.apply(
            lambda row: players.get(row['player2'], Player(row['player2'])).preferred_surface(), axis=1
        )
        
        # Head-to-head
        df['head_to_head_wins_p1'] = df.apply(
            lambda row: players.get(row['player1'], Player(row['player1'])).head_to_head_stats(row['player2']).get('wins', 0), axis=1
        )
        df['head_to_head_wins_p2'] = df.apply(
            lambda row: players.get(row['player2'], Player(row['player2'])).head_to_head_stats(row['player1']).get('wins', 0), axis=1
        )
        
        # Surface match
        df['player1_surface_match'] = df.apply(
            lambda row: row['surface'] == players.get(row['player1'], Player(row['player1'])).preferred_surface(), axis=1
        )
        df['player2_surface_match'] = df.apply(
            lambda row: row['surface'] == players.get(row['player2'], Player(row['player2'])).preferred_surface(), axis=1
        )
        df['surface_preference_diff'] = df['player1_surface_match'].astype(int) - df['player2_surface_match'].astype(int)
        
        return df
    
    def create_player_vs_player_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create player vs player dataset with shuffled assignments."""
        df_new = df.copy()
        swap = np.random.rand(len(df)) < 0.5
        
        df_new['player1'] = np.where(swap, df_new['winner_name'], df_new['loser_name'])
        df_new['player2'] = np.where(swap, df_new['loser_name'], df_new['winner_name'])
        df_new['player1_seed'] = np.where(swap, df_new['winner_seed'], df_new['loser_seed'])
        df_new['player2_seed'] = np.where(swap, df_new['loser_seed'], df_new['winner_seed'])
        df_new['player1_rank'] = np.where(swap, df_new['winner_rank'], df_new['loser_rank'])
        df_new['player2_rank'] = np.where(swap, df_new['loser_rank'], df_new['winner_rank'])
        df_new['player1_age'] = np.where(swap, df_new['winner_age'], df_new['loser_age'])
        df_new['player2_age'] = np.where(swap, df_new['loser_age'], df_new['winner_age'])
        df_new['player1_ht'] = np.where(swap, df_new['winner_ht'], df_new['loser_ht'])
        df_new['player2_ht'] = np.where(swap, df_new['loser_ht'], df_new['winner_ht'])
        df_new['player1_hand'] = np.where(swap, df_new['winner_hand'], df_new['loser_hand'])
        df_new['player2_hand'] = np.where(swap, df_new['loser_hand'], df_new['winner_hand'])
        df_new['result'] = np.where(swap, 1, 2)
        
        return df_new

