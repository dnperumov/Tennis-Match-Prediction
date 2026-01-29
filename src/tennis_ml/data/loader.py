"""
Data loading and preprocessing for tennis match prediction.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from .player import Player


class DataLoader:
    """Handles loading and preprocessing of ATP match data."""
    
    def __init__(self, data_dir: str = 'tennis_datav2'):
        self.data_dir = data_dir
        self.players: Dict[str, Player] = {}
        
    def load_data(self, years: range = range(2000, 2025), 
                  download_missing: bool = True) -> pd.DataFrame:
        """
        Load ATP match data for specified years.
        
        Args:
            years: Range of years to load
            download_missing: Whether to download missing files from GitHub
            
        Returns:
            Combined DataFrame of all matches
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        dfs = []
        for year in years:
            file_path = os.path.join(self.data_dir, f'atp_matches_{year}.csv')
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
            elif download_missing:
                url = f'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv'
                try:
                    df = pd.read_csv(url)
                    df.to_csv(file_path, index=False)
                except Exception as e:
                    print(f"Warning: Could not download data for {year}: {e}")
                    continue
            else:
                print(f"Warning: File not found for {year}: {file_path}")
                continue
            
            dfs.append(df)
        
        if not dfs:
            raise ValueError("No data files found or loaded")
        
        atp_matches = pd.concat(dfs, ignore_index=True)
        atp_matches['tourney_date'] = pd.to_datetime(atp_matches['tourney_date'], format='%Y%m%d')
        atp_matches = atp_matches.sort_values(by='tourney_date').reset_index(drop=True)
        
        return atp_matches
    
    def split_data_chronologically(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically for temporal validation.
        
        Args:
            df: Full dataset
            
        Returns:
            Tuple of (initial_50, next_25, final_25) splits
        """
        total_matches = len(df)
        initial_50 = df.iloc[:total_matches//2].copy()
        next_25 = df.iloc[total_matches//2:total_matches*3//4].copy()
        final_25 = df.iloc[total_matches*3//4:].copy()
        
        return initial_50, next_25, final_25
    
    def initialize_players(self, df: pd.DataFrame):
        """Initialize player statistics from historical data."""
        self.players = {}
        
        for index, row in df.iterrows():
            self._update_player_stats(row, is_winner=True)
            self._update_player_stats(row, is_winner=False)
    
    def _update_player_stats(self, row: pd.Series, is_winner: bool):
        """Update statistics for a player after a match."""
        name = row['winner_name'] if is_winner else row['loser_name']
        opponent = row['loser_name'] if is_winner else row['winner_name']
        opponent_hand = row['loser_hand'] if is_winner else row['winner_hand']
        surface = row['surface'] if pd.notna(row['surface']) else 'Unknown'
        rank = row['winner_rank'] if is_winner else row['loser_rank']
        height = row['winner_ht'] if is_winner else row['loser_ht']
        hand = row['winner_hand'] if is_winner else row['loser_hand']
        seed = row['winner_seed'] if is_winner else row['loser_seed']
        age = row['winner_age'] if is_winner else row['loser_age']
        tourney_id = row['tourney_id']

        if name not in self.players:
            self.players[name] = Player(name)

        self.players[name].update_stats(
            opponent=opponent,
            opponent_hand=opponent_hand,
            is_winner=is_winner,
            surface=surface,
            rank=rank,
            height=height,
            hand=hand,
            seed=seed,
            age=age,
            tourney_id=tourney_id
        )
    
    def update_players_incremental(self, df: pd.DataFrame):
        """Update player statistics incrementally (for validation/test sets)."""
        for index, row in df.iterrows():
            self._update_player_stats(row, is_winner=True)
            self._update_player_stats(row, is_winner=False)

