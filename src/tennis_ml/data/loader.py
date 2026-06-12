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
        
    def load_data(self, years: range = range(2000, 2027), 
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

            df = self._append_supplement(df, year)
            dfs.append(df)
        
        if not dfs:
            raise ValueError("No data files found or loaded")
        
        atp_matches = pd.concat(dfs, ignore_index=True)
        atp_matches['tourney_date'] = pd.to_datetime(atp_matches['tourney_date'], format='%Y%m%d')
        sort_column = 'match_date' if 'match_date' in atp_matches.columns else 'tourney_date'
        atp_matches = atp_matches.sort_values(by=sort_column).reset_index(drop=True)
        
        return atp_matches

    def _append_supplement(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Append agent-scraped supplement rows for a year, if present.

        Supplement files (atp_matches_{year}_supplement.csv, produced by
        scripts/refresh_match_data.py) gap-fill matches not yet covered by
        the Sackmann yearly file. Exact duplicate (winner_name, loser_name,
        tourney_date) pairs are dropped, keeping the main-file row.
        """
        supplement_path = os.path.join(self.data_dir, f'atp_matches_{year}_supplement.csv')
        if not os.path.exists(supplement_path):
            return df

        try:
            supplement = pd.read_csv(supplement_path)
        except Exception as e:
            print(f"Warning: Could not read supplement for {year}: {e}")
            return df

        if supplement.empty:
            return df

        combined = pd.concat([df, supplement], ignore_index=True)
        combined['tourney_date'] = pd.to_numeric(combined['tourney_date'], errors='coerce')
        combined = combined.drop_duplicates(
            subset=['winner_name', 'loser_name', 'tourney_date'], keep='first'
        ).reset_index(drop=True)
        return combined

    def split_data_chronologically(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically for temporal validation.
        
        Args:
            df: Full dataset
            
        Returns:
            Tuple of (initial_50, next_25, final_25) splits
        """
        date_column = 'match_date' if 'match_date' in df.columns else 'tourney_date'
        sort_columns = [col for col in [date_column, 'tourney_date', 'tourney_id', 'match_num'] if col in df.columns]
        df = df.sort_values(by=sort_columns).reset_index(drop=True)
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
        match_date = row.get('match_date', row.get('tourney_date'))
        minutes = row.get('minutes')
        serve_points_won_pct, return_points_won_pct, ace_rate, double_fault_rate = self._point_stats(row, is_winner)

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
            tourney_id=tourney_id,
            match_date=match_date,
            minutes=minutes,
            serve_points_won_pct=serve_points_won_pct,
            return_points_won_pct=return_points_won_pct,
            ace_rate=ace_rate,
            double_fault_rate=double_fault_rate
        )
    
    def update_players_incremental(self, df: pd.DataFrame):
        """Update player statistics incrementally (for validation/test sets)."""
        for index, row in df.iterrows():
            self._update_player_stats(row, is_winner=True)
            self._update_player_stats(row, is_winner=False)

    @staticmethod
    def _point_stats(row: pd.Series, is_winner: bool):
        prefix = 'w' if is_winner else 'l'
        opponent_prefix = 'l' if is_winner else 'w'

        service_points = pd.to_numeric(row.get(f'{prefix}_svpt'), errors='coerce')
        first_won = pd.to_numeric(row.get(f'{prefix}_1stWon'), errors='coerce')
        second_won = pd.to_numeric(row.get(f'{prefix}_2ndWon'), errors='coerce')
        opponent_service_points = pd.to_numeric(row.get(f'{opponent_prefix}_svpt'), errors='coerce')
        opponent_first_won = pd.to_numeric(row.get(f'{opponent_prefix}_1stWon'), errors='coerce')
        opponent_second_won = pd.to_numeric(row.get(f'{opponent_prefix}_2ndWon'), errors='coerce')
        aces = pd.to_numeric(row.get(f'{prefix}_ace'), errors='coerce')
        double_faults = pd.to_numeric(row.get(f'{prefix}_df'), errors='coerce')

        serve_points_won = None
        return_points_won = None
        ace_rate = None
        double_fault_rate = None
        if pd.notna(service_points) and service_points > 0:
            serve_points_won = (first_won + second_won) / service_points
            ace_rate = aces / service_points
            double_fault_rate = double_faults / service_points
        if pd.notna(opponent_service_points) and opponent_service_points > 0:
            return_points_won = 1 - ((opponent_first_won + opponent_second_won) / opponent_service_points)

        return serve_points_won, return_points_won, ace_rate, double_fault_rate
