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

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        players: Dict,
        data_loader,
        betting_feature_engineer=None,
        elo_tracker=None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Create match features without look-ahead bias.

        Each row is featurized from player histories as they existed before
        that match, then player histories are updated with the match outcome.
        """
        rng = np.random.default_rng(random_state)
        feature_rows = []

        sort_columns = [col for col in ['match_date', 'tourney_date', 'tourney_id', 'match_num'] if col in df.columns]
        for _, row in df.sort_values(by=sort_columns).iterrows():
            swap = bool(rng.random() < 0.5)
            surface = self.surface_mapping.get(row.get('surface'), row.get('surface'))
            surface = surface if pd.notna(surface) else 'Unknown'

            if swap:
                player1_name = row['winner_name']
                player2_name = row['loser_name']
                result = 1
                player1_prefix = 'winner'
                player2_prefix = 'loser'
            else:
                player1_name = row['loser_name']
                player2_name = row['winner_name']
                result = 2
                player1_prefix = 'loser'
                player2_prefix = 'winner'

            player1 = players.get(player1_name, Player(player1_name))
            player2 = players.get(player2_name, Player(player2_name))
            player1_rank = row.get(f'{player1_prefix}_rank')
            player2_rank = row.get(f'{player2_prefix}_rank')
            player1_age = row.get(f'{player1_prefix}_age')
            player2_age = row.get(f'{player2_prefix}_age')
            player1_seed = pd.to_numeric(row.get(f'{player1_prefix}_seed'), errors='coerce')
            player2_seed = pd.to_numeric(row.get(f'{player2_prefix}_seed'), errors='coerce')
            player1_ht = row.get(f'{player1_prefix}_ht')
            player2_ht = row.get(f'{player2_prefix}_ht')
            player1_hand = row.get(f'{player1_prefix}_hand')
            player2_hand = row.get(f'{player2_prefix}_hand')
            match_date = row.get('match_date', row.get('tourney_date'))
            round_value = row.get('round')
            round_order = {
                'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
                'QF': 5, 'SF': 6, 'F': 7, 'BR': 6, 'RR': 3,
            }
            tourney_level = row.get('tourney_level')
            draw_size = pd.to_numeric(row.get('draw_size'), errors='coerce')
            best_of = pd.to_numeric(row.get('best_of'), errors='coerce')

            row_features = {
                'player1': player1_name,
                'player2': player2_name,
                'winner_name': row.get('winner_name'),
                'loser_name': row.get('loser_name'),
                'tourney_id': row.get('tourney_id'),
                'tourney_name': row.get('tourney_name'),
                'round': row.get('round'),
                'surface': surface,
                'tourney_date': row.get('tourney_date'),
                'match_date': match_date,
                'tourney_start_date': row.get('tourney_start_date', row.get('tourney_date')),
                'match_date_aligned': row.get('match_date_aligned', False),
                'player1_rank': player1_rank,
                'player2_rank': player2_rank,
                'rank_diff': player1_rank - player2_rank,
                'log_rank_diff': np.log1p(abs(player1_rank - player2_rank)),
                'top_10_vs_not': ((player1_rank <= 10) and (player2_rank > 10)) or
                                 ((player2_rank <= 10) and (player1_rank > 10)),
                'age_diff': player1_age - player2_age,
                'young_vs_old': ((player1_age < 25) and (player2_age > 30)) or
                                ((player2_age < 25) and (player1_age > 30)),
                'seed_diff': player1_seed - player2_seed,
                'seeded_vs_unseeded': (pd.notna(player1_seed) and pd.isna(player2_seed)) or
                                      (pd.notna(player2_seed) and pd.isna(player1_seed)),
                'height_diff': player1_ht - player2_ht,
                'tall_vs_short': ((player1_ht > 190) and (player2_ht < 180)) or
                                ((player2_ht > 190) and (player1_ht < 180)),
                'same_hand': player1_hand == player2_hand,
                'left_vs_right': ((player1_hand == 'L') and (player2_hand == 'R')) or
                                 ((player2_hand == 'L') and (player1_hand == 'R')),
                'player1_last_5_win_percentage': player1.last_5_win_percentage(),
                'player2_last_5_win_percentage': player2.last_5_win_percentage(),
                'player1_last_10_win_percentage': player1.last_10_win_percentage(),
                'player2_last_10_win_percentage': player2.last_10_win_percentage(),
                'player1_surface_last_10_win_percentage': player1.surface_last_10_win_percentage(surface),
                'player2_surface_last_10_win_percentage': player2.surface_last_10_win_percentage(surface),
                'player1_surface_match': surface == player1.preferred_surface(),
                'player2_surface_match': surface == player2.preferred_surface(),
                'surface_preference_diff': int(surface == player1.preferred_surface()) -
                                           int(surface == player2.preferred_surface()),
                'head_to_head_wins_p1': player1.head_to_head_stats(player2_name).get('wins', 0),
                'head_to_head_wins_p2': player2.head_to_head_stats(player1_name).get('wins', 0),
                'surface_encoded': {'Clay': 0, 'Grass': 1, 'Hard': 2, 'Unknown': 3}.get(surface, 3),
                'round_encoded': round_order.get(round_value, 0),
                'best_of': best_of,
                'draw_size': draw_size,
                'is_grand_slam': tourney_level == 'G',
                'is_davis_cup': tourney_level == 'D',
                'is_final': round_value == 'F',
                'is_early_round': round_value in {'R128', 'R64', 'R32'},
                'player1_days_since_last_match': player1.days_since_last_match(match_date),
                'player2_days_since_last_match': player2.days_since_last_match(match_date),
                'days_since_last_match_diff': player1.days_since_last_match(match_date) -
                                              player2.days_since_last_match(match_date),
                'player1_matches_last_7d': player1.matches_last_days(match_date, 7),
                'player2_matches_last_7d': player2.matches_last_days(match_date, 7),
                'matches_last_7d_diff': player1.matches_last_days(match_date, 7) -
                                        player2.matches_last_days(match_date, 7),
                'player1_matches_last_14d': player1.matches_last_days(match_date, 14),
                'player2_matches_last_14d': player2.matches_last_days(match_date, 14),
                'matches_last_14d_diff': player1.matches_last_days(match_date, 14) -
                                         player2.matches_last_days(match_date, 14),
                'player1_matches_last_30d': player1.matches_last_days(match_date, 30),
                'player2_matches_last_30d': player2.matches_last_days(match_date, 30),
                'matches_last_30d_diff': player1.matches_last_days(match_date, 30) -
                                         player2.matches_last_days(match_date, 30),
                'player1_avg_minutes_last_3': player1.avg_minutes(3),
                'player2_avg_minutes_last_3': player2.avg_minutes(3),
                'avg_minutes_last_3_diff': player1.avg_minutes(3) - player2.avg_minutes(3),
                'player1_serve_points_won_l10': player1.recent_serve_points_won(),
                'player2_serve_points_won_l10': player2.recent_serve_points_won(),
                'serve_points_won_l10_diff': player1.recent_serve_points_won() -
                                             player2.recent_serve_points_won(),
                'player1_return_points_won_l10': player1.recent_return_points_won(),
                'player2_return_points_won_l10': player2.recent_return_points_won(),
                'return_points_won_l10_diff': player1.recent_return_points_won() -
                                              player2.recent_return_points_won(),
                'player1_ace_rate_l10': player1.recent_ace_rate(),
                'player2_ace_rate_l10': player2.recent_ace_rate(),
                'ace_rate_l10_diff': player1.recent_ace_rate() - player2.recent_ace_rate(),
                'player1_double_fault_rate_l10': player1.recent_double_fault_rate(),
                'player2_double_fault_rate_l10': player2.recent_double_fault_rate(),
                'double_fault_rate_l10_diff': player1.recent_double_fault_rate() -
                                             player2.recent_double_fault_rate(),
                'result': result
            }

            if elo_tracker is not None:
                row_features.update(elo_tracker.features(player1_name, player2_name, surface))

            if betting_feature_engineer is not None:
                tournament_importance = {
                    'G': 5,
                    'M': 4,
                    'A': 3,
                    'C': 2,
                    'F': 1,
                    'D': 1
                }
                row_features.update({
                    'tournament_importance': tournament_importance.get(row.get('tourney_level'), 1),
                    'player1_momentum': betting_feature_engineer._calculate_weighted_form(player1_name, players),
                    'player2_momentum': betting_feature_engineer._calculate_weighted_form(player2_name, players),
                    'player1_confidence': betting_feature_engineer._calculate_confidence(player1_name, players),
                    'player2_confidence': betting_feature_engineer._calculate_confidence(player2_name, players),
                    'player1_surface_advantage': player1.win_percentage(surface),
                    'player2_surface_advantage': player2.win_percentage(surface)
                })

            feature_rows.append(row_features)
            if elo_tracker is not None:
                elo_tracker.update(row['winner_name'], row['loser_name'], surface, match_date=match_date)
            data_loader._update_player_stats(row, is_winner=True)
            data_loader._update_player_stats(row, is_winner=False)

        return pd.DataFrame(feature_rows)
