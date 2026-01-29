"""
Match TA scraped data to base dataset.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

from .normalize import (
    normalize_player_name, normalize_score, create_tournament_key,
    create_match_key, normalize_round
)

logger = logging.getLogger(__name__)


def prepare_ta_dataframe(ta_matches: list) -> pd.DataFrame:
    """
    Convert TA match list to DataFrame with normalized keys.
    
    Args:
        ta_matches: List of TA match dictionaries
        
    Returns:
        DataFrame with normalized columns
    """
    if not ta_matches:
        return pd.DataFrame()
    
    df = pd.DataFrame(ta_matches)
    
    # Normalize player names
    if 'winner_name' in df.columns:
        df['winner_name_norm'] = df['winner_name'].apply(normalize_player_name)
    if 'loser_name' in df.columns:
        df['loser_name_norm'] = df['loser_name'].apply(normalize_player_name)
    
    # Normalize scores
    if 'score' in df.columns:
        df['score_norm'] = df['score'].apply(normalize_score)
    
    # Normalize rounds
    if 'round' in df.columns:
        df['round_norm'] = df['round'].apply(normalize_round)
    
    # Create tournament keys
    df['tournament_key'] = df.apply(
        lambda row: create_tournament_key(
            row.get('tourney_name', ''),
            int(row.get('start_date', '2025')[:4]) if pd.notna(row.get('start_date')) else 2025,
            row.get('tourney_level', ''),
            row.get('surface', '')
        ),
        axis=1
    )
    
    # Create match keys
    df['match_key'] = df.apply(
        lambda row: create_match_key(
            int(row.get('start_date', '2025')[:4]) if pd.notna(row.get('start_date')) else 2025,
            row['tournament_key'],
            row.get('round_norm', ''),
            row.get('winner_name_norm', ''),
            row.get('loser_name_norm', ''),
            row.get('score_norm', ''),
            row.get('start_date') if pd.notna(row.get('start_date')) else None
        ),
        axis=1
    )
    
    # Flatten winner_stats and loser_stats
    if 'winner_stats' in df.columns:
        for stat in ['ace_pct', 'first_in_pct', 'first_won_pct', 'second_won_pct', 'bpsvd']:
            df[f'ta_{stat}_w'] = df['winner_stats'].apply(
                lambda x: x.get(stat) if isinstance(x, dict) else None
            )
    
    if 'loser_stats' in df.columns:
        for stat in ['ace_pct', 'first_in_pct', 'first_won_pct', 'second_won_pct', 'bpsvd']:
            df[f'ta_{stat}_l'] = df['loser_stats'].apply(
                lambda x: x.get(stat) if isinstance(x, dict) else None
            )
    
    if 'dr' in df.columns:
        df['ta_dr_w'] = df['dr']
    
    return df


def prepare_base_dataframe(base_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Prepare base dataframe with normalized keys for matching.
    
    Args:
        base_df: Base dataset DataFrame
        year: Year to filter
        
    Returns:
        DataFrame with normalized columns
    """
    df = base_df.copy()
    
    # Filter to year if tourney_date exists
    if 'tourney_date' in df.columns:
        df['year'] = df['tourney_date'].astype(str).str[:4]
        df = df[df['year'] == str(year)].copy()
    
    # Normalize player names
    if 'winner_name' in df.columns:
        df['winner_name_norm'] = df['winner_name'].apply(normalize_player_name)
    if 'loser_name' in df.columns:
        df['loser_name_norm'] = df['loser_name'].apply(normalize_player_name)
    
    # Normalize scores
    if 'score' in df.columns:
        df['score_norm'] = df['score'].apply(normalize_score)
    
    # Normalize rounds
    if 'round' in df.columns:
        df['round_norm'] = df['round'].apply(normalize_round)
    
    # Create tournament keys
    tournament_keys = []
    for _, row in df.iterrows():
        key = create_tournament_key(
            str(row.get('tourney_name', '')),
            year,
            str(row.get('tourney_level', '')),
            str(row.get('surface', ''))
        )
        tournament_keys.append(key)
    df['tournament_key'] = tournament_keys
    
    # Create match keys
    match_keys = []
    for _, row in df.iterrows():
        key = create_match_key(
            year,
            str(row['tournament_key']),
            str(row.get('round_norm', '')),
            str(row.get('winner_name_norm', '')),
            str(row.get('loser_name_norm', '')),
            str(row.get('score_norm', '')),
            str(row.get('tourney_date')) if 'tourney_date' in df.columns and pd.notna(row.get('tourney_date')) else None
        )
        match_keys.append(key)
    df['match_key'] = match_keys
    
    return df


def match_datasets(base_df: pd.DataFrame, ta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match TA data to base dataset.
    
    Args:
        base_df: Base dataset with normalized keys
        ta_df: TA dataset with normalized keys
        
    Returns:
        Base dataset with TA columns joined
    """
    # Start with base dataframe
    result_df = base_df.copy()
    
    # Initialize TA columns
    ta_columns = [
        'ta_ace_pct_w', 'ta_first_in_pct_w', 'ta_first_won_pct_w', 'ta_second_won_pct_w',
        'ta_bpsvd_w', 'ta_dr_w',
        'ta_ace_pct_l', 'ta_first_in_pct_l', 'ta_first_won_pct_l', 'ta_second_won_pct_l',
        'ta_bpsvd_l',
        'ta_match_found', 'ta_join_quality'
    ]
    
    for col in ta_columns:
        result_df[col] = None
    
    result_df['ta_match_found'] = False
    result_df['ta_join_quality'] = ''
    
    if ta_df.empty:
        logger.warning("TA dataframe is empty, no matches possible")
        return result_df
    
    # Strategy 1: Exact match on match_key
    exact_matches = pd.merge(
        result_df,
        ta_df,
        on='match_key',
        how='left',
        suffixes=('', '_ta')
    )
    
    matched_mask = exact_matches['match_key'].isin(ta_df['match_key'])
    result_df.loc[matched_mask, 'ta_match_found'] = True
    result_df.loc[matched_mask, 'ta_join_quality'] = 'exact'
    
    # Copy TA columns from exact matches
    for col in ta_columns:
        if col in exact_matches.columns:
            result_df.loc[matched_mask, col] = exact_matches.loc[matched_mask, col]
    
    # Strategy 2: Relaxed score match (if exact didn't match)
    unmatched = result_df[~result_df['ta_match_found']].copy()
    
    if len(unmatched) > 0:
        # Try matching on tournament + round + players (ignore score)
        for idx, row in unmatched.iterrows():
            ta_match = ta_df[
                (ta_df['tournament_key'] == row['tournament_key']) &
                (ta_df['round_norm'] == row['round_norm']) &
                (ta_df['winner_name_norm'] == row['winner_name_norm']) &
                (ta_df['loser_name_norm'] == row['loser_name_norm'])
            ]
            
            if len(ta_match) == 1:
                ta_row = ta_match.iloc[0]
                result_df.loc[idx, 'ta_match_found'] = True
                result_df.loc[idx, 'ta_join_quality'] = 'relaxed_score'
                
                # Copy TA stats
                for col in ta_columns:
                    if col in ta_row.index:
                        result_df.loc[idx, col] = ta_row[col]
    
    logger.info(f"Matched {result_df['ta_match_found'].sum()} out of {len(result_df)} base matches")
    
    return result_df

