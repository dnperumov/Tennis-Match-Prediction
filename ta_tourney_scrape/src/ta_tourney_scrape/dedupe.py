"""
Deduplicate and merge match records from multiple sources (tournament page + player profiles).

PROBLEM:
- Same match appears on BOTH winner and loser player profiles
- Need to merge them into a single canonical record with both sides' stats

STRATEGY:
- Generate strong match keys (tournament, date, round, players, score)
- Group matches by key
- Prefer winner profile for winner-side stats, loser profile for loser-side stats
- Deduplicate keeping one record per match
"""

import logging
import pandas as pd
from typing import List, Dict, Any
from .normalize import normalize_player_name, normalize_score, normalize_round, create_match_key

logger = logging.getLogger(__name__)


def deduplicate_profile_matches(profile_matches: List[Dict[str, Any]], year: int) -> List[Dict[str, Any]]:
    """
    Deduplicate match records scraped from player profiles.
    
    The same match will appear on both winner and loser profiles.
    We want to merge them:
    - Winner profile provides winner-side serve stats (A%, DF%, 1stIn, 1st%, 2nd%, BPSvd, etc.)
    - Loser profile provides loser-side serve stats
    
    Args:
        profile_matches: List of match records from player profiles
        year: Year
        
    Returns:
        Deduplicated list with merged stats
    """
    if not profile_matches:
        return []
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(profile_matches)
    
    # Normalize for matching
    if 'winner_name' in df.columns:
        df['winner_name_norm'] = df['winner_name'].apply(normalize_player_name)
    if 'loser_name' in df.columns:
        df['loser_name_norm'] = df['loser_name'].apply(normalize_player_name)
    if 'score' in df.columns:
        df['score_norm'] = df['score'].apply(normalize_score)
    if 'round' in df.columns:
        df['round_norm'] = df['round'].apply(normalize_round)
    
    # Create match keys
    match_keys = []
    for _, row in df.iterrows():
        key = create_match_key(
            year,
            row.get('tourney_name', ''),
            row.get('round_norm', ''),
            row.get('winner_name_norm', ''),
            row.get('loser_name_norm', ''),
            row.get('score_norm', ''),
            row.get('match_date')
        )
        match_keys.append(key)
    
    df['match_key'] = match_keys
    
    # Group by match_key
    grouped = df.groupby('match_key')
    
    deduped = []
    for key, group in grouped:
        if len(group) == 1:
            # Only one record, use as-is
            deduped.append(group.iloc[0].to_dict())
        else:
            # Multiple records (likely winner + loser profiles)
            # Merge them
            merged = merge_profile_group(group)
            deduped.append(merged)
    
    logger.info(f"Deduplicated {len(profile_matches)} profile matches to {len(deduped)} unique matches")
    return deduped


def merge_profile_group(group: pd.DataFrame) -> Dict[str, Any]:
    """
    Merge multiple profile records for the same match.
    
    Strategy:
    - Use winner profile for winner_stats
    - Use loser profile for loser_stats
    - Prefer non-null values for other fields
    
    Args:
        group: DataFrame group of records for same match
        
    Returns:
        Merged match record
    """
    merged = {}
    
    # Base fields (take first non-null)
    base_fields = ['match_date', 'tourney_name', 'surface', 'round', 'winner_name', 'loser_name', 
                   'winner_rank', 'loser_rank', 'score']
    
    for field in base_fields:
        if field in group.columns:
            non_null = group[field].dropna()
            if not non_null.empty:
                merged[field] = non_null.iloc[0]
    
    # Find winner and loser profile rows (from perspective of source_player)
    winner_row = None
    loser_row = None

    for _, row in group.iterrows():
        if row.get('player_won') is True:
            winner_row = row
        elif row.get('player_won') is False:
            loser_row = row

    # Merge classic-profile per-player stats into winner/loser side columns.
    # These stats are from the perspective of the profile owner (source_player).
    # If profile owner won, their stats map to winner-side; otherwise loser-side.
    stat_fields = [
        'ta_dr',
        'ta_ace_pct',
        'ta_df_pct',
        'ta_1stin_pct',
        'ta_1stwon_pct',
        'ta_2ndwon_pct',
        'ta_bpsvd_num',
        'ta_bpsvd_den',
        'ta_bpsvd_pct',
    ]

    def _copy_side(prefix: str, src_row: pd.Series) -> None:
        for f in stat_fields:
            if f in src_row.index and pd.notna(src_row.get(f)):
                merged[f"{prefix}{f}"] = src_row.get(f)
        # minutes: match-level; take first non-null across group

    if winner_row is not None:
        _copy_side("w_", winner_row)
    if loser_row is not None:
        _copy_side("l_", loser_row)

    # Also expose Sackmann/TA-style dict fields used by the main dataset
    def _stats_dict_from_row(src_row: pd.Series) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if pd.notna(src_row.get("ta_ace_pct")):
            d["ace_pct"] = float(src_row.get("ta_ace_pct"))
        if pd.notna(src_row.get("ta_df_pct")):
            d["df_pct"] = float(src_row.get("ta_df_pct"))
        if pd.notna(src_row.get("ta_1stin_pct")):
            d["first_in_pct"] = float(src_row.get("ta_1stin_pct"))
        if pd.notna(src_row.get("ta_1stwon_pct")):
            d["first_won_pct"] = float(src_row.get("ta_1stwon_pct"))
        if pd.notna(src_row.get("ta_2ndwon_pct")):
            d["second_won_pct"] = float(src_row.get("ta_2ndwon_pct"))
        if pd.notna(src_row.get("ta_bpsvd_pct")):
            d["bpsvd_pct"] = float(src_row.get("ta_bpsvd_pct"))
        if pd.notna(src_row.get("ta_bpsvd_num")):
            d["bpsvd_num"] = int(src_row.get("ta_bpsvd_num"))
        if pd.notna(src_row.get("ta_bpsvd_den")):
            d["bpsvd_den"] = int(src_row.get("ta_bpsvd_den"))
        return d

    if winner_row is not None:
        merged["winner_stats"] = _stats_dict_from_row(winner_row)
    if loser_row is not None:
        merged["loser_stats"] = _stats_dict_from_row(loser_row)

    # Winner/loser ranks from classic table (Rk/vRk) if available
    # If profile owner won, their rank is winner_rank and opponent_rank is loser_rank; vice-versa.
    if winner_row is not None:
        wr = winner_row.get("player_rank")
        lr = winner_row.get("opponent_rank")
        if pd.notna(wr):
            merged["winner_rank"] = int(wr)
        if pd.notna(lr):
            merged["loser_rank"] = int(lr)
    if loser_row is not None:
        # fill in any missing ranks
        wr = loser_row.get("opponent_rank")
        lr = loser_row.get("player_rank")
        if "winner_rank" not in merged and pd.notna(wr):
            merged["winner_rank"] = int(wr)
        if "loser_rank" not in merged and pd.notna(lr):
            merged["loser_rank"] = int(lr)

    # dr is match-level; use winner-side dr if available else any
    if "dr" not in merged:
        if winner_row is not None and pd.notna(winner_row.get("ta_dr")):
            merged["dr"] = float(winner_row.get("ta_dr"))
        elif loser_row is not None and pd.notna(loser_row.get("ta_dr")):
            merged["dr"] = float(loser_row.get("ta_dr"))

    if 'minutes' in group.columns:
        non_null_min = group['minutes'].dropna()
        if not non_null_min.empty:
            merged['time'] = int(non_null_min.iloc[0])
    
    # Source tracking
    merged['source'] = 'player_profile_classic'
    merged['ta_stats_source'] = 'player_profile_classic'
    
    source_players = group['source_player'].dropna().unique().tolist() if 'source_player' in group.columns else []
    merged['source_profile_players'] = source_players
    
    return merged


def merge_tournament_and_profile_data(tournament_matches: List[Dict[str, Any]], 
                                     profile_matches: List[Dict[str, Any]],
                                     year: int) -> List[Dict[str, Any]]:
    """
    Merge tournament page matches with profile-enriched stats.
    
    STRATEGY:
    - Tournament matches are authoritative for match existence
    - Profile matches provide stats when tournament page lacks them
    - Match on: (year, tournament_norm, round, winner, loser, score)
    
    Args:
        tournament_matches: Matches from tournament pages
        profile_matches: Deduped matches from player profiles
        year: Year
        
    Returns:
        Merged match list
    """
    if not tournament_matches:
        return profile_matches
    
    if not profile_matches:
        return tournament_matches
    
    # Convert to DataFrames
    tourney_df = pd.DataFrame(tournament_matches)
    profile_df = pd.DataFrame(profile_matches)
    
    # Normalize both sides
    for df in [tourney_df, profile_df]:
        if 'winner_name' in df.columns:
            df['winner_name_norm'] = df['winner_name'].apply(normalize_player_name)
        if 'loser_name' in df.columns:
            df['loser_name_norm'] = df['loser_name'].apply(normalize_player_name)
        if 'score' in df.columns:
            df['score_norm'] = df['score'].apply(normalize_score)
        if 'round' in df.columns:
            df['round_norm'] = df['round'].apply(normalize_round)
    
    # Create match keys
    for df in [tourney_df, profile_df]:
        keys = []
        for _, row in df.iterrows():
            key = create_match_key(
                year,
                row.get('tourney_name', ''),
                row.get('round_norm', ''),
                row.get('winner_name_norm', ''),
                row.get('loser_name_norm', ''),
                row.get('score_norm', ''),
                row.get('match_date')
            )
            keys.append(key)
        df['match_key'] = keys
    
    # Left join: tournament matches are base
    merged_df = tourney_df.merge(
        profile_df.add_suffix('_profile'),
        left_on='match_key',
        right_on='match_key_profile',
        how='left'
    )
    
    # Enrich tournament matches with profile stats where missing
    enriched = []
    for _, row in merged_df.iterrows():
        match = row.to_dict()
        
        # If tournament match has no stats, use profile stats
        if not match.get('has_stats') and pd.notna(match.get('match_key_profile')):
            # Copy profile stats
            match['winner_stats'] = match.get('winner_stats_profile', {})
            match['loser_stats'] = match.get('loser_stats_profile', {})
            match['match_date'] = match.get('match_date_profile', match.get('match_date'))
            match['ta_stats_source'] = 'player_profile'
            match['has_stats'] = True
        
        # Clean up merged columns
        clean_match = {k: v for k, v in match.items() if not k.endswith('_profile') and pd.notna(v)}
        enriched.append(clean_match)
    
    logger.info(f"Merged {len(tournament_matches)} tournament matches with {len(profile_matches)} profile matches")
    return enriched

