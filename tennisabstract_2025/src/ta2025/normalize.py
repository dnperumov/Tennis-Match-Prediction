"""
Normalize raw match records to Sackmann-compatible format.
"""

import logging
from typing import Dict, List, Any, Optional
from .id_map import PlayerIDMapper
from .utils import parse_date, normalize_name

logger = logging.getLogger(__name__)

# Column list will be provided via function parameter
# This ensures we don't hardcode column names


def normalize_match_record(raw_record: Dict[str, Any], 
                          player_mapper: PlayerIDMapper,
                          column_list: List[str]) -> Dict[str, Any]:
    """
    Normalize a raw match record to match Sackmann schema.
    
    Args:
        raw_record: Raw match record from scraping
        player_mapper: PlayerIDMapper instance
        column_list: List of column names in order
        
    Returns:
        Normalized record with all columns
    """
    normalized = {col: "" for col in column_list}
    
    # Map known fields from scraping
    winner_name = raw_record.get('winner_name', '').strip()
    loser_name = raw_record.get('loser_name', '').strip()
    
    # Player names
    if 'winner_name' in column_list:
        normalized['winner_name'] = winner_name
    if 'loser_name' in column_list:
        normalized['loser_name'] = loser_name
    
    # Map player IDs
    winner_id = player_mapper.find_player_id(winner_name)
    loser_id = player_mapper.find_player_id(loser_name)
    
    if 'winner_id' in column_list:
        normalized['winner_id'] = winner_id if winner_id else ""
    if 'loser_id' in column_list:
        normalized['loser_id'] = loser_id if loser_id else ""
    
    # Score
    if 'score' in column_list:
        normalized['score'] = raw_record.get('score', '').strip()
    
    # Round
    if 'round' in column_list:
        normalized['round'] = raw_record.get('round', '').strip()
    
    # Tournament metadata
    if 'tourney_name' in column_list:
        normalized['tourney_name'] = raw_record.get('tourney_name', '').strip()
    if 'surface' in column_list:
        normalized['surface'] = raw_record.get('surface', '').strip()
    if 'tourney_date' in column_list:
        date_str = raw_record.get('tourney_date', '')
        if date_str:
            parsed_date = parse_date(date_str)
            normalized['tourney_date'] = parsed_date if parsed_date else ""
        else:
            normalized['tourney_date'] = ""
    
    # Tournament ID - generate from name/date if available
    if 'tourney_id' in column_list:
        tourney_id = raw_record.get('tourney_id', '')
        if not tourney_id and normalized.get('tourney_name'):
            # Generate a simple ID from tournament name (can be improved)
            tourney_id = normalized['tourney_name'].upper().replace(' ', '')[:10]
        normalized['tourney_id'] = tourney_id if tourney_id else ""
    
    # Match number - from raw record if available
    if 'match_num' in column_list:
        normalized['match_num'] = raw_record.get('match_num', '').strip()
    
    # All other columns remain empty ("") as per requirements
    # We do NOT infer or compute: seeds, ranks, points, stats, etc.
    
    return normalized


def normalize_all_records(raw_records: List[Dict[str, Any]],
                         player_mapper: PlayerIDMapper,
                         column_list: List[str]) -> List[Dict[str, Any]]:
    """
    Normalize all raw match records.
    
    Args:
        raw_records: List of raw match records
        player_mapper: PlayerIDMapper instance
        column_list: List of column names in order
        
    Returns:
        List of normalized records
    """
    logger.info(f"Normalizing {len(raw_records)} match records...")
    
    normalized_records = []
    for raw_record in raw_records:
        try:
            normalized = normalize_match_record(raw_record, player_mapper, column_list)
            normalized_records.append(normalized)
        except Exception as e:
            logger.error(f"Error normalizing record: {e}", exc_info=True)
    
    logger.info(f"Normalized {len(normalized_records)} records")
    return normalized_records

