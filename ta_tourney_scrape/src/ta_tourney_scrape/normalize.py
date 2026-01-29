"""
Normalize player names, scores, and tournament identifiers for matching.
"""

import re
from typing import Dict, Any
from unidecode import unidecode


def normalize_player_name(name: str) -> str:
    """
    Normalize player name for matching.
    
    Args:
        name: Raw player name
        
    Returns:
        Normalized name
    """
    if not name:
        return ""
    
    # Remove accents and convert to ASCII
    normalized = unidecode(name)
    
    # Remove common suffixes in brackets
    normalized = re.sub(r'\s*\[.*?\]\s*', '', normalized)
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Uppercase for consistency
    normalized = normalized.upper()
    
    return normalized


def normalize_score(score: str) -> str:
    """
    Normalize score string for matching.
    
    Args:
        score: Raw score string
        
    Returns:
        Normalized score
    """
    if not score:
        return ""
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', score).strip()
    
    # Standardize tiebreak format: 7-6(4) -> 7-6(4)
    normalized = re.sub(r'7-6\s*\(\s*(\d+)\s*\)', r'7-6(\1)', normalized)
    
    # Remove trailing/leading spaces around hyphens
    normalized = re.sub(r'\s*-\s*', '-', normalized)
    
    # Preserve RET, W/O, W/O
    normalized = re.sub(r'(RET|W/O|W-O)', r'\1', normalized, flags=re.IGNORECASE)
    
    return normalized


def create_tournament_key(tourney_name: str, year: int, level: str = '', surface: str = '') -> str:
    """
    Create a normalized tournament key for matching.
    
    Args:
        tourney_name: Tournament name
        year: Year
        level: Tournament level (A, C, etc.)
        surface: Surface (Hard, Clay, Grass)
        
    Returns:
        Normalized tournament key
    """
    # Normalize name
    name_slug = unidecode(tourney_name).upper()
    name_slug = re.sub(r'[^A-Z0-9]', '', name_slug)
    name_slug = name_slug[:20]  # Limit length
    
    # Build key
    parts = [str(year), name_slug]
    if level:
        parts.append(level.upper())
    if surface:
        parts.append(surface.upper()[:3])
    
    return '_'.join(parts)


def normalize_round(round_str: str) -> str:
    """
    Normalize round label for matching.
    
    Args:
        round_str: Raw round string (e.g., "R32", "QF", "F")
        
    Returns:
        Normalized round
    """
    if not round_str:
        return ""
    
    normalized = round_str.strip().upper()
    
    # Standardize common formats
    normalized = re.sub(r'^R(\d+)$', r'R\1', normalized)  # R32, R64, etc.
    normalized = re.sub(r'^Q(\d+)$', r'Q\1', normalized)  # Q1, Q2, etc.
    normalized = re.sub(r'^RR$', 'RR', normalized)  # Round robin
    
    return normalized


def create_match_key(
    year: int,
    tournament_key: str,
    round: str,
    winner_name: str,
    loser_name: str,
    score: str,
    match_date: str = None
) -> str:
    """
    Create a match key for exact matching.
    
    Args:
        year: Year
        tournament_key: Normalized tournament key
        round: Round (R32, QF, etc.)
        winner_name: Winner name (normalized)
        loser_name: Loser name (normalized)
        score: Score (normalized)
        match_date: Optional match date (ISO format)
        
    Returns:
        Match key string
    """
    winner_norm = normalize_player_name(winner_name)
    loser_norm = normalize_player_name(loser_name)
    score_norm = normalize_score(score)
    round_norm = round.upper().strip()
    
    parts = [
        str(year),
        tournament_key,
        round_norm,
        winner_norm,
        loser_norm,
        score_norm
    ]
    
    if match_date:
        parts.insert(1, match_date)
    
    return '|'.join(parts)


def normalize_round(round_str: str) -> str:
    """
    Normalize round label.
    
    Args:
        round_str: Raw round string
        
    Returns:
        Normalized round (R128, R64, R32, R16, QF, SF, F, RR, Q1, Q2, Q3)
    """
    if not round_str:
        return ""
    
    round_str = round_str.upper().strip()
    
    # Map common variations
    round_map = {
        'R128': 'R128',
        'R64': 'R64',
        'R32': 'R32',
        'R16': 'R16',
        'ROUND OF 16': 'R16',
        'QUARTERFINAL': 'QF',
        'QUARTERFINALIST': 'QF',
        'SEMIFINAL': 'SF',
        'SEMIFINALIST': 'SF',
        'FINAL': 'F',
        'ROUND ROBIN': 'RR',
        'Q1': 'Q1',
        'Q2': 'Q2',
        'Q3': 'Q3',
        'QUALIFYING 1': 'Q1',
        'QUALIFYING 2': 'Q2',
        'QUALIFYING 3': 'Q3',
    }
    
    if round_str in round_map:
        return round_map[round_str]
    
    # Try to extract round number
    match = re.match(r'R(\d+)', round_str)
    if match:
        return f"R{match.group(1)}"
    
    return round_str

