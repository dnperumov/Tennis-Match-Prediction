"""
Parse player profile pages to extract match statistics.

This module handles fallback enrichment when tournament pages lack per-match stats.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from .http import get_session, BASE_URL
from .normalize import normalize_player_name, normalize_score

logger = logging.getLogger(__name__)


def construct_player_profile_url(player_name: str, use_classic: bool = True) -> str:
    """
    Construct TennisAbstract player profile URL from player name.
    
    Format: https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p=PlayerName
    
    Args:
        player_name: Player name (e.g., "Novak Djokovic")
        use_classic: If True, use player-classic.cgi (has more match data). Default True.
        
    Returns:
        Player profile URL
    """
    # Convert name to TA format: FirstnameLastname (no spaces, capitalize first letters)
    name_parts = player_name.strip().split()
    if len(name_parts) >= 2:
        ta_name = ''.join([part.capitalize() for part in name_parts])
    else:
        ta_name = player_name.replace(' ', '').capitalize()
    
    page = "player-classic.cgi" if use_classic else "player.cgi"
    return f"{BASE_URL}/cgi-bin/{page}?p={ta_name}"


def fetch_player_profile(player_name: str) -> Optional[BeautifulSoup]:
    """
    Fetch player profile page HTML.
    
    Args:
        player_name: Player name
        
    Returns:
        BeautifulSoup object or None if fetch failed
    """
    url = construct_player_profile_url(player_name)
    logger.debug(f"Fetching profile for {player_name}: {url}")
    
    session = get_session()
    html = session.get_html(url)
    
    if not html:
        logger.warning(f"Failed to fetch profile for {player_name}")
        return None
    
    return html


def parse_match_table_from_profile(html: BeautifulSoup, player_name: str, year: int, 
                                   tournament_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Parse match statistics table from player profile page.
    
    The table typically has columns:
    Date | Tournament | Surface | Rd | Rk | vRk | Opponent | Score | DR | A% | DF% | 1stIn | 1st% | 2nd% | BPSvd | Time
    
    Args:
        html: BeautifulSoup object of profile page
        player_name: Player name (for context)
        year: Year to filter matches
        tournament_filter: Optional tournament name to filter by
        
    Returns:
        List of match records with stats
    """
    matches = []
    
    # Find the matches table
    # Look for table with headers containing "Date", "Tournament", etc.
    tables = html.find_all('table')
    
    match_table = None
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        if 'Date' in headers and 'Tournament' in headers and 'Score' in headers:
            match_table = table
            break
    
    if not match_table:
        logger.warning(f"No match table found on profile for {player_name}")
        return matches
    
    # Parse headers to get column indices
    headers = [th.get_text(strip=True) for th in match_table.find_all('th')]
    header_map = {h: i for i, h in enumerate(headers)}
    
    # Required columns
    required = ['Date', 'Tournament', 'Score']
    if not all(col in header_map for col in required):
        logger.warning(f"Match table missing required columns for {player_name}")
        return matches
    
    # Parse rows
    rows = match_table.find_all('tr')[1:]  # Skip header row
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) < len(headers):
            continue
        
        try:
            # Extract date
            date_str = cells[header_map['Date']].get_text(strip=True)
            try:
                match_date = date_parser.parse(date_str)
                if match_date.year != year:
                    continue  # Skip matches not in target year
            except:
                logger.debug(f"Could not parse date: {date_str}")
                continue
            
            # Extract tournament name
            tourney_cell = cells[header_map['Tournament']]
            tourney_name = tourney_cell.get_text(strip=True)
            
            # Apply tournament filter if provided
            if tournament_filter:
                tourney_norm = tourney_name.lower().replace('-', '').replace(' ', '')
                filter_norm = tournament_filter.lower().replace('-', '').replace(' ', '')
                if filter_norm not in tourney_norm:
                    continue
            
            # Extract opponent and determine winner/loser
            # Profile rows show: "d. OpponentName" (win) or "OpponentName" (loss)
            score_cell = cells[header_map['Score']]
            score_text = score_cell.get_text(strip=True)
            
            # Determine if player won or lost
            player_won = 'd.' in score_text or cells[header_map.get('Rd', 0)].get_text(strip=True).startswith('W')
            
            # Extract opponent name
            opponent_name = ''
            if 'vRk' in header_map:
                opp_cell_idx = header_map['vRk'] + 1  # Opponent usually after vRk column
                if opp_cell_idx < len(cells):
                    opponent_name = cells[opp_cell_idx].get_text(strip=True)
            
            # If opponent not found, try parsing from score text
            if not opponent_name:
                # Score format might be: "d. Opponent 6-4 6-3" or "Opponent 4-6 3-6"
                opp_match = re.search(r'(?:d\.\s+)?([A-Z][a-zA-Z\s\-\.]+?)\s+\d+-\d+', score_text)
                if opp_match:
                    opponent_name = opp_match.group(1).strip()
            
            # Clean score (remove "d." and opponent name)
            score_clean = re.sub(r'd\.\s+', '', score_text)
            score_clean = re.sub(r'[A-Z][a-zA-Z\s\-\.]+?\s+', '', score_clean, count=1)
            score_clean = score_clean.strip()
            
            # Build match record
            match_record = {
                'source': 'player_profile',
                'source_player': player_name,
                'match_date': match_date.strftime('%Y-%m-%d'),
                'tourney_name': tourney_name,
                'round': cells[header_map.get('Rd', 0)].get_text(strip=True) if 'Rd' in header_map else '',
                'player_rank': cells[header_map.get('Rk', 0)].get_text(strip=True) if 'Rk' in header_map else '',
                'opponent_rank': cells[header_map.get('vRk', 0)].get_text(strip=True) if 'vRk' in header_map else '',
                'opponent_name': opponent_name,
                'player_won': player_won,
                'score': score_clean,
            }
            
            # Assign winner/loser
            if player_won:
                match_record['winner_name'] = player_name
                match_record['loser_name'] = opponent_name
                match_record['winner_rank'] = match_record['player_rank']
                match_record['loser_rank'] = match_record['opponent_rank']
            else:
                match_record['winner_name'] = opponent_name
                match_record['loser_name'] = player_name
                match_record['winner_rank'] = match_record['opponent_rank']
                match_record['loser_rank'] = match_record['player_rank']
            
            # Extract surface
            if 'Surface' in header_map:
                surface_text = cells[header_map['Surface']].get_text(strip=True)
                match_record['surface'] = surface_text
            
            # Extract percentage stats (these are what we're after!)
            stat_columns = {
                'DR': 'dr',
                'A%': 'ace_pct',
                'DF%': 'df_pct',
                '1stIn': 'first_in_pct',
                '1st%': 'first_won_pct',
                '2nd%': 'second_won_pct',
                'BPSvd': 'bpsvd',
                'SPW': 'spw',
                'RPW': 'rpw',
                'TPW': 'tpw',
                'Time': 'time'
            }
            
            player_stats = {}
            for ta_col, our_col in stat_columns.items():
                if ta_col in header_map:
                    value_str = cells[header_map[ta_col]].get_text(strip=True)
                    
                    # Parse percentage (remove % sign, convert to float 0-1)
                    if '%' in value_str:
                        try:
                            player_stats[our_col] = float(value_str.replace('%', '')) / 100.0
                        except:
                            player_stats[our_col] = None
                    elif ta_col == 'DR':
                        try:
                            player_stats[our_col] = float(value_str)
                        except:
                            player_stats[our_col] = None
                    elif ta_col == 'BPSvd':
                        # Format: "3/5" or similar
                        if '/' in value_str:
                            parts = value_str.split('/')
                            try:
                                player_stats[f'{our_col}_num'] = int(parts[0])
                                player_stats[f'{our_col}_den'] = int(parts[1])
                                player_stats[f'{our_col}_pct'] = int(parts[0]) / int(parts[1]) if int(parts[1]) > 0 else None
                            except:
                                player_stats[our_col] = None
                    elif ta_col == 'Time':
                        player_stats[our_col] = value_str
                    else:
                        player_stats[our_col] = value_str
            
            # Store stats under correct side (winner or loser)
            if player_won:
                match_record['winner_stats'] = player_stats
                match_record['loser_stats'] = {}
            else:
                match_record['winner_stats'] = {}
                match_record['loser_stats'] = player_stats
            
            matches.append(match_record)
            
        except Exception as e:
            logger.debug(f"Error parsing match row for {player_name}: {e}")
            continue
    
    logger.info(f"Extracted {len(matches)} matches from profile for {player_name} (year={year})")
    return matches


def scrape_player_profile_matches(player_name: str, year: int, 
                                  tournament_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Scrape matches from a player's profile page.
    
    Args:
        player_name: Player name
        year: Year to filter matches
        tournament_filter: Optional tournament name to filter
        
    Returns:
        List of match records with stats
    """
    html = fetch_player_profile(player_name)
    if not html:
        return []
    
    matches = parse_match_table_from_profile(html, player_name, year, tournament_filter)
    return matches


def enrich_tournament_with_profile_stats(tournament_matches: List[Dict[str, Any]], 
                                        year: int) -> List[Dict[str, Any]]:
    """
    Enrich tournament matches that lack stats by scraping player profiles.
    
    APPROACH:
    - For each match missing stats, scrape both winner and loser profiles
    - Extract the specific match stats from their profile match logs
    - Merge winner-side and loser-side stats
    - Deduplicate
    
    Args:
        tournament_matches: List of match records from tournament page
        year: Year
        
    Returns:
        List of enriched match records
    """
    enriched = []
    
    # Identify matches missing stats
    for match in tournament_matches:
        if match.get('has_stats'):
            # Already has stats from tournament page
            enriched.append(match)
            continue
        
        # Need to fetch from profiles
        winner_name = match.get('winner_name')
        loser_name = match.get('loser_name')
        tourney_name = match.get('tourney_name')
        
        if not winner_name or not loser_name:
            logger.warning(f"Match missing player names, cannot enrich: {match}")
            enriched.append(match)
            continue
        
        # Scrape winner profile
        logger.info(f"Fetching stats from profile: {winner_name} vs {loser_name}")
        winner_matches = scrape_player_profile_matches(winner_name, year, tourney_name)
        
        # Find this specific match in winner's profile
        match_found = False
        for w_match in winner_matches:
            # Match by: opponent name, score, round
            if (normalize_player_name(w_match.get('loser_name', '')) == normalize_player_name(loser_name) and
                normalize_score(w_match.get('score', '')) == normalize_score(match.get('score', ''))):
                # Found it! Merge stats
                match['winner_stats'] = w_match.get('winner_stats', {})
                match['match_date'] = w_match.get('match_date', match.get('match_date', ''))
                match['surface'] = w_match.get('surface', match.get('surface', ''))
                match['ta_stats_source'] = 'player_profile'
                match['source_profile_player'] = winner_name
                match_found = True
                break
        
        if not match_found:
            logger.warning(f"Could not find match in winner profile: {winner_name} vs {loser_name}")
            match['ta_stats_source'] = 'missing'
        
        # Optionally also scrape loser profile for loser-side stats
        # (Skipping for now to reduce requests; winner-side stats are usually sufficient)
        
        enriched.append(match)
    
    return enriched

