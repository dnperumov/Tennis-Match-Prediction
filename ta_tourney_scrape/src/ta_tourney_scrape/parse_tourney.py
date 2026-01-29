"""
Parse tournament pages to extract match data and percentage stats.

CRITICAL: Tournament pages vary in format:
- Some have detailed stat tables with columns (A%, DF%, 1stIn, etc.)
- Others only list completed matches (round, players, score) with NO per-match stats
- This module DETECTS which format and sets a flag 'has_stats'
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def parse_tournament_metadata(html: BeautifulSoup, url: str) -> Dict[str, Any]:
    """
    Extract tournament metadata from page.
    
    Args:
        html: BeautifulSoup object
        url: Tournament page URL
        
    Returns:
        Dictionary with tournament metadata
    """
    metadata = {
        'tourney_name': '',
        'surface': '',
        'draw_size': '',
        'tourney_level': '',
        'start_date': '',
        'ta_url': url
    }
    
    page_text = html.get_text()
    
    # Extract tournament name from title or headers
    title = html.find('title')
    if title:
        title_text = title.get_text()
        # Remove "Tennis Abstract:" prefix
        if ':' in title_text:
            metadata['tourney_name'] = title_text.split(':', 1)[1].strip()
        else:
            metadata['tourney_name'] = title_text.strip()
    
    # Extract surface
    if 'Surface:' in page_text:
        surface_match = re.search(r'Surface:\s*(\w+)', page_text, re.IGNORECASE)
        if surface_match:
            metadata['surface'] = surface_match.group(1).strip()
    
    # Extract draw size
    if 'Draw:' in page_text:
        draw_match = re.search(r'Draw:\s*(\d+)', page_text, re.IGNORECASE)
        if draw_match:
            metadata['draw_size'] = draw_match.group(1).strip()
    
    # Extract date
    date_match = re.search(r'(\w+\s+\d{1,2},\s+\d{4})', page_text)
    if date_match:
        try:
            date_str = date_match.group(1)
            dt = datetime.strptime(date_str, '%B %d, %Y')
            metadata['start_date'] = dt.strftime('%Y-%m-%d')
        except:
            pass
    
    # Extract level from URL or page
    if 'ATP' in url.upper() or 'ATP' in page_text.upper():
        metadata['tourney_level'] = 'A'
    elif 'CH' in url.upper() or 'Challenger' in page_text:
        metadata['tourney_level'] = 'C'
    
    return metadata


def parse_percentage(value: str) -> Optional[float]:
    """
    Parse percentage string to float in [0, 1].
    
    Args:
        value: String like "9.4%" or "79.7%"
        
    Returns:
        Float in [0, 1] or None
    """
    if not value or value.strip() == '':
        return None
    
    # Remove % and whitespace
    cleaned = value.replace('%', '').strip()
    
    try:
        pct = float(cleaned) / 100.0
        return max(0.0, min(1.0, pct))  # Clamp to [0, 1]
    except (ValueError, TypeError):
        return None


def parse_ratio(value: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """
    Parse ratio string like "2/2" or "4/7".
    
    Args:
        value: String like "2/2" or "4/7"
        
    Returns:
        Tuple of (numerator, denominator, percentage) or (None, None, None)
    """
    if not value or value.strip() == '':
        return (None, None, None)
    
    match = re.match(r'(\d+)/(\d+)', value.strip())
    if match:
        num = int(match.group(1))
        den = int(match.group(2))
        pct = num / den if den > 0 else None
        return (num, den, pct)
    
    return (None, None, None)


def parse_match_row(row, tournament_metadata: Dict[str, Any], headers: List[str] = None) -> Optional[Dict[str, Any]]:
    """
    Parse a match row from the tournament results table.
    
    Args:
        row: BeautifulSoup table row element
        tournament_metadata: Tournament metadata
        headers: Optional list of column headers
        
    Returns:
        Match record dictionary or None
    """
    cells = row.find_all(['td', 'th'])
    if len(cells) < 5:  # Need at least round, players, score
        return None
    
    # Skip header rows
    if row.find('th'):
        return None
    
    match_record = {
        **tournament_metadata,
        'round': '',
        'winner_name': '',
        'loser_name': '',
        'winner_rank': '',
        'loser_rank': '',
        'score': '',
        'time': '',
        'dr': None,
        'winner_stats': {},
        'loser_stats': {},
    }
    
    # Get cell texts
    cell_texts = [cell.get_text().strip() for cell in cells]
    
    # Parse based on header structure if available
    if headers:
        header_map = {h.lower(): i for i, h in enumerate(headers)}
        
        # Extract round
        if 'rd' in header_map:
            match_record['round'] = cell_texts[header_map['rd']]
        
        # Extract winner rank
        if 'wrk' in header_map or 'wrank' in header_map:
            idx = header_map.get('wrk', header_map.get('wrank', -1))
            if idx >= 0 and idx < len(cell_texts):
                match_record['winner_rank'] = cell_texts[idx] if cell_texts[idx].isdigit() else ''
        
        # Extract loser rank
        if 'lrk' in header_map or 'lrank' in header_map:
            idx = header_map.get('lrk', header_map.get('lrank', -1))
            if idx >= 0 and idx < len(cell_texts):
                match_record['loser_rank'] = cell_texts[idx] if cell_texts[idx].isdigit() else ''
        
        # Extract winner name (usually has a link)
        winner_idx = None
        for key in ['winner', 'player1']:
            if key in header_map:
                winner_idx = header_map[key]
                break
        if winner_idx is None:
            # Find first column with a link after round/rank
            for i in range(min(5, len(cells))):
                if cells[i].find('a'):
                    winner_idx = i
                    break
        
        if winner_idx is not None and winner_idx < len(cells):
            winner_link = cells[winner_idx].find('a')
            if winner_link:
                match_record['winner_name'] = winner_link.get_text().strip()
            else:
                match_record['winner_name'] = cell_texts[winner_idx]
        
        # Extract loser name (after "d." or winner)
        loser_idx = None
        for i in range(winner_idx + 1 if winner_idx else 0, len(cells)):
            if 'd.' in cell_texts[i].lower():
                continue
            if cells[i].find('a'):
                loser_idx = i
                break
        
        if loser_idx is not None and loser_idx < len(cells):
            loser_link = cells[loser_idx].find('a')
            if loser_link:
                match_record['loser_name'] = loser_link.get_text().strip()
            else:
                match_record['loser_name'] = cell_texts[loser_idx]
        
        # Extract score
        if 'score' in header_map:
            match_record['score'] = cell_texts[header_map['score']]
        else:
            # Find score by pattern
            for i, text in enumerate(cell_texts):
                if re.search(r'\d+-\d+', text):
                    match_record['score'] = text
                    break
        
        # Extract DR
        if 'dr' in header_map:
            try:
                dr_val = float(cell_texts[header_map['dr']])
                if 0.5 < dr_val < 3.0:
                    match_record['dr'] = dr_val
            except:
                pass
        
        # Extract winner stats: W: A%, 1stIn, 1st%, 2nd%, BPSvd
        # Look for columns after score
        score_col = header_map.get('score', -1)
        if score_col >= 0:
            stats_start = score_col + 1
            
            # Winner A%
            if stats_start < len(cell_texts):
                ace_pct = parse_percentage(cell_texts[stats_start])
                if ace_pct is not None:
                    match_record['winner_stats']['ace_pct'] = ace_pct
                stats_start += 1
            
            # Winner 1stIn
            if stats_start < len(cell_texts):
                first_in = parse_percentage(cell_texts[stats_start])
                if first_in is not None:
                    match_record['winner_stats']['first_in_pct'] = first_in
                stats_start += 1
            
            # Winner 1st%
            if stats_start < len(cell_texts):
                first_won = parse_percentage(cell_texts[stats_start])
                if first_won is not None:
                    match_record['winner_stats']['first_won_pct'] = first_won
                stats_start += 1
            
            # Winner 2nd%
            if stats_start < len(cell_texts):
                second_won = parse_percentage(cell_texts[stats_start])
                if second_won is not None:
                    match_record['winner_stats']['second_won_pct'] = second_won
                stats_start += 1
            
            # Winner BPSvd
            if stats_start < len(cell_texts):
                num, den, pct = parse_ratio(cell_texts[stats_start])
                if pct is not None:
                    match_record['winner_stats']['bpsvd'] = pct
                stats_start += 1
            
            # Loser stats: L: A%, 1stIn, 1st%, 2nd%, BPSvd
            if stats_start < len(cell_texts):
                ace_pct = parse_percentage(cell_texts[stats_start])
                if ace_pct is not None:
                    match_record['loser_stats']['ace_pct'] = ace_pct
                stats_start += 1
            
            if stats_start < len(cell_texts):
                first_in = parse_percentage(cell_texts[stats_start])
                if first_in is not None:
                    match_record['loser_stats']['first_in_pct'] = first_in
                stats_start += 1
            
            if stats_start < len(cell_texts):
                first_won = parse_percentage(cell_texts[stats_start])
                if first_won is not None:
                    match_record['loser_stats']['first_won_pct'] = first_won
                stats_start += 1
            
            if stats_start < len(cell_texts):
                second_won = parse_percentage(cell_texts[stats_start])
                if second_won is not None:
                    match_record['loser_stats']['second_won_pct'] = second_won
                stats_start += 1
            
            if stats_start < len(cell_texts):
                num, den, pct = parse_ratio(cell_texts[stats_start])
                if pct is not None:
                    match_record['loser_stats']['bpsvd'] = pct
                stats_start += 1
            
            # Time (last column usually)
            if stats_start < len(cell_texts):
                time_text = cell_texts[stats_start]
                if ':' in time_text and len(time_text) <= 6:
                    match_record['time'] = time_text
    
    # Fallback: try to parse without headers (simpler logic)
    if not match_record['winner_name']:
        # Basic parsing: round, winner_rank, winner, d., loser_rank, loser, score, ...
        if len(cells) >= 6:
            match_record['round'] = cell_texts[0]
            if cell_texts[1].isdigit():
                match_record['winner_rank'] = cell_texts[1]
            
            # Find winner (first link)
            for i, cell in enumerate(cells):
                link = cell.find('a')
                if link and not match_record['winner_name']:
                    match_record['winner_name'] = link.get_text().strip()
                    winner_idx = i
                elif link and match_record['winner_name'] and not match_record['loser_name']:
                    match_record['loser_name'] = link.get_text().strip()
                    break
            
            # Find score
            for text in cell_texts:
                if re.search(r'\d+-\d+', text):
                    match_record['score'] = text
                    break
    
    # Validate we have at least winner and loser
    if not match_record['winner_name'] or not match_record['loser_name']:
        return None
    
    return match_record


def parse_completed_matches_text(html: BeautifulSoup, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse completed matches from plain text format (used in /current/ pages).
    
    Example format:
    F: (7)Alex De Minaur (AUS) d. (12)Alejandro Davidovich Fokina (ESP) 5-7 6-1 7-6(3)
    SF: (12)Alejandro Davidovich Fokina (ESP) d. (4)Ben Shelton (USA) 6-2 7-5
    
    Args:
        html: BeautifulSoup object
        metadata: Tournament metadata
        
    Returns:
        List of match records
    """
    matches = []
    
    # Find "Completed Matches" section
    page_text = html.get_text()
    
    # Look for "Completed Matches" heading
    completed_section_match = re.search(r'Completed Matches\s*(.*?)(?=\n\n\n|\Z)', page_text, re.DOTALL | re.IGNORECASE)
    
    if not completed_section_match:
        return matches
    
    completed_text = completed_section_match.group(1)
    
    # Pattern: ROUND: (seed)Player1 (IOC) d. (seed)Player2 (IOC) score
    # Example: F: (7)Alex De Minaur (AUS) d. (12)Alejandro Davidovich Fokina (ESP) 5-7 6-1 7-6(3)
    match_pattern = re.compile(
        r'^(?P<round>[RQ]\d+|QF|SF|F|RR):\s*'  # Round
        r'(?:\((?P<winner_seed>[^)]+)\))?\s*'  # Optional winner seed
        r'(?P<winner_name>[^(]+?)\s*'  # Winner name
        r'(?:\((?P<winner_ioc>[A-Z]{3})\))?\s+'  # Optional winner IOC
        r'd\.\s*'  # "d." separator
        r'(?:\((?P<loser_seed>[^)]+)\))?\s*'  # Optional loser seed
        r'(?P<loser_name>[^(]+?)\s*'  # Loser name
        r'(?:\((?P<loser_ioc>[A-Z]{3})\))?\s*'  # Optional loser IOC
        r'(?P<score>.+?)$',  # Score
        re.MULTILINE
    )
    
    match_num = 1
    for match in match_pattern.finditer(completed_text):
        record = {
            **metadata,
            'round': match.group('round').strip(),
            'winner_name': match.group('winner_name').strip(),
            'loser_name': match.group('loser_name').strip(),
            'score': match.group('score').strip(),
            'winner_seed': match.group('winner_seed') or '',
            'loser_seed': match.group('loser_seed') or '',
            'winner_ioc': match.group('winner_ioc') or '',
            'loser_ioc': match.group('loser_ioc') or '',
            'winner_rank': '',
            'loser_rank': '',
            'time': '',
            'dr': None,
            'winner_stats': {},
            'loser_stats': {},
            'match_num': match_num,
        }
        matches.append(record)
        match_num += 1
    
    return matches


def scrape_tournament(url: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Scrape a tournament page and extract matches.
    
    Args:
        url: Tournament page URL
        
    Returns:
        Tuple of (tournament_metadata, list_of_matches)
    """
    from .http import get_session
    
    logger.info(f"Scraping tournament: {url}")
    session = get_session()
    html = session.get_html(url)
    
    if not html:
        logger.error(f"Failed to fetch tournament page: {url}")
        return ({}, [])
    
    # Extract tournament metadata
    metadata = parse_tournament_metadata(html, url)
    
    # Find "Singles Results" table
    matches = []
    
    # Method 1: Try table-based parsing (for pages with stat tables)
    for table in html.find_all('table'):
        # Check if this looks like a results table
        header_row = table.find('tr')
        if header_row:
            headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            if any(h in ['Rd', 'Round', 'wRk', 'wRk', 'Score', 'DR'] for h in headers):
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    match_record = parse_match_row(row, metadata, headers=headers)
                    if match_record:
                        matches.append(match_record)
                
                if matches:
                    logger.info(f"Found {len(matches)} matches in table")
                    break
    
    # Method 2: If no table matches found, try text-based parsing (for /current/ pages)
    if not matches:
        logger.debug(f"No table-based matches found, trying text-based parsing for {url}")
        matches = parse_completed_matches_text(html, metadata)
        if matches:
            logger.info(f"Found {len(matches)} matches from text format")
    
    logger.info(f"Scraped {len(matches)} matches from {url}")
    return (metadata, matches)

