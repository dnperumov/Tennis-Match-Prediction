"""
Scrape individual tournament pages from TennisAbstract.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
from .utils import fetch_page, safe_extract_text, get_session

logger = logging.getLogger(__name__)


def extract_tournament_metadata(html: BeautifulSoup, url: str) -> Dict[str, Any]:
    """
    Extract tournament-level metadata from page.
    
    Args:
        html: BeautifulSoup object of tournament page
        url: Source URL
        
    Returns:
        Dictionary with tournament metadata
    """
    metadata = {
        'tourney_name': '',
        'surface': '',
        'tourney_date': '',
        'tourney_id': '',
        'source_url': url
    }
    
    # Extract tournament name from URL or page
    # URL format: .../2025ATPTournamentName.html
    import re
    url_match = re.search(r'(\d{4})ATP(.+?)\.html', url, re.IGNORECASE)
    if url_match:
        year = url_match.group(1)
        tourney_name = url_match.group(2).replace('-', ' ').strip()
        metadata['tourney_name'] = tourney_name
        metadata['tourney_id'] = f"{year}{tourney_name.upper().replace(' ', '')[:10]}"
    
    # Try to extract from page title as fallback
    if not metadata['tourney_name']:
        title = html.find('title')
        if title:
            title_text = safe_extract_text(title)
            if ' - ' in title_text:
                metadata['tourney_name'] = title_text.split(' - ')[0].strip()
            else:
                metadata['tourney_name'] = title_text
    
    # Try to find surface information
    # Look for common surface indicators in the page
    page_text = html.get_text().upper()
    if 'HARD' in page_text or 'HARDCOURT' in page_text:
        metadata['surface'] = 'Hard'
    elif 'CLAY' in page_text:
        metadata['surface'] = 'Clay'
    elif 'GRASS' in page_text:
        metadata['surface'] = 'Grass'
    
    # Try to extract date from URL or page
    # Look for date patterns in URL or page content
    date_match = re.search(r'(\d{4})(\d{2})(\d{2})', url)
    if date_match:
        metadata['tourney_date'] = ''.join(date_match.groups())
    else:
        # Try to find year from URL
        year_match = re.search(r'(\d{4})', url)
        if year_match:
            year = year_match.group(1)
            # Default to first day of year if no specific date
            metadata['tourney_date'] = f"{year}0101"
    
    return metadata


def parse_text_based_matches(html: BeautifulSoup, tournament_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse completed matches from text format (not table).
    Format: "R32: (WC)Yibing Wu (CHN) d. Fabian Marozsan (HUN) 6-4 6-2"
    
    Args:
        html: BeautifulSoup object
        tournament_metadata: Tournament metadata
        
    Returns:
        List of match records
    """
    matches = []
    
    # Find "Completed Matches" text
    completed_section = None
    for element in html.find_all(string=re.compile(r'Completed Matches', re.I)):
        parent = element.find_parent()
        if parent:
            completed_section = parent
            break
    
    if not completed_section:
        return []
    
    # Get all text after "Completed Matches"
    # Look for lines matching the pattern: Round: Player d. Player Score
    # Pattern: R32, R16, QF, SF, F, Q1, Q2, etc. followed by colon, then match details
    match_pattern = re.compile(
        r'([RQ]\d+|QF|SF|F):\s*'  # Round (R32, Q1, QF, etc.)
        r'(?:\([^)]+\))?\s*'  # Optional seed/entry like (WC), (8), (Q)
        r'([A-Z][a-zA-Z\s\-\.]+?)\s+'  # Winner name
        r'\(([A-Z]{3})\)\s+'  # Winner country code
        r'd\.\s+'  # "d." separator
        r'(?:\([^)]+\))?\s*'  # Optional seed/entry
        r'([A-Z][a-zA-Z\s\-\.]+?)\s+'  # Loser name
        r'\(([A-Z]{3})\)\s+'  # Loser country code
        r'(\d+-\d+(?:\s+\d+-\d+)*)',  # Score
        re.IGNORECASE | re.MULTILINE
    )
    
    # Check for JavaScript variable containing matches (common on TennisAbstract)
    # Look for: var completedSingles = '...'
    section_text = ""
    found_matches = []
    
    for script in html.find_all('script'):
        script_text = script.string
        if not script_text:
            continue
        
        # Look for completedSingles or completedDoubles variable
        completed_match = re.search(r"var\s+completed(?:Singles|Doubles)\s*=\s*['\"](.*?)['\"];", script_text, re.DOTALL)
        if completed_match:
            html_content = completed_match.group(1)
            # Unescape HTML entities
            html_content = html_content.replace('&nbsp;', ' ').replace('&amp;', '&').replace('<br/>', '\n').replace('<br>', '\n')
            
            # Parse the HTML to extract text
            try:
                match_html = BeautifulSoup(html_content, 'lxml')
                section_text = match_html.get_text()
            except:
                # Fallback: just remove HTML tags with regex
                section_text = re.sub(r'<[^>]+>', ' ', html_content)
            
            logger.debug(f"Found completedSingles/Doubles variable with {len(section_text)} chars")
            break
    
    # If not found in JavaScript variable, check regular HTML text
    if not section_text:
        parent_container = completed_section.find_parent(['div', 'p', 'td', 'body', 'html'])
        if parent_container:
            section_text = parent_container.get_text()
        else:
            section_text = html.get_text()
        
        # Find the position of "Completed Matches" and get text after it
        completed_pos = section_text.find('Completed Matches')
        if completed_pos >= 0:
            section_text = section_text[completed_pos:]
    
    # Find all matches in the text
    found_matches = match_pattern.findall(section_text)
    
    match_num = 1
    for round_name, winner_name, winner_country, loser_name, loser_country, score in found_matches:
        # Clean up names
        winner_name = re.sub(r'\s+', ' ', winner_name).strip()
        loser_name = re.sub(r'\s+', ' ', loser_name).strip()
        
        # Remove any remaining parentheses or special chars from names
        winner_name = re.sub(r'^\([^)]+\)\s*', '', winner_name).strip()
        loser_name = re.sub(r'^\([^)]+\)\s*', '', loser_name).strip()
        
        # Clean score
        score = re.sub(r'\s+', ' ', score).strip()
        
        match_record = {
            'round': round_name.strip(),
            'winner_name': winner_name,
            'loser_name': loser_name,
            'score': score,
            'match_num': str(match_num),
            **tournament_metadata
        }
        
        matches.append(match_record)
        match_num += 1
    
    return matches


def find_completed_matches_section(html: BeautifulSoup) -> Optional[Any]:
    """
    Locate the "Completed Matches" section on the page.
    
    Args:
        html: BeautifulSoup object
        
    Returns:
        BeautifulSoup element containing completed matches or None
    """
    # First, check for JavaScript-embedded tables (common on TennisAbstract)
    # Look for tables that contain actual match results (not forecasts)
    for script in html.find_all('script'):
        script_text = script.string
        if not script_text:
            continue
        
        # Look for table HTML in JavaScript variables
        # Pattern: var projXX = '<table...>' or var matchesXX = '<table...>'
        table_patterns = [
            r"var\s+(?:matches|results|completed)\d*\s*=\s*['\"](<table[^>]*>.*?</table>)['\"]",
            r"var\s+proj\d+\s*=\s*['\"](<table[^>]*>.*?</table>)['\"]",
        ]
        
        for pattern in table_patterns:
            table_match = re.search(pattern, script_text, re.DOTALL | re.IGNORECASE)
            if table_match:
                table_html = table_match.group(1)
                # Unescape HTML entities
                table_html = table_html.replace('&nbsp;', ' ').replace('&amp;', '&').replace('\\"', '"').replace("\\'", "'")
                try:
                    embedded_table = BeautifulSoup(table_html, 'lxml')
                    table = embedded_table.find('table')
                    if table:
                        # Check if this looks like match results (has "def" or scores)
                        table_text = table.get_text()
                        if re.search(r'def\.?|d\.?|\d+-\d+', table_text, re.IGNORECASE):
                            logger.debug("Found match results table in JavaScript")
                            return table
                except Exception as e:
                    logger.debug(f"Error parsing embedded table: {e}")
                    continue
        
        # Also look for table HTML directly in script (without var assignment)
        table_match = re.search(r'(<table[^>]*cellpadding.*?</table>)', script_text, re.DOTALL | re.IGNORECASE)
        if table_match:
            table_html = table_match.group(1)
            table_html = table_html.replace('&nbsp;', ' ').replace('&amp;', '&').replace('\\"', '"')
            try:
                embedded_table = BeautifulSoup(table_html, 'lxml')
                table = embedded_table.find('table')
                if table:
                    table_text = table.get_text()
                    # Only return if it looks like match results
                    if re.search(r'def\.?|d\.?|\d+-\d+', table_text, re.IGNORECASE):
                        return table
            except Exception:
                continue
    
    # Look for section with exact text "Completed Matches"
    for header in html.find_all(['h2', 'h3', 'h4', 'div', 'span', 'p', 'b'], string=re.compile(r'Completed Matches', re.I)):
        # Return the parent container
        parent = header.find_parent(['div', 'section', 'table', 'body'])
        if parent and hasattr(parent, 'find'):
            # Look for table in this section
            table = parent.find('table')
            if table:
                return table
    
    # Alternative: look for text node and find following table
    for text_node in html.find_all(string=re.compile(r'Completed Matches', re.I)):
        if not hasattr(text_node, 'find_parent'):
            continue
            
        parent = text_node.find_parent()
        if not parent:
            continue
        
        # Look for next table in siblings
        current = parent.next_sibling
        while current:
            if hasattr(current, 'name'):
                if current.name == 'table':
                    return current
                elif hasattr(current, 'find') and current.find('table'):
                    return current.find('table')
            current = current.next_sibling
        
        # Also check parent's next siblings
        parent_sibling = parent.next_sibling
        while parent_sibling:
            if hasattr(parent_sibling, 'name'):
                if parent_sibling.name == 'table':
                    return parent_sibling
                elif hasattr(parent_sibling, 'find') and parent_sibling.find('table'):
                    return parent_sibling.find('table')
            parent_sibling = parent_sibling.next_sibling
    
    # Last resort: find any table on the page (if structure is simple)
    tables = html.find_all('table')
    if len(tables) == 1:
        return tables[0]
    
    return None


def parse_match_row(row, tournament_metadata: Dict[str, Any], match_num: int = 0) -> Optional[Dict[str, Any]]:
    """
    Parse a single match row from the table.
    
    Args:
        row: BeautifulSoup table row element
        tournament_metadata: Tournament-level metadata
        match_num: Match number in tournament
        
    Returns:
        Match record dictionary or None if parsing failed
    """
    cells = row.find_all(['td', 'th'])
    if len(cells) < 1:  # Need at least some data
        return None
    
    match_record = {
        'round': '',
        'winner_name': '',
        'loser_name': '',
        'score': '',
        'match_num': str(match_num),
        **tournament_metadata
    }
    
    # Extract round (usually first column)
    if len(cells) > 0:
        round_text = safe_extract_text(cells[0])
        # Clean round text
        round_text = re.sub(r'^\d+\.?\s*', '', round_text)  # Remove leading numbers
        match_record['round'] = round_text
    
    # Get full row text for pattern matching
    row_text = row.get_text()
    
    # Look for "def." or "d." pattern (most common in TennisAbstract)
    def_pattern = re.compile(r'([A-Z][a-zA-Z\s\-\.]+?)\s+(?:def\.?|d\.?|defeated)\s+([A-Z][a-zA-Z\s\-\.]+?)(?:\s+(\d+-\d+(?:\s*[,\s]\d+-\d+)*))?', re.IGNORECASE)
    match = def_pattern.search(row_text)
    
    if match:
        winner_text = match.group(1).strip()
        loser_text = match.group(2).strip()
        score_text = match.group(3).strip() if match.group(3) else ''
        
        # Clean up names (remove extra whitespace, numbers, etc.)
        winner_text = re.sub(r'\s+', ' ', winner_text).strip()
        loser_text = re.sub(r'\s+', ' ', loser_text).strip()
        
        # Remove score from names if it got captured
        score_pattern = re.compile(r'\d+-\d+')
        winner_text = score_pattern.sub('', winner_text).strip()
        loser_text = score_pattern.sub('', loser_text).strip()
        
        match_record['winner_name'] = winner_text
        match_record['loser_name'] = loser_text
        if score_text:
            match_record['score'] = score_text.replace(' ', '')
    else:
        # Try column-based extraction (if structured as separate columns)
        # Winner might be in column 1 or 2, loser in 2 or 3
        if len(cells) >= 2:
            # Try to identify which column has names vs scores
            for i, cell in enumerate(cells[1:], 1):
                cell_text = safe_extract_text(cell)
                # If it looks like a name (has capital letters, no numbers)
                if re.match(r'^[A-Z][a-zA-Z\s]+$', cell_text) and not match_record['winner_name']:
                    match_record['winner_name'] = cell_text
                elif re.match(r'^[A-Z][a-zA-Z\s]+$', cell_text) and match_record['winner_name'] and not match_record['loser_name']:
                    match_record['loser_name'] = cell_text
                # If it looks like a score
                elif re.search(r'\d+-\d+', cell_text) and not match_record['score']:
                    match_record['score'] = cell_text.replace(' ', '')
    
    # Extract score if not already found
    if not match_record['score']:
        score_pattern = re.compile(r'(\d+-\d+(?:\s*[,\s]\d+-\d+)*)')
        score_match = score_pattern.search(row_text)
        if score_match:
            match_record['score'] = score_match.group(1).strip().replace(' ', '')
        elif len(cells) >= 3:
            # Try last column
            last_cell_text = safe_extract_text(cells[-1])
            if re.search(r'\d+-\d+', last_cell_text):
                match_record['score'] = last_cell_text.replace(' ', '')
    
    # Clean up names - remove common prefixes/suffixes and invalid chars
    for name_field in ['winner_name', 'loser_name']:
        name = match_record[name_field]
        if not name:
            continue
        
        # Remove common prefixes (qualifier, wildcard, etc.)
        name = re.sub(r'^(Q|LL|WC|SE|PR|Alt|\[|\()\s*', '', name, flags=re.IGNORECASE)
        # Remove trailing numbers, brackets, parentheses
        name = re.sub(r'\s*[\(\[].*?[\)\]]\s*$', '', name)
        name = re.sub(r'\s*\d+\s*$', '', name)
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        # Remove trailing dots and dashes
        name = name.rstrip('.- ')
        
        match_record[name_field] = name
    
    # Validate we have at least winner and loser with reasonable names
    if not match_record['winner_name'] or not match_record['loser_name']:
        return None
    
    # Additional validation: names should be at least 3 characters and contain letters
    if len(match_record['winner_name']) < 3 or len(match_record['loser_name']) < 3:
        return None
    
    if not re.search(r'[A-Za-z]', match_record['winner_name']) or not re.search(r'[A-Za-z]', match_record['loser_name']):
        return None
    
    return match_record


def scrape_tournament(url: str) -> List[Dict[str, Any]]:
    """
    Scrape all completed matches from a tournament page.
    
    Args:
        url: Tournament page URL
        
    Returns:
        List of match records
    """
    logger.info(f"Scraping tournament: {url}")
    
    html = fetch_page(url)
    if not html:
        logger.error(f"Failed to fetch tournament page: {url}")
        return []
    
    # Extract tournament metadata
    tournament_metadata = extract_tournament_metadata(html, url)
    logger.debug(f"Tournament metadata: {tournament_metadata}")
    
    # Find completed matches section
    matches_section = find_completed_matches_section(html)
    if not matches_section:
        logger.warning(f"Could not find 'Completed Matches' section on {url}")
        # Try to find any table as fallback
        tables = html.find_all('table')
        if not tables:
            logger.error(f"No tables found on page: {url}")
            # Try alternative: look for divs with match-like content
            logger.info("Trying alternative parsing: looking for match patterns in page text...")
            return _parse_matches_from_text(html, tournament_metadata, url)
        logger.info(f"Using fallback: found {len(tables)} table(s) on page")
    else:
        # Find all tables in the section
        if hasattr(matches_section, 'name') and matches_section.name == 'table':
            tables = [matches_section]
        elif hasattr(matches_section, 'find_all'):
            tables = matches_section.find_all('table')
        else:
            tables = []
    
    matches = []
    match_num = 1
    for table in tables:
        rows = table.find_all('tr')
        logger.debug(f"Processing table with {len(rows)} rows")
        
        for row in rows:
            # Skip header rows
            if row.find('th'):
                continue
            
            match_record = parse_match_row(row, tournament_metadata, match_num=match_num)
            if match_record:
                matches.append(match_record)
                match_num += 1
    
    # If no matches found in tables, try text-based parsing
    if not matches:
        logger.info("No matches found in tables, trying text-based parsing...")
        # First try the new text-based parser for "Completed Matches" section
        text_matches = parse_text_based_matches(html, tournament_metadata)
        if text_matches:
            matches = text_matches
        else:
            # Fallback to old text parsing
            matches = _parse_matches_from_text(html, tournament_metadata, url)
    
    if not matches:
        logger.warning(f"No completed matches found on {url}")
        logger.warning("This may be because:")
        logger.warning("  - Tournament hasn't started yet")
        logger.warning("  - No matches have been completed")
        logger.warning("  - Page structure is different than expected")
    
    logger.info(f"Scraped {len(matches)} matches from {url}")
    return matches


def _parse_matches_from_text(html: BeautifulSoup, tournament_metadata: Dict[str, Any], url: str) -> List[Dict[str, Any]]:
    """
    Fallback: Parse matches from page text when table structure isn't found.
    Also checks JavaScript content for match data.
    
    Args:
        html: BeautifulSoup object
        tournament_metadata: Tournament metadata
        url: Source URL
        
    Returns:
        List of match records
    """
    matches = []
    
    # First, check all script tags for match data
    for script in html.find_all('script'):
        script_text = script.string
        if not script_text:
            continue
        
        # Look for match patterns in JavaScript strings
        # Pattern: "Player1 def. Player2" or "Player1 d. Player2" with scores
        def_pattern = re.compile(r'([A-Z][a-zA-Z\s\-\.]+?)\s+(?:def\.?|d\.?|defeated)\s+([A-Z][a-zA-Z\s\-\.]+?)(?:\s+(\d+-\d+(?:\s*[,\s]\d+-\d+)*))?', re.IGNORECASE)
        found_matches = def_pattern.findall(script_text)
        
        match_num = len(matches) + 1
        for winner, loser, score in found_matches:
            # Clean up names
            winner = re.sub(r'\s+', ' ', winner).strip()
            loser = re.sub(r'\s+', ' ', loser).strip()
            
            # Remove HTML tags if present
            winner = re.sub(r'<[^>]+>', '', winner).strip()
            loser = re.sub(r'<[^>]+>', '', loser).strip()
            
            # Skip if names are too short or contain invalid chars
            if len(winner.split()) < 2 or len(loser.split()) < 2:
                continue
            if '<' in winner or '<' in loser:
                continue
            
            match_record = {
                'round': '',
                'winner_name': winner,
                'loser_name': loser,
                'score': score.strip().replace(' ', '') if score else '',
                'match_num': str(match_num),
                **tournament_metadata
            }
            
            matches.append(match_record)
            match_num += 1
    
    # If no matches found in scripts, try page text
    if not matches:
        page_text = html.get_text()
        def_pattern = re.compile(r'([A-Z][a-zA-Z\s\-\.]+?)\s+(?:def\.?|d\.?|defeated)\s+([A-Z][a-zA-Z\s\-\.]+?)(?:\s+(\d+-\d+(?:\s*,\s*\d+-\d+)*))?', re.IGNORECASE | re.MULTILINE)
        
        found_matches = def_pattern.findall(page_text)
        match_num = 1
        
        for winner, loser, score in found_matches:
            # Clean up names
            winner = re.sub(r'\s+', ' ', winner).strip()
            loser = re.sub(r'\s+', ' ', loser).strip()
            
            # Skip if names are too short (likely false positives)
            if len(winner.split()) < 2 or len(loser.split()) < 2:
                continue
            
            match_record = {
                'round': '',
                'winner_name': winner,
                'loser_name': loser,
                'score': score.strip() if score else '',
                'match_num': str(match_num),
                **tournament_metadata
            }
            
            matches.append(match_record)
            match_num += 1
    
    return matches


def scrape_all_tournaments(tournament_urls: List[str]) -> List[Dict[str, Any]]:
    """
    Scrape all tournaments from a list of URLs.
    
    Args:
        tournament_urls: List of tournament URLs
        
    Returns:
        List of all match records
    """
    all_matches = []
    
    for url in tournament_urls:
        try:
            matches = scrape_tournament(url)
            all_matches.extend(matches)
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}", exc_info=True)
    
    logger.info(f"Total matches scraped: {len(all_matches)}")
    return all_matches

