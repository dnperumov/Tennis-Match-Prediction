"""
Parse JavaScript-loaded tournament pages using browser automation.

TennisAbstract /current/ format pages load match data dynamically via JavaScript.
This module uses Playwright to load pages in a headless browser and extract the
rendered match content.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def parse_tournament_metadata_js(html_content: str, url: str) -> Dict[str, Any]:
    """
    Extract tournament metadata from JavaScript-loaded page.
    
    Args:
        html_content: Rendered HTML content
        url: Tournament page URL
        
    Returns:
        Dictionary with tournament metadata
    """
    soup = BeautifulSoup(html_content, 'lxml')
    metadata = {
        'tourney_name': '',
        'surface': '',
        'draw_size': '',
        'tourney_level': '',
        'start_date': '',
        'ta_url': url,
        'scrape_method': 'browser_automation'
    }
    
    page_text = soup.get_text()
    
    # Extract tournament name from title
    title = soup.find('title')
    if title:
        title_text = title.get_text()
        # Remove "Tennis Abstract:" prefix and clean up
        if ':' in title_text:
            metadata['tourney_name'] = title_text.split(':', 1)[1].strip()
        else:
            metadata['tourney_name'] = title_text.strip()
    
    # Extract surface
    surface_match = re.search(r'Surface:\s*(\w+)', page_text, re.IGNORECASE)
    if surface_match:
        metadata['surface'] = surface_match.group(1).strip()
    
    # Extract draw size
    draw_match = re.search(r'Draw:\s*(\d+)', page_text, re.IGNORECASE)
    if draw_match:
        metadata['draw_size'] = draw_match.group(1).strip()
    
    # Extract tournament level from name/URL
    name_lower = metadata['tourney_name'].lower()
    if any(slam in name_lower for slam in ['australian open', 'roland garros', 'french open', 'wimbledon', 'us open']):
        metadata['tourney_level'] = 'G'
    elif 'masters' in name_lower or 'finals' in name_lower:
        metadata['tourney_level'] = 'M'
    elif '500' in name_lower or any(t in name_lower for t in ['dubai', 'barcelona', 'washington', 'beijing', 'tokyo']):
        metadata['tourney_level'] = 'A'  # ATP 500 is level A
    elif 'laver cup' in name_lower:
        metadata['tourney_level'] = 'O'
    else:
        metadata['tourney_level'] = 'A'  # Default to ATP Tour
    
    # Extract date
    date_pattern = re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}')
    date_match = date_pattern.search(page_text)
    if date_match:
        try:
            date_str = date_match.group(0)
            dt = datetime.strptime(date_str, '%B %d, %Y')
            metadata['start_date'] = dt.strftime('%Y-%m-%d')
        except ValueError:
            logger.debug(f"Could not parse date string: {date_str}")
    
    return metadata


def parse_completed_matches_from_js(html_content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse completed matches from JavaScript-rendered HTML.
    
    Example format from <span id="completed">:
    F: (7)Alex De Minaur (AUS) d. (12)Alejandro Davidovich Fokina (ESP) 5-7 6-1 7-6(3)
    SF: (12)Alejandro Davidovich Fokina (ESP) d. (4)Ben Shelton (USA) 6-2 7-5
    
    Args:
        html_content: Rendered HTML content
        metadata: Tournament metadata
        
    Returns:
        List of match records
    """
    matches = []
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Find the "completed" span that contains matches
    completed_span = soup.find('span', id='completed')
    
    if not completed_span:
        logger.warning(f"No 'completed' span found in {metadata.get('ta_url', 'unknown URL')}")
        return matches
    
    # Get the HTML content (to preserve structure)
    completed_html = str(completed_span)
    
    # Convert <br> tags to newlines for proper line-by-line parsing
    completed_html = completed_html.replace('<br>', '\n').replace('<br/>', '\n')
    
    # Now extract text
    completed_soup = BeautifulSoup(completed_html, 'lxml')
    completed_text = completed_soup.get_text()
    
    if not completed_text or completed_text.strip() == '':
        logger.info(f"No completed matches in {metadata.get('tourney_name', 'unknown tournament')}")
        return matches
    
    # Split into lines and parse each line
    lines = completed_text.split('\n')
    
    # Pattern: ROUND: (seed)Player1 (IOC) d. (seed)Player2 (IOC) score
    # Use a simpler pattern that captures broadly, then clean up
    match_pattern = re.compile(
        r'^(?P<round>[RQ]\d+|QF|SF|F|RR):\s*'  # Round
        r'(?:\((?P<winner_seed>[^)]+)\))?\s*'  # Optional winner seed
        r'(?P<winner_part>.+?)\s+d\.\s+'  # Everything before "d." (winner + IOC)
        r'(?:\((?P<loser_seed>[^)]+)\))?\s*'  # Optional loser seed
        r'(?P<loser_part>.+?)\s+'  # Everything after "d." before score (loser + IOC)
        r'(?P<score>[\d\-]+.*)$',  # Score (starts with digits)
        re.IGNORECASE
    )
    
    def clean_player_name(text: str) -> Tuple[str, str]:
        """Extract player name and IOC from text like 'Alex De Minaur (AUS)'"""
        text = text.strip()
        # Remove extra hyphens and spaces (artifacts from HTML)
        text = re.sub(r'\s*-\s*', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Extract IOC if present
        ioc_match = re.search(r'\(([A-Z]{3})\)\s*$', text)
        if ioc_match:
            ioc = ioc_match.group(1)
            name = text[:ioc_match.start()].strip()
        else:
            ioc = ''
            name = text.strip()
        
        return (name, ioc)
    
    match_num = 1
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        match = match_pattern.match(line)
        if match:
            # Clean up winner and loser parts
            winner_name, winner_ioc_extracted = clean_player_name(match.group('winner_part'))
            loser_name, loser_ioc_extracted = clean_player_name(match.group('loser_part'))
            
            record = {
                **metadata,
                'round': match.group('round').strip(),
                'winner_name': winner_name,
                'loser_name': loser_name,
                'score': match.group('score').strip(),
                'winner_seed': match.group('winner_seed') or '',
                'loser_seed': match.group('loser_seed') or '',
                'winner_ioc': winner_ioc_extracted,
                'loser_ioc': loser_ioc_extracted,
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
    
    logger.info(f"Parsed {len(matches)} matches from JavaScript-loaded content")
    return matches


def scrape_tournament_with_browser(url: str, timeout: int = 30000) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Scrape a JavaScript-loaded tournament page using Playwright.
    
    Args:
        url: Tournament page URL
        timeout: Page load timeout in milliseconds (default 30 seconds)
        
    Returns:
        Tuple of (tournament_metadata, list_of_matches)
    """
    logger.info(f"Scraping with browser automation: {url}")
    
    try:
        with sync_playwright() as p:
            # Launch headless browser
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = context.new_page()
            
            # Navigate to URL
            logger.debug(f"Loading page: {url}")
            page.goto(url, wait_until='networkidle', timeout=timeout)
            
            # Wait for the "completed" span to be present (even if empty)
            try:
                page.wait_for_selector('#completed', timeout=5000)
                logger.debug("Found #completed span")
            except PlaywrightTimeout:
                logger.warning(f"Timeout waiting for #completed span on {url}")
            
            # Get the fully rendered HTML
            html_content = page.content()
            
            # Close browser
            browser.close()
            
            # Parse metadata and matches
            metadata = parse_tournament_metadata_js(html_content, url)
            matches = parse_completed_matches_from_js(html_content, metadata)
            
            logger.info(f"Browser scraping complete: {len(matches)} matches from {url}")
            return (metadata, matches)
            
    except PlaywrightTimeout as e:
        logger.error(f"Timeout loading page {url}: {e}")
        return ({}, [])
    except Exception as e:
        logger.error(f"Error scraping with browser {url}: {e}")
        return ({}, [])


def is_js_tournament_url(url: str) -> bool:
    """
    Determine if a URL is a JavaScript-loaded tournament page.
    
    Args:
        url: Tournament page URL
        
    Returns:
        True if URL uses /current/ format (JavaScript-loaded)
    """
    return '/current/' in url and '.html' in url

