"""
Discover TennisAbstract ATP 2025 tournament pages.
"""

import re
import logging
from typing import List, Set
from bs4 import BeautifulSoup
from .utils import fetch_page, get_session

logger = logging.getLogger(__name__)

# Base URLs for TennisAbstract
BASE_URL = "https://www.tennisabstract.com"
ATP_INDEX_PATTERNS = [
    "{}/current/{}ATP.html",
    "{}/current/{}.html",
    "{}/current/",
]

# Fallback tournament URLs (if discovery fails)
FALLBACK_TOURNAMENTS_2025 = [
    # Add known 2025 tournament URLs here if needed
    # Example: f"{BASE_URL}/current/2025ATP-AustralianOpen.html"
]

FALLBACK_TOURNAMENTS_2026 = [
    "https://www.tennisabstract.com/current/2026ATPHongKong.html",
    # Add more 2026 tournaments as needed
]


def find_tournament_links(html: BeautifulSoup, year: int = 2025) -> Set[str]:
    """
    Extract tournament links from an index page.
    
    Args:
        html: BeautifulSoup object of the index page
        year: Year to filter for
        
    Returns:
        Set of tournament URLs
    """
    tournament_urls = set()
    
    # Pattern to match ATP tournament pages for the given year
    pattern = re.compile(rf'/current/{year}ATP.*\.html', re.IGNORECASE)
    
    # Find all links
    for link in html.find_all('a', href=True):
        href = link.get('href', '')
        if pattern.search(href):
            # Make absolute URL
            if href.startswith('/'):
                full_url = f"{BASE_URL}{href}"
            elif href.startswith('http'):
                full_url = href
            else:
                full_url = f"{BASE_URL}/current/{href}"
            
            tournament_urls.add(full_url)
    
    # Also try to find links in page text (some pages use different structures)
    page_text = html.get_text()
    text_matches = pattern.findall(page_text)
    for match in text_matches:
        if match.startswith('/'):
            tournament_urls.add(f"{BASE_URL}{match}")
        elif match.startswith('http'):
            tournament_urls.add(match)
    
    return tournament_urls


def discover_tournaments(year: int = 2025) -> List[str]:
    """
    Discover all ATP tournament pages for a given year.
    
    Args:
        year: Year to discover tournaments for
        
    Returns:
        List of tournament URLs
    """
    logger.info(f"Discovering ATP {year} tournaments...")
    
    all_tournaments = set()
    session = get_session()
    
    # Try each index pattern
    for pattern in ATP_INDEX_PATTERNS:
        index_url = pattern.format(BASE_URL, year)
        logger.info(f"Trying index: {index_url}")
        html = fetch_page(index_url, session=session)
        
        if html:
            tournaments = find_tournament_links(html, year=year)
            if tournaments:
                logger.info(f"Found {len(tournaments)} tournaments from {index_url}")
                all_tournaments.update(tournaments)
                break
    
    # If discovery failed, use fallback
    if not all_tournaments:
        logger.warning("Discovery failed, using fallback tournament list")
        if year == 2025:
            all_tournaments.update(FALLBACK_TOURNAMENTS_2025)
        elif year == 2026:
            all_tournaments.update(FALLBACK_TOURNAMENTS_2026)
    
    tournament_list = sorted(list(all_tournaments))
    logger.info(f"Total tournaments discovered: {len(tournament_list)}")
    
    return tournament_list


def validate_tournament_url(url: str) -> bool:
    """
    Validate that a URL looks like a TennisAbstract tournament page.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = re.compile(r'/current/\d{4}ATP.*\.html', re.IGNORECASE)
    return bool(pattern.search(url))


def add_tournament_url(url: str, year: int = 2025):
    """
    Manually add a tournament URL to the fallback list.
    
    Args:
        url: Tournament URL to add
        year: Year of the tournament
    """
    if year == 2025:
        FALLBACK_TOURNAMENTS_2025.append(url)
    elif year == 2026:
        FALLBACK_TOURNAMENTS_2026.append(url)

