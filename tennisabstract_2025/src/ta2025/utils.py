"""
Utility functions for scraping and data processing.
"""

import time
import requests
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
from unidecode import unidecode
import logging

logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """
    Normalize player name for matching.
    
    Args:
        name: Raw player name
        
    Returns:
        Normalized name (ASCII, uppercase, stripped)
    """
    if not name:
        return ""
    # Remove accents and convert to ASCII
    normalized = unidecode(name)
    # Uppercase and strip
    normalized = normalized.upper().strip()
    return normalized


def get_session() -> requests.Session:
    """
    Create a requests session with proper headers.
    
    Returns:
        Configured requests session
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session


def fetch_page(url: str, session: Optional[requests.Session] = None, 
               max_retries: int = 3, delay: float = 0.5) -> Optional[BeautifulSoup]:
    """
    Fetch and parse a web page with retries and rate limiting.
    
    Args:
        url: URL to fetch
        session: Optional requests session
        max_retries: Maximum number of retry attempts
        delay: Delay between requests in seconds
        
    Returns:
        BeautifulSoup object or None if failed
    """
    if session is None:
        session = get_session()
    
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            response = session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'lxml')
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                return None
    
    return None


def safe_extract_text(element, default: str = "") -> str:
    """
    Safely extract text from BeautifulSoup element.
    
    Args:
        element: BeautifulSoup element
        default: Default value if element is None or empty
        
    Returns:
        Extracted text or default
    """
    if element is None:
        return default
    text = element.get_text(strip=True)
    return text if text else default


def parse_date(date_str: str) -> Optional[str]:
    """
    Parse date string to YYYYMMDD format.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Date in YYYYMMDD format or None
    """
    from dateutil import parser as date_parser
    
    try:
        dt = date_parser.parse(date_str)
        return dt.strftime('%Y%m%d')
    except (ValueError, TypeError):
        return None


def validate_match_record(record: Dict[str, Any]) -> bool:
    """
    Validate that a match record has required fields.
    
    Args:
        record: Match record dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required = ['winner_name', 'loser_name']
    return all(record.get(field) for field in required)

