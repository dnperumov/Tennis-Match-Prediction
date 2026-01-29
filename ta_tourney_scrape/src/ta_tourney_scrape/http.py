"""
HTTP client with caching, rate limiting, and retries.
"""

import time
import logging
from typing import Optional
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse

import requests
from requests_cache import CachedSession
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.tennisabstract.com"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


class ThrottledSession:
    """Session with rate limiting and caching."""
    
    def __init__(self, cache_name: str = "ta_cache", rate_limit: float = 2.5, max_concurrent: int = 1):
        """
        Initialize throttled session.
        
        Args:
            cache_name: Name for requests-cache database
            rate_limit: Minimum seconds between requests (default 2.5s to avoid 429 errors)
            max_concurrent: Maximum concurrent requests (default 1 for safety)
        """
        self.rate_limit = rate_limit
        self.max_concurrent = max_concurrent
        self.last_request_time = 0.0
        self.active_requests = 0
        
        # Create cached session
        self.session = CachedSession(
            cache_name,
            expire_after=86400 * 7,  # 7 days
            backend='sqlite'
        )
        self.session.headers.update({'User-Agent': USER_AGENT})
        
        # Check robots.txt
        self.robots_parser = self._load_robots_txt()
    
    def _load_robots_txt(self) -> Optional[RobotFileParser]:
        """Load and parse robots.txt."""
        try:
            robots_url = urljoin(BASE_URL, '/robots.txt')
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            logger.info("Loaded robots.txt")
            return rp
        except Exception as e:
            logger.warning(f"Could not load robots.txt: {e}")
            return None
    
    def _check_robots(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        if self.robots_parser is None:
            return True  # If we can't load robots.txt, proceed
        
        if not self.robots_parser.can_fetch(USER_AGENT, url):
            logger.error(f"robots.txt disallows: {url}")
            return False
        return True
    
    def _throttle(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException,)),
        reraise=True
    )
    def get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """
        GET request with throttling and retries.
        
        Args:
            url: URL to fetch
            **kwargs: Additional arguments for requests.get
            
        Returns:
            Response object or None if failed
        """
        # Check robots.txt
        if not self._check_robots(url):
            raise ValueError(f"robots.txt disallows: {url}")
        
        # Throttle
        self._throttle()
        
        try:
            response = self.session.get(url, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            raise
    
    def get_html(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse HTML.
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object or None
        """
        response = self.get(url)
        if response is None:
            return None
        
        return BeautifulSoup(response.content, 'lxml')


# Global session instance
_session: Optional[ThrottledSession] = None


def get_session(cache_name: str = "ta_cache", rate_limit: float = 1.0) -> ThrottledSession:
    """Get or create global session."""
    global _session
    if _session is None:
        _session = ThrottledSession(cache_name=cache_name, rate_limit=rate_limit)
    return _session

