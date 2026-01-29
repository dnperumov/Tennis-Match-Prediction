#!/usr/bin/env python3
"""
Test script for the TennisAbstract scraper.
Tests with the provided sample URL.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Check dependencies first
try:
    from bs4 import BeautifulSoup
    import requests
except ImportError as e:
    print("❌ Missing dependencies!")
    print(f"Error: {e}")
    print("\nPlease install dependencies:")
    print("  pip install beautifulsoup4 requests pandas unidecode tqdm python-dateutil lxml")
    print("\nOr install the package:")
    print("  pip install -e .")
    sys.exit(1)

from ta2025.scrape_tournament import scrape_tournament
from ta2025.discover import validate_tournament_url

# Sample URL provided by user
SAMPLE_URL = "https://www.tennisabstract.com/current/2026ATPHongKong.html"

def test_tournament_scraping():
    """Test scraping the sample tournament."""
    print("=" * 60)
    print("Testing Tournament Scraping")
    print("=" * 60)
    
    # Validate URL
    if not validate_tournament_url(SAMPLE_URL):
        print(f"❌ Invalid tournament URL format: {SAMPLE_URL}")
        return False
    
    print(f"✅ URL format valid: {SAMPLE_URL}")
    
    # Test scraping
    print(f"\nScraping tournament: {SAMPLE_URL}")
    try:
        matches = scrape_tournament(SAMPLE_URL)
        
        if matches:
            print(f"\n✅ Successfully scraped {len(matches)} matches")
            
            # Group by round
            rounds = {}
            for match in matches:
                round_name = match.get('round', 'Unknown')
                if round_name not in rounds:
                    rounds[round_name] = []
                rounds[round_name].append(match)
            
            print(f"\nMatches by round:")
            for round_name in sorted(rounds.keys()):
                print(f"  {round_name}: {len(rounds[round_name])} matches")
            
            print("\nSample match records:")
            for i, match in enumerate(matches[:5], 1):
                print(f"\nMatch {i}:")
                print(f"  Round: {match.get('round', 'N/A')}")
                print(f"  Winner: {match.get('winner_name', 'N/A')}")
                print(f"  Loser: {match.get('loser_name', 'N/A')}")
                print(f"  Score: {match.get('score', 'N/A')}")
                print(f"  Tournament: {match.get('tourney_name', 'N/A')}")
            
            if len(matches) > 5:
                print(f"\n... and {len(matches) - 5} more matches")
            
            return True
        else:
            print("❌ No matches found")
            return False
            
    except Exception as e:
        print(f"❌ Error scraping tournament: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_tournament_scraping()
    sys.exit(0 if success else 1)

