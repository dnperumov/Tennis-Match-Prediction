"""
Discover TennisAbstract tournament URLs for a given year.

STRATEGY:
1. Try to discover from TennisAbstract index pages (cgi-bin/index.cgi, tournaments/)
2. If that fails, use a COMPLETE seed list of ALL 2025 ATP tournaments
3. Optionally supplement with base dataset tournament names + search
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Set
from urllib.parse import urljoin, quote
from datetime import datetime

from .http import get_session, BASE_URL

logger = logging.getLogger(__name__)


# COMPLETE seed list of ALL 2025 ATP tournaments
# Based on official 2025 ATP Tour calendar
# IMPORTANT: TennisAbstract uses TWO URL formats:
# 1. /cgi-bin/tourney.cgi?t=YYYY-XXXX%2FName (older tournaments, some current)
# 2. /current/YYYYATPName.html (many 2025 tournaments use this format)
SEED_ATP_2025_TOURNAMENTS = [
    # Grand Slams (both formats)
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0580%2FAustralian-Open",
    "https://www.tennisabstract.com/current/2025ATPAustralianOpen.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0520%2FRoland-Garros",
    "https://www.tennisabstract.com/current/2025ATPRolandGarros.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0540%2FWimbledon",
    "https://www.tennisabstract.com/current/2025ATPWimbledon.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0560%2FUS-Open",
    "https://www.tennisabstract.com/current/2025ATPUSOpen.html",
    
    # ATP Masters 1000 (both formats)
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0416%2FIndian-Wells",
    "https://www.tennisabstract.com/current/2025ATPIndianWells.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0418%2FMiami",
    "https://www.tennisabstract.com/current/2025ATPMiami.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0316%2FMonte-Carlo",
    "https://www.tennisabstract.com/current/2025ATPMonteCarlo.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0417%2FMadrid",
    "https://www.tennisabstract.com/current/2025ATPMadrid.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0416%2FRome",
    "https://www.tennisabstract.com/current/2025ATPRome.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0421%2FCanada",  # Toronto
    "https://www.tennisabstract.com/current/2025ATPToronto.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0422%2FCincinnati",
    "https://www.tennisabstract.com/current/2025ATPCincinnati.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-5014%2FShanghai",
    "https://www.tennisabstract.com/current/2025ATPShanghai.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0352%2FParis-Masters",
    "https://www.tennisabstract.com/current/2025ATPParis.html",
    
    # ATP 500 (both formats)
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0319%2FBarcelona",
    "https://www.tennisabstract.com/current/2025ATPBarcelona.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0407%2FRotterdam",
    "https://www.tennisabstract.com/current/2025ATPRotterdam.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0403%2FDubai",
    "https://www.tennisabstract.com/current/2025ATPDubai.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0352%2FDoha",
    "https://www.tennisabstract.com/current/2025ATPDoha.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0773%2FRio-de-Janeiro",
    "https://www.tennisabstract.com/current/2025ATPRio.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0404%2FAcapulco",
    "https://www.tennisabstract.com/current/2025ATPAcapulco.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0533%2FDallas",
    "https://www.tennisabstract.com/current/2025ATPDallas.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0421%2FMunich",
    "https://www.tennisabstract.com/current/2025ATPMunich.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0316%2FHamburg",
    "https://www.tennisabstract.com/current/2025ATPHamburg.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-9163%2FLondon-Queens",
    "https://www.tennisabstract.com/current/2025ATPQueens.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0500%2FHalle",
    "https://www.tennisabstract.com/current/2025ATPHalle.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0341%2FWashington",
    "https://www.tennisabstract.com/current/2025ATPWashington.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0329%2FTokyo",
    "https://www.tennisabstract.com/current/2025ATPTokyo.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0747%2FBeijing",
    "https://www.tennisabstract.com/current/2025ATPBeijing.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0337%2FVienna",
    "https://www.tennisabstract.com/current/2025ATPVienna.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0328%2FBasel",
    "https://www.tennisabstract.com/current/2025ATPBasel.html",
    
    # ATP 250 (both formats)
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0301%2FBrisbane",
    "https://www.tennisabstract.com/current/2025ATPBrisbane.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0341%2FHong-Kong",
    "https://www.tennisabstract.com/current/2025ATPHongKong.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0891%2FAdelaide",
    "https://www.tennisabstract.com/current/2025ATPAdelaide.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0339%2FAuckland",
    "https://www.tennisabstract.com/current/2025ATPAuckland.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0499%2FMontpellier",
    "https://www.tennisabstract.com/current/2025ATPMontpellier.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0425%2FMarseille",
    "https://www.tennisabstract.com/current/2025ATPMarseille.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0424%2FDelray-Beach",
    "https://www.tennisabstract.com/current/2025ATPDelrayBeach.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0495%2FBuenos-Aires",
    "https://www.tennisabstract.com/current/2025ATPBuenosAires.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0464%2FSantiago",
    "https://www.tennisabstract.com/current/2025ATPSantiago.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0506%2FCordoba",
    "https://www.tennisabstract.com/current/2025ATPCordoba.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0410%2FHouston",
    "https://www.tennisabstract.com/current/2025ATPHouston.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0435%2FMarrakech",
    "https://www.tennisabstract.com/current/2025ATPMarrakech.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0360%2FGeneva",
    "https://www.tennisabstract.com/current/2025ATPGeneva.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0321%2FStuttgart",
    "https://www.tennisabstract.com/current/2025ATPStuttgart.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0441%2Fs-Hertogenbosch",
    "https://www.tennisabstract.com/current/2025ATPsHertogenbosch.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0495%2FMallorca",
    "https://www.tennisabstract.com/current/2025ATPMallorca.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0741%2FEastbourne",
    "https://www.tennisabstract.com/current/2025ATPEastbourne.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0315%2FLos-Cabos",
    "https://www.tennisabstract.com/current/2025ATPLosCabos.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0329%2FGstaad",
    "https://www.tennisabstract.com/current/2025ATPGstaad.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0316%2FBastad",
    "https://www.tennisabstract.com/current/2025ATPBastad.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0319%2FKitzbuhel",
    "https://www.tennisabstract.com/current/2025ATPKitzbuhel.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0439%2FUmag",
    "https://www.tennisabstract.com/current/2025ATPUmag.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0424%2FWinston-Salem",
    "https://www.tennisabstract.com/current/2025ATPWinstonSalem.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0891%2FChengdu",
    "https://www.tennisabstract.com/current/2025ATPChengdu.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-6932%2FAlmaty",
    "https://www.tennisabstract.com/current/2025ATPAlmaty.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0429%2FStockholm",
    "https://www.tennisabstract.com/current/2025ATPStockholm.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0419%2FAntwerp",  # European Open
    "https://www.tennisabstract.com/current/2025ATPAntwerp.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0438%2FBelgrade",
    "https://www.tennisabstract.com/current/2025ATPBelgrade.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0605%2FMetz",
    "https://www.tennisabstract.com/current/2025ATPMetz.html",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0505%2FPune",
    "https://www.tennisabstract.com/current/2025ATPPune.html",
    
    # Special Events
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-9210%2FLaver-Cup",
    "https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-9000%2FATP-Finals",
]


def discover_from_index(year: int) -> List[str]:
    """
    Discover tournament URLs from TennisAbstract index/listing pages.
    
    Args:
        year: Year to discover tournaments for
        
    Returns:
        List of tournament page URLs
    """
    session = get_session()
    tournament_urls = set()
    
    # Try various index endpoints
    index_patterns = [
        f"{BASE_URL}/cgi-bin/index.cgi",  # Main index
        f"{BASE_URL}/tournaments/",        # Tournament directory
        f"{BASE_URL}/current/",            # Current tournaments
    ]
    
    for index_url in index_patterns:
        logger.info(f"Trying index: {index_url}")
        html = session.get_html(index_url)
        
        if not html:
            continue
            
        # Look for tournament links matching pattern: /cgi-bin/tourney.cgi?t=YYYY-XXXX...
        pattern = re.compile(rf'/cgi-bin/tourney\.cgi\?t={year}-[^"\s&]+', re.IGNORECASE)
        
        for link in html.find_all('a', href=True):
            href = link.get('href', '')
            match = pattern.search(href)
            if match:
                full_url = urljoin(BASE_URL, match.group(0))
                # Filter for ATP (not WTA, not challengers unless desired)
                if 'ATP' in full_url.upper() or re.search(r'\d{4}-\d{4}', full_url):
                    tournament_urls.add(full_url)
        
        if tournament_urls:
            logger.info(f"Found {len(tournament_urls)} tournaments from {index_url}")
            break
    
    return sorted(list(tournament_urls))


def get_seed_list(year: int) -> List[str]:
    """
    Get seed list of known tournament URLs for a given year.
    
    Args:
        year: Year
        
    Returns:
        List of seed URLs
    """
    if year == 2025:
        return SEED_ATP_2025_TOURNAMENTS.copy()
    else:
        logger.warning(f"No seed list available for year {year}")
        return []


def discover_from_base_dataset(base_csv_path: str, year: int) -> List[str]:
    """
    Discover tournament URLs by searching TA for tournaments in base dataset.
    
    APPROACH:
    - Extract unique tournament names from base dataset for the year
    - For each tournament, attempt to find the TA page by searching or constructing URL
    - Validate by fetching and checking for tournament content
    
    Args:
        base_csv_path: Path to base dataset CSV
        year: Year to search for
        
    Returns:
        List of tournament page URLs
    """
    import pandas as pd
    
    logger.info(f"Loading base dataset from {base_csv_path} to discover tournaments")
    df = pd.read_csv(base_csv_path)
    
    # Filter to year
    if 'tourney_date' in df.columns:
        df['year'] = df['tourney_date'].astype(str).str[:4]
        df = df[df['year'] == str(year)]
    
    # Get unique tournaments
    if 'tourney_name' not in df.columns:
        logger.error("Base dataset missing 'tourney_name' column")
        return []
    
    unique_tourneys = df['tourney_name'].dropna().unique()
    logger.info(f"Found {len(unique_tourneys)} unique tournament names in base dataset")
    
    # Try to find TA pages for these tournaments
    tournament_urls = set()
    session = get_session()
    
    for tourney_name in unique_tourneys[:10]:  # Limit to avoid too many requests
        # Construct potential TA URL patterns
        name_slug = re.sub(r'[^a-zA-Z0-9]+', '-', tourney_name).strip('-')
        
        # Try searching TA
        search_url = f"{BASE_URL}/cgi-bin/index.cgi"
        html = session.get_html(search_url)
        
        if html:
            # Look for links containing the tournament name
            for link in html.find_all('a', href=True):
                link_text = link.get_text().lower()
                href = link.get('href', '')
                
                if tourney_name.lower() in link_text and f'{year}' in href:
                    full_url = urljoin(BASE_URL, href)
                    tournament_urls.add(full_url)
                    break
    
    logger.info(f"Found {len(tournament_urls)} tournaments from base dataset search")
    return sorted(list(tournament_urls))


def discover_tournaments(year: int, base_csv_path: str = None) -> List[str]:
    """
    Discover tournament URLs using multi-method approach with fallbacks.
    
    PRIORITY:
    1. Seed list (COMPLETE list of all ATP tournaments - highest priority)
    2. Index-based discovery (if available)
    3. Base dataset search (if base_csv provided)
    
    Args:
        year: Year to discover tournaments for
        base_csv_path: Optional path to base dataset for supplemental discovery
        
    Returns:
        List of tournament page URLs
    """
    logger.info(f"Discovering tournaments for year {year}")
    all_urls = set()
    
    # Method 1: Seed list (ALWAYS use for 2025 - it's complete!)
    seed_urls = get_seed_list(year)
    if seed_urls:
        logger.info(f"Using COMPLETE seed list: {len(seed_urls)} tournaments")
        all_urls.update(seed_urls)
    
    # Method 2: Index-based discovery (supplement)
    try:
        urls = discover_from_index(year)
        if urls:
            logger.info(f"Index discovery found {len(urls)} additional tournaments")
            all_urls.update(urls)
    except Exception as e:
        logger.warning(f"Index discovery failed: {e}")
    
    # Method 3: Base dataset search (optional supplement)
    if base_csv_path and Path(base_csv_path).exists():
        try:
            logger.info("Supplementing with base dataset tournament search")
            base_urls = discover_from_base_dataset(base_csv_path, year)
            if base_urls:
                all_urls.update(base_urls)
        except Exception as e:
            logger.warning(f"Base dataset discovery failed: {e}")
    
    final_urls = sorted(list(all_urls))
    logger.info(f"Total discovered: {len(final_urls)} tournament URLs")
    
    if not final_urls:
        logger.error(f"No tournaments discovered for {year}. Pipeline cannot proceed.")
    
    return final_urls


def save_tournament_urls(urls: List[str], output_path: str, year: int):
    """Save tournament URLs to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'year': year,
        'tournament_urls': urls,
        'count': len(urls),
        'discovery_timestamp': datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(urls)} tournament URLs to {output_path}")


def load_tournament_urls(input_path: str) -> List[str]:
    """Load tournament URLs from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    return data.get('tournament_urls', [])
