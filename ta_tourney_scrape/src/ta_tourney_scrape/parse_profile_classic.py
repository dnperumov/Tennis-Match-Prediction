"""
Parse player-classic profile pages to extract match data.
The classic player pages load match data dynamically via JavaScript.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError, Page
from unidecode import unidecode

logger = logging.getLogger(__name__)

def _parse_pct(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        return float(s) / 100.0
    except ValueError:
        return None


def _parse_ratio(s: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    s = (s or "").strip()
    if not s or "/" not in s:
        return None, None, None
    a, b = s.split("/", 1)
    try:
        num = int(a.strip())
        den = int(b.strip())
    except ValueError:
        return None, None, None
    if den == 0:
        return num, den, None
    return num, den, num / den


def _parse_time_to_minutes(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    # "3:04" -> 184 minutes
    if ":" in s:
        h, m = s.split(":", 1)
        try:
            return int(h) * 60 + int(m)
        except ValueError:
            return None
    try:
        return int(s)
    except ValueError:
        return None


def construct_classic_profile_url(player_name: str) -> str:
    """
    Construct TennisAbstract player-classic profile URL from player name.
    
    Args:
        player_name: Player name (e.g., "Jannik Sinner")
        
    Returns:
        Player classic profile URL
    """
    # Convert name to TA format: FirstnameLastname (no spaces; ASCII; strip punctuation)
    cleaned = unidecode(player_name or "").strip()
    cleaned = re.sub(r"[’'`.-]", " ", cleaned)  # normalize common punctuation to spaces
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", cleaned)
    name_parts = [p for p in cleaned.split() if p]
    if not name_parts:
        ta_name = ""
    elif len(name_parts) >= 2:
        ta_name = ''.join([part[:1].upper() + part[1:] for part in name_parts])
    else:
        ta_name = name_parts[0][:1].upper() + name_parts[0][1:]
    
    return f"https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p={ta_name}"


def scrape_player_classic_matches(player_name: str, year: int, tournament_filter: str = None) -> List[Dict[str, Any]]:
    """
    Scrape matches from a player's classic profile page using browser automation.
    
    Args:
        player_name: Player name
        year: Year to filter matches
        tournament_filter: Optional tournament name to filter (e.g., "Wimbledon")
        
    Returns:
        List of match records
    """
    url = construct_classic_profile_url(player_name)
    logger.info(f"Scraping classic profile for {player_name}: {url}")
    
    matches = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            matches = _scrape_player_with_page(page, player_name=player_name, year=year, tournament_filter=tournament_filter)
            
        except PlaywrightTimeoutError:
            logger.warning(f"Timeout loading classic profile for {player_name}")
        except Exception as e:
            logger.error(f"Error scraping classic profile for {player_name}: {e}")
        finally:
            browser.close()
    
    logger.info(f"Scraped {len(matches)} matches from classic profile for {player_name}")
    return matches


def _scrape_player_with_page(page: Page, player_name: str, year: int, tournament_filter: Optional[str]) -> List[Dict[str, Any]]:
    url = construct_classic_profile_url(player_name)
    # Navigate to the page
    resp = page.goto(url, wait_until='domcontentloaded', timeout=60000)
    if resp is not None and resp.status >= 400:
        return []
    # Make sure navigation actually swapped the document (important when reusing a Page)
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except PlaywrightTimeoutError:
        # still proceed; some pages keep long-polling
        pass

    expected_last = unidecode(player_name).split()[-1].lower() if player_name else ""
    if expected_last:
        try:
            page.wait_for_function(
                "exp => document.title.toLowerCase().includes(exp)",
                arg=expected_last,
                timeout=10000,
            )
        except PlaywrightTimeoutError:
            # proceed; title check is best-effort
            pass

    try:
        # Some profiles load slower; keep this moderately patient.
        page.wait_for_selector('table#matches', timeout=15000)
    except PlaywrightTimeoutError:
        return []

    # Wait until at least one data row is present (JS-populated) AFTER navigation
    try:
        page.wait_for_selector('table#matches tr:nth-child(2) td', timeout=15000)
    except PlaywrightTimeoutError:
        return []
    page.wait_for_timeout(250)

    table = page.query_selector('table#matches')
    if not table:
        logger.warning(f"No matches table found for {player_name}")
        return []

    rows = table.query_selector_all('tr')
    logger.debug(f"Found {len(rows)} rows in matches table for {player_name}")

    matches: List[Dict[str, Any]] = []
    for row in rows[1:]:
        cells = row.query_selector_all('td')
        if len(cells) < 8:
            continue

        try:
            date_text = cells[0].inner_text().strip()
            if not date_text:
                continue

            event_text = cells[1].inner_text().strip()
            surface_text = cells[2].inner_text().strip()
            round_text = cells[3].inner_text().strip()
            rk_text = cells[4].inner_text().strip() if len(cells) > 4 else ""
            vrk_text = cells[5].inner_text().strip() if len(cells) > 5 else ""
            result_text = cells[6].inner_text().strip() if len(cells) > 6 else ''
            score_text = cells[7].inner_text().strip() if len(cells) > 7 else ''
            more_text = cells[8].inner_text().strip() if len(cells) > 8 else ''
            dr_text = cells[9].inner_text().strip() if len(cells) > 9 else ''
            a_pct_text = cells[10].inner_text().strip() if len(cells) > 10 else ''
            df_pct_text = cells[11].inner_text().strip() if len(cells) > 11 else ''
            first_in_text = cells[12].inner_text().strip() if len(cells) > 12 else ''
            first_won_text = cells[13].inner_text().strip() if len(cells) > 13 else ''
            second_won_text = cells[14].inner_text().strip() if len(cells) > 14 else ''
            bpsvd_text = cells[15].inner_text().strip() if len(cells) > 15 else ''
            time_text = cells[16].inner_text().strip() if len(cells) > 16 else ''

            result_text = result_text.replace('\xa0', ' ')

            if tournament_filter and tournament_filter.lower() not in event_text.lower():
                continue

            # Date format: "30‑Jun‑2025"
            try:
                match_date = datetime.strptime(date_text, '%d‑%b‑%Y')
            except ValueError:
                try:
                    match_date = datetime.strptime(date_text, '%d-%b-%Y')
                except ValueError:
                    continue
            if match_date.year != year:
                continue

            winner_name = ''
            loser_name = ''
            if ' d. ' in result_text:
                winner_part, loser_part = result_text.split(' d. ', 1)
                winner_name = re.sub(r'^\([A-Z0-9]+\)\s*', '', winner_part.strip())
                winner_name = re.sub(r'\s*\[.*?\]\s*$', '', winner_name).strip()
                loser_name = re.sub(r'^\([A-Z0-9]+\)\s*', '', loser_part.strip())
                loser_name = re.sub(r'\s*\[.*?\]\s*$', '', loser_name).strip()

            player_last = player_name.split()[-1].lower()
            player_won = bool(winner_name and player_last in winner_name.lower())

            if winner_name and winner_name.lower() == player_last:
                winner_name = player_name
            if loser_name and loser_name.lower() == player_last:
                loser_name = player_name

            ta_dr = None
            try:
                ta_dr = float((dr_text or "").strip()) if (dr_text or "").strip() else None
            except ValueError:
                ta_dr = None

            bpsvd_num, bpsvd_den, bpsvd_pct = _parse_ratio(bpsvd_text)

            def _to_int(s: str) -> Optional[int]:
                s = (s or "").strip()
                if not s:
                    return None
                try:
                    return int(s)
                except ValueError:
                    return None

            player_rank = _to_int(rk_text)
            opp_rank = _to_int(vrk_text)

            matches.append(
                {
                    'tourney_name': event_text,
                    'surface': surface_text,
                    'round': round_text,
                    'match_date': match_date.strftime('%Y-%m-%d'),
                    'year': year,
                    'winner_name': winner_name,
                    'loser_name': loser_name,
                    'score': score_text,
                    'player_won': player_won,
                    'player_rank': player_rank,
                    'opponent_rank': opp_rank,
                    'source_player': player_name,
                    'source_url': url,
                    'ta_more': more_text,
                    'ta_dr': ta_dr,
                    'ta_ace_pct': _parse_pct(a_pct_text),
                    'ta_df_pct': _parse_pct(df_pct_text),
                    'ta_1stin_pct': _parse_pct(first_in_text),
                    'ta_1stwon_pct': _parse_pct(first_won_text),
                    'ta_2ndwon_pct': _parse_pct(second_won_text),
                    'ta_bpsvd_num': bpsvd_num,
                    'ta_bpsvd_den': bpsvd_den,
                    'ta_bpsvd_pct': bpsvd_pct,
                    'minutes': _parse_time_to_minutes(time_text),
                }
            )
        except Exception:
            continue

    return matches


def scrape_players_classic_matches(
    player_names: List[str],
    year: int,
    tournament_filter: Optional[str] = None,
    sleep_seconds: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Bulk scrape classic player pages while reusing a single browser instance.
    This is dramatically faster than launching a browser per player.
    """
    all_matches: List[Dict[str, Any]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            total = len(player_names)
            for idx, name in enumerate(player_names, start=1):
                page = browser.new_page()
                try:
                    all_matches.extend(_scrape_player_with_page(page, player_name=name, year=year, tournament_filter=tournament_filter))
                except PlaywrightTimeoutError:
                    logger.warning(f"Timeout loading classic profile for {name}")
                except Exception:
                    logger.exception(f"Error scraping classic profile for {name}")
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass
                if idx % 25 == 0 or idx == total:
                    print(f"… scraped {idx}/{total} players; raw matches so far: {len(all_matches)}", flush=True)
                if sleep_seconds:
                    time.sleep(sleep_seconds)
        finally:
            browser.close()
    return all_matches

