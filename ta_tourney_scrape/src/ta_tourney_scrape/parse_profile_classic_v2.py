"""
Enhanced parser for player-classic profile pages - supports BOTH serve and return stats.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Literal
from datetime import datetime
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError, Page
from unidecode import unidecode

logger = logging.getLogger(__name__)

StatMode = Literal["serve", "return"]


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


def _to_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _to_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def construct_classic_profile_url(player_name: str, stat_mode: StatMode = "serve") -> str:
    """
    Construct TennisAbstract player-classic profile URL.
    
    stat_mode:
      - "serve": default view (or &f=s1)
      - "return": &f=r1
    """
    cleaned = unidecode(player_name or "").strip()
    cleaned = re.sub(r"[''`.-]", " ", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", cleaned)
    name_parts = [p for p in cleaned.split() if p]
    if not name_parts:
        ta_name = ""
    elif len(name_parts) >= 2:
        ta_name = ''.join([part[:1].upper() + part[1:] for part in name_parts])
    else:
        ta_name = name_parts[0][:1].upper() + name_parts[0][1:]
    
    base = f"https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p={ta_name}"
    if stat_mode == "return":
        base += "&f=r1"
    return base


def _parse_serve_stats(cells: List[Any]) -> Dict[str, Any]:
    """Parse serve stat columns: DR, A%, DF%, 1stIn, 1st%, 2nd%, BPSvd, Time"""
    dr_text = cells[9].inner_text().strip() if len(cells) > 9 else ''
    a_pct_text = cells[10].inner_text().strip() if len(cells) > 10 else ''
    df_pct_text = cells[11].inner_text().strip() if len(cells) > 11 else ''
    first_in_text = cells[12].inner_text().strip() if len(cells) > 12 else ''
    first_won_text = cells[13].inner_text().strip() if len(cells) > 13 else ''
    second_won_text = cells[14].inner_text().strip() if len(cells) > 14 else ''
    bpsvd_text = cells[15].inner_text().strip() if len(cells) > 15 else ''
    time_text = cells[16].inner_text().strip() if len(cells) > 16 else ''
    
    bpsvd_num, bpsvd_den, bpsvd_pct = _parse_ratio(bpsvd_text)
    
    return {
        "dr": _to_float(dr_text),
        "ace_pct": _parse_pct(a_pct_text),
        "df_pct": _parse_pct(df_pct_text),
        "first_in_pct": _parse_pct(first_in_text),
        "first_won_pct": _parse_pct(first_won_text),
        "second_won_pct": _parse_pct(second_won_text),
        "bpsvd_num": bpsvd_num,
        "bpsvd_den": bpsvd_den,
        "bpsvd_pct": bpsvd_pct,
        "minutes": _parse_time_to_minutes(time_text),
    }


def _parse_return_stats(cells: List[Any]) -> Dict[str, Any]:
    """Parse return stat columns: DR, TPW, RPW, vA%, v1st%, v2nd%, BPCnv, Time"""
    dr_text = cells[9].inner_text().strip() if len(cells) > 9 else ''
    tpw_text = cells[10].inner_text().strip() if len(cells) > 10 else ''
    rpw_text = cells[11].inner_text().strip() if len(cells) > 11 else ''
    va_pct_text = cells[12].inner_text().strip() if len(cells) > 12 else ''
    v1st_pct_text = cells[13].inner_text().strip() if len(cells) > 13 else ''
    v2nd_pct_text = cells[14].inner_text().strip() if len(cells) > 14 else ''
    bpcnv_text = cells[15].inner_text().strip() if len(cells) > 15 else ''
    time_text = cells[16].inner_text().strip() if len(cells) > 16 else ''
    
    bpcnv_num, bpcnv_den, bpcnv_pct = _parse_ratio(bpcnv_text)
    
    return {
        "dr": _to_float(dr_text),
        "tpw_pct": _parse_pct(tpw_text),
        "rpw_pct": _parse_pct(rpw_text),
        "v_ace_pct": _parse_pct(va_pct_text),
        "v_first_won_pct": _parse_pct(v1st_pct_text),
        "v_second_won_pct": _parse_pct(v2nd_pct_text),
        "bpcnv_num": bpcnv_num,
        "bpcnv_den": bpcnv_den,
        "bpcnv_pct": bpcnv_pct,
        "minutes": _parse_time_to_minutes(time_text),
    }


def _scrape_player_with_page(
    page: Page,
    player_name: str,
    year: int,
    tournament_filter: Optional[str],
    stat_mode: StatMode = "serve",
) -> List[Dict[str, Any]]:
    url = construct_classic_profile_url(player_name, stat_mode=stat_mode)
    
    resp = page.goto(url, wait_until='domcontentloaded', timeout=60000)
    if resp is not None and resp.status >= 400:
        logger.warning(f"HTTP {resp.status} for {player_name} ({stat_mode})")
        return []
    
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except PlaywrightTimeoutError:
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
            pass

    try:
        page.wait_for_selector('table#matches', timeout=15000)
    except PlaywrightTimeoutError:
        return []

    try:
        page.wait_for_selector('table#matches tr:nth-child(2) td', timeout=15000)
    except PlaywrightTimeoutError:
        return []
    page.wait_for_timeout(250)

    table = page.query_selector('table#matches')
    if not table:
        return []

    rows = table.query_selector_all('tr')
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

            result_text = result_text.replace('\xa0', ' ')

            if tournament_filter and tournament_filter.lower() not in event_text.lower():
                continue

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

            # Parse stats based on mode
            if stat_mode == "serve":
                stats = _parse_serve_stats(cells)
            else:
                stats = _parse_return_stats(cells)

            matches.append({
                'tourney_name': event_text,
                'surface': surface_text,
                'round': round_text,
                'match_date': match_date.strftime('%Y-%m-%d'),
                'year': year,
                'winner_name': winner_name,
                'loser_name': loser_name,
                'score': score_text,
                'player_won': player_won,
                'player_rank': _to_int(rk_text),
                'opponent_rank': _to_int(vrk_text),
                'source_player': player_name,
                'source_url': url,
                'stat_mode': stat_mode,
                **stats,
            })
        except Exception:
            continue

    return matches


def scrape_player_both_stats(
    player_name: str,
    year: int,
    tournament_filter: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Scrape BOTH serve and return stats for a player.
    Returns {"serve": [...], "return": [...]}
    """
    result = {"serve": [], "return": []}
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            for mode in ["serve", "return"]:
                page = browser.new_page()
                try:
                    matches = _scrape_player_with_page(
                        page, player_name, year, tournament_filter, stat_mode=mode
                    )
                    result[mode] = matches
                except Exception as e:
                    logger.error(f"Error scraping {mode} for {player_name}: {e}")
                finally:
                    page.close()
                time.sleep(0.5)  # Be polite between requests
        finally:
            browser.close()
    
    return result


def scrape_players_both_stats(
    player_names: List[str],
    year: int,
    tournament_filter: Optional[str] = None,
    sleep_seconds: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Bulk scrape serve + return stats for multiple players.
    Returns merged match records with both serve and return stats.
    """
    # Collect all matches keyed by signature
    serve_by_sig: Dict[Tuple, Dict] = {}
    return_by_sig: Dict[Tuple, Dict] = {}
    
    def sig(m: Dict) -> Tuple:
        return (
            m.get('tourney_name', ''),
            m.get('round', ''),
            m.get('winner_name', ''),
            m.get('loser_name', ''),
            m.get('score', ''),
        )
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            total = len(player_names)
            for idx, name in enumerate(player_names, start=1):
                for mode in ["serve", "return"]:
                    page = browser.new_page()
                    try:
                        matches = _scrape_player_with_page(
                            page, name, year, tournament_filter, stat_mode=mode
                        )
                        target = serve_by_sig if mode == "serve" else return_by_sig
                        for m in matches:
                            target[sig(m)] = m
                    except Exception as e:
                        logger.error(f"Error scraping {mode} for {name}: {e}")
                    finally:
                        page.close()
                    time.sleep(sleep_seconds)
                
                if idx % 10 == 0 or idx == total:
                    print(f"… scraped {idx}/{total} players", flush=True)
        finally:
            browser.close()
    
    # Merge serve and return by signature
    all_sigs = set(serve_by_sig.keys()) | set(return_by_sig.keys())
    merged = []
    
    for s in all_sigs:
        serve_m = serve_by_sig.get(s, {})
        return_m = return_by_sig.get(s, {})
        
        base = serve_m or return_m
        record = {
            'tourney_name': base.get('tourney_name'),
            'surface': base.get('surface'),
            'round': base.get('round'),
            'match_date': base.get('match_date'),
            'year': base.get('year'),
            'winner_name': base.get('winner_name'),
            'loser_name': base.get('loser_name'),
            'score': base.get('score'),
            'player_won': base.get('player_won'),
            'player_rank': base.get('player_rank'),
            'opponent_rank': base.get('opponent_rank'),
            'source_player': base.get('source_player'),
        }
        
        # Add serve stats (prefixed)
        if serve_m:
            record['serve_dr'] = serve_m.get('dr')
            record['serve_ace_pct'] = serve_m.get('ace_pct')
            record['serve_df_pct'] = serve_m.get('df_pct')
            record['serve_first_in_pct'] = serve_m.get('first_in_pct')
            record['serve_first_won_pct'] = serve_m.get('first_won_pct')
            record['serve_second_won_pct'] = serve_m.get('second_won_pct')
            record['serve_bpsvd_num'] = serve_m.get('bpsvd_num')
            record['serve_bpsvd_den'] = serve_m.get('bpsvd_den')
            record['serve_bpsvd_pct'] = serve_m.get('bpsvd_pct')
            record['minutes'] = serve_m.get('minutes')
        
        # Add return stats (prefixed)
        if return_m:
            record['return_dr'] = return_m.get('dr')
            record['return_tpw_pct'] = return_m.get('tpw_pct')
            record['return_rpw_pct'] = return_m.get('rpw_pct')
            record['return_v_ace_pct'] = return_m.get('v_ace_pct')
            record['return_v_first_won_pct'] = return_m.get('v_first_won_pct')
            record['return_v_second_won_pct'] = return_m.get('v_second_won_pct')
            record['return_bpcnv_num'] = return_m.get('bpcnv_num')
            record['return_bpcnv_den'] = return_m.get('bpcnv_den')
            record['return_bpcnv_pct'] = return_m.get('bpcnv_pct')
            if not record.get('minutes'):
                record['minutes'] = return_m.get('minutes')
        
        merged.append(record)
    
    return merged


if __name__ == "__main__":
    # Quick test
    import json
    result = scrape_player_both_stats("Taylor Fritz", 2025, tournament_filter="Australian Open")
    print(f"Serve matches: {len(result['serve'])}")
    print(f"Return matches: {len(result['return'])}")
    if result['serve']:
        print("Sample serve:", json.dumps(result['serve'][0], indent=2, default=str))
    if result['return']:
        print("Sample return:", json.dumps(result['return'][0], indent=2, default=str))
