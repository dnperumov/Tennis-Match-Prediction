"""
CLI for TennisAbstract ATP scraping pipeline.

COMMANDS:
- discover: Find tournament URLs for a year
- scrape_tournaments: Scrape tournament pages (detect if stats present)
- scrape_profiles: Enrich matches lacking stats via player profiles (fallback)
- build: Join to base dataset, compute metrics, generate report
"""

import argparse
import logging
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def discover_command(args):
    """
    Discover tournament URLs for a given year.
    
    Saves to: data/intermediate/ta_tournaments_{year}.json
    """
    from .discover import discover_tournaments, save_tournament_urls
    
    year = args.year
    base_csv = args.base_csv if hasattr(args, 'base_csv') else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Discovering tournaments for year {year}")
    
    urls = discover_tournaments(year, base_csv)
    
    if not urls:
        logger.error(f"No tournaments discovered for {year}")
        return
    
    output_file = output_dir / f"ta_tournaments_{year}.json"
    save_tournament_urls(urls, str(output_file), year)
    
    logger.info(f"Discovered {len(urls)} tournaments, saved to {output_file}")


def scrape_tournaments_command(args):
    """
    Scrape tournament pages for match listings and stats (if available).
    Uses browser automation for JavaScript-loaded pages (/current/ URLs).
    
    Reads: data/intermediate/ta_tournaments_{year}.json
    Saves: data/intermediate/ta_tournament_matches_{year}.parquet
    """
    from .discover import load_tournament_urls
    from .parse_tourney import scrape_tournament
    from .parse_js_tourney import scrape_tournament_with_browser, is_js_tournament_url
    
    year = args.year
    intermediate_dir = Path(args.output_dir)
    
    tournaments_file = intermediate_dir / f"ta_tournaments_{year}.json"
    if not tournaments_file.exists():
        logger.error(f"Tournaments file not found: {tournaments_file}")
        logger.error("Run 'discover' command first")
        return
    
    urls = load_tournament_urls(str(tournaments_file))
    logger.info(f"Scraping {len(urls)} tournaments (hybrid: HTML + browser automation)...")
    
    # Separate JS-loaded from regular HTML pages
    js_urls = [url for url in urls if is_js_tournament_url(url)]
    html_urls = [url for url in urls if not is_js_tournament_url(url)]
    
    logger.info(f"  - {len(html_urls)} HTML pages (fast scraping)")
    logger.info(f"  - {len(js_urls)} JavaScript pages (browser automation)")
    
    all_matches = []
    tournaments_needing_profiles = []
    
    # Scrape HTML pages first (faster)
    for url in tqdm(html_urls, desc="Scraping HTML pages"):
        try:
            metadata, matches = scrape_tournament(url)
            
            # Track if this tournament needs profile fallback
            if not metadata.get('tourney_has_match_stats'):
                tournaments_needing_profiles.append({
                    'url': url,
                    'tourney_name': metadata.get('tourney_name'),
                    'match_count': len(matches)
                })
            
            all_matches.extend(matches)
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            continue
    
    # Scrape JavaScript pages with browser (slower)
    for url in tqdm(js_urls, desc="Scraping JS pages (browser)"):
        try:
            metadata, matches = scrape_tournament_with_browser(url)
            
            # JS pages don't have per-match stats, always need profile fallback
            if matches:
                tournaments_needing_profiles.append({
                    'url': url,
                    'tourney_name': metadata.get('tourney_name'),
                    'match_count': len(matches)
                })
            
            all_matches.extend(matches)
        except Exception as e:
            logger.error(f"Error browser-scraping {url}: {e}")
            continue
    
    if not all_matches:
        logger.warning("No matches scraped")
        return
    
    # Save matches
    df = pd.DataFrame(all_matches)
    output_file = intermediate_dir / f"ta_tournament_matches_{year}.parquet"
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved {len(all_matches)} matches to {output_file}")
    
    # Save list of tournaments needing profile enrichment
    if tournaments_needing_profiles:
        profile_needed_file = intermediate_dir / f"tournaments_needing_profiles_{year}.json"
        with open(profile_needed_file, 'w') as f:
            json.dump(tournaments_needing_profiles, f, indent=2)
        logger.info(f"{len(tournaments_needing_profiles)} tournaments need profile fallback, saved to {profile_needed_file}")


def scrape_profiles_command(args):
    """
    Scrape player profiles to enrich matches lacking stats.
    
    Reads: 
    - data/intermediate/ta_tournament_matches_{year}.parquet
    - data/intermediate/tournaments_needing_profiles_{year}.json
    Saves: data/intermediate/ta_profile_fallback_hits_{year}.parquet
    """
    from .parse_profile import scrape_player_profile_matches
    from .dedupe import deduplicate_profile_matches
    
    year = args.year
    intermediate_dir = Path(args.output_dir)
    
    # Load tournament matches
    matches_file = intermediate_dir / f"ta_tournament_matches_{year}.parquet"
    if not matches_file.exists():
        logger.error(f"Tournament matches file not found: {matches_file}")
        logger.error("Run 'scrape_tournaments' command first")
        return
    
    df = pd.read_parquet(matches_file)
    
    # Filter to matches without stats
    matches_without_stats = df[df['has_stats'] == False].to_dict('records')
    
    if not matches_without_stats:
        logger.info("All matches already have stats, no profile scraping needed")
        return
    
    logger.info(f"Found {len(matches_without_stats)} matches without stats")
    
    # Get unique players from these matches
    players = set()
    for match in matches_without_stats:
        if match.get('winner_name'):
            players.add(match['winner_name'])
        if match.get('loser_name'):
            players.add(match['loser_name'])
    
    logger.info(f"Scraping profiles for {len(players)} players...")
    
    all_profile_matches = []
    for player in tqdm(list(players), desc="Scraping player profiles"):
        try:
            profile_matches = scrape_player_profile_matches(player, year)
            all_profile_matches.extend(profile_matches)
        except Exception as e:
            logger.warning(f"Error scraping profile for {player}: {e}")
            continue
    
    if not all_profile_matches:
        logger.warning("No matches found in player profiles")
        return
    
    # Deduplicate
    deduped = deduplicate_profile_matches(all_profile_matches, year)
    
    # Save
    df_profile = pd.DataFrame(deduped)
    output_file = intermediate_dir / f"ta_profile_fallback_hits_{year}.parquet"
    df_profile.to_parquet(output_file, index=False)
    logger.info(f"Saved {len(deduped)} profile-enriched matches to {output_file}")


def build_command(args):
    """
    Build final dataset: merge tournament + profile data, join to base, compute metrics, generate report.
    
    Reads:
    - data/intermediate/ta_tournament_matches_{year}.parquet
    - data/intermediate/ta_profile_fallback_hits_{year}.parquet (if exists)
    - base CSV
    Saves:
    - data/atp_matches_{year}_enriched.csv
    - data/atp_matches_{year}_joined_to_base.csv
    - data/ta_reconciliation_report_{year}.md
    """
    from .dedupe import merge_tournament_and_profile_data
    from .match import prepare_base_dataframe, match_ta_to_base
    from .metrics import compute_derived_metrics
    from .export import write_enriched_csv, generate_reconciliation_report
    
    year = args.year
    base_csv = args.base_csv
    intermediate_dir = Path(args.output_dir)
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tournament matches
    tourney_matches_file = intermediate_dir / f"ta_tournament_matches_{year}.parquet"
    if not tourney_matches_file.exists():
        logger.error(f"Tournament matches file not found: {tourney_matches_file}")
        return
    
    tourney_df = pd.read_parquet(tourney_matches_file)
    tournament_matches = tourney_df.to_dict('records')
    logger.info(f"Loaded {len(tournament_matches)} tournament matches")
    
    # Load profile matches if available
    profile_matches_file = intermediate_dir / f"ta_profile_fallback_hits_{year}.parquet"
    profile_matches = []
    if profile_matches_file.exists():
        profile_df = pd.read_parquet(profile_matches_file)
        profile_matches = profile_df.to_dict('records')
        logger.info(f"Loaded {len(profile_matches)} profile-enriched matches")
    
    # Merge tournament + profile data
    merged_matches = merge_tournament_and_profile_data(tournament_matches, profile_matches, year)
    logger.info(f"Merged to {len(merged_matches)} total matches")
    
    # Save enriched dataset (TA data only)
    enriched_df = pd.DataFrame(merged_matches)
    enriched_file = output_dir / f"atp_matches_{year}_enriched.csv"
    enriched_df.to_csv(enriched_file, index=False)
    logger.info(f"Saved enriched dataset to {enriched_file}")
    
    # Load base dataset
    if not Path(base_csv).exists():
        logger.error(f"Base CSV not found: {base_csv}")
        return
    
    base_df = pd.read_csv(base_csv)
    logger.info(f"Loaded {len(base_df)} rows from base dataset")
    
    # Prepare base dataset (normalize, create keys)
    base_df_prep = prepare_base_dataframe(base_df, year)
    logger.info(f"Prepared base dataset: {len(base_df_prep)} rows for year {year}")
    
    # Compute derived metrics from base counts
    base_df_with_metrics = compute_derived_metrics(base_df_prep)
    
    # Match TA to base
    joined_df = match_ta_to_base(enriched_df, base_df_with_metrics)
    
    # Save joined dataset
    joined_file = output_dir / f"atp_matches_{year}_joined_to_base.csv"
    joined_df.to_csv(joined_file, index=False)
    logger.info(f"Saved joined dataset to {joined_file}")
    
    # Generate reconciliation report
    report_file = output_dir / f"ta_reconciliation_report_{year}.md"
    generate_reconciliation_report(joined_df, report_file, year)
    logger.info(f"Generated reconciliation report at {report_file}")
    
    logger.info("Build complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TennisAbstract ATP Scraper and Data Integration Pipeline"
    )
    parser.add_argument('--year', type=int, default=2025, help="Year to process")
    parser.add_argument('--output-dir', type=str, default='data/intermediate', help="Output directory for intermediate files")
    parser.add_argument('--base-csv', type=str, help="Path to base ATP matches CSV")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # discover command
    discover_parser = subparsers.add_parser('discover', help="Discover tournament URLs")
    discover_parser.set_defaults(func=discover_command)
    
    # scrape_tournaments command
    scrape_tourney_parser = subparsers.add_parser('scrape_tournaments', help="Scrape tournament pages")
    scrape_tourney_parser.set_defaults(func=scrape_tournaments_command)
    
    # scrape_profiles command
    scrape_profiles_parser = subparsers.add_parser('scrape_profiles', help="Scrape player profiles for fallback enrichment")
    scrape_profiles_parser.set_defaults(func=scrape_profiles_command)
    
    # build command
    build_parser = subparsers.add_parser('build', help="Build final dataset with joins and reconciliation")
    build_parser.set_defaults(func=build_command)
    
    # all command (run full pipeline)
    all_parser = subparsers.add_parser('all', help="Run complete pipeline")
    all_parser.set_defaults(func=lambda args: (
        discover_command(args),
        scrape_tournaments_command(args),
        scrape_profiles_command(args),
        build_command(args)
    ))
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
