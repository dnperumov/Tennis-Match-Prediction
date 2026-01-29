"""
Command-line interface for the scraper.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm

from .discover import discover_tournaments
from .scrape_tournament import scrape_all_tournaments
from .id_map import PlayerIDMapper
from .normalize import normalize_all_records
from .write_csv import write_csv

# Global column list - EXACT order from Jeff Sackmann's atp_matches format
COLUMN_LIST = [
    'tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level', 'tourney_date',
    'match_num', 'winner_id', 'winner_seed', 'winner_entry', 'winner_name', 'winner_hand',
    'winner_ht', 'winner_ioc', 'winner_age', 'loser_id', 'loser_seed', 'loser_entry',
    'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 'score', 'best_of',
    'round', 'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
    'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon',
    'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'winner_rank', 'winner_rank_points',
    'loser_rank', 'loser_rank_points'
]

# Column list will be loaded from configuration
# See load_column_list() function below

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scrape_command(args):
    """Scrape tournaments and save raw JSONL."""
    year = args.year
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = output_dir / f"tennisabstract_{year}_matches.jsonl"
    
    logger.info(f"Starting scrape for year {year}")
    
    # Discover tournaments
    tournament_urls = discover_tournaments(year=year)
    
    if not tournament_urls:
        logger.error("No tournaments discovered")
        return
    
    # Scrape all tournaments
    all_matches = scrape_all_tournaments(tournament_urls)
    
    # Write to JSONL
    logger.info(f"Writing {len(all_matches)} matches to {jsonl_path}")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for match in tqdm(all_matches, desc="Writing JSONL"):
            f.write(json.dumps(match, ensure_ascii=False) + '\n')
    
    logger.info(f"Scraping complete. Saved to {jsonl_path}")


def load_column_list(column_list_path: str = None) -> List[str]:
    """
    Load column list from file or use default.
    
    Args:
        column_list_path: Optional path to column list file
        
    Returns:
        List of column names in order
    """
    global COLUMN_LIST
    
    if column_list_path and Path(column_list_path).exists():
        with open(column_list_path, 'r') as f:
            COLUMN_LIST = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(COLUMN_LIST)} columns from {column_list_path}")
        return COLUMN_LIST
    
    if not COLUMN_LIST:
        logger.error("Column list not configured!")
        logger.error("Please provide column list via:")
        logger.error("  1. Set COLUMN_LIST in cli.py, OR")
        logger.error("  2. Provide --column-list-file path")
        raise ValueError("Column list must be configured")
    
    return COLUMN_LIST


def build_csv_command(args):
    """Build CSV from raw JSONL."""
    year = args.year
    data_dir = Path(args.data_dir)
    
    jsonl_path = data_dir / "raw" / f"tennisabstract_{year}_matches.jsonl"
    csv_path = data_dir / "processed" / f"atp_matches_{year}.csv"
    players_csv = data_dir / "sackmann" / "atp_players.csv"
    overrides_csv = Path(args.mappings_dir) / "name_overrides.csv"
    missing_players_csv = data_dir / "raw" / "missing_player_ids.csv"
    
    if not jsonl_path.exists():
        logger.error(f"JSONL file not found: {jsonl_path}")
        logger.info("Run 'scrape' command first")
        return
    
    # Load column list
    column_list = load_column_list(getattr(args, 'column_list_file', None))
    
    # Load raw records
    logger.info(f"Loading records from {jsonl_path}")
    raw_records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))
    
    logger.info(f"Loaded {len(raw_records)} raw records")
    
    # Initialize player mapper
    player_mapper = PlayerIDMapper(
        players_csv_path=str(players_csv),
        overrides_csv_path=str(overrides_csv) if overrides_csv.exists() else None
    )
    
    # Normalize records
    normalized_records = normalize_all_records(
        raw_records, player_mapper, column_list
    )
    
    # Save missing players
    if player_mapper.get_missing_players():
        player_mapper.save_missing_players(str(missing_players_csv))
        logger.warning(f"Found {len(player_mapper.get_missing_players())} missing player IDs")
    
    # Write CSV
    write_csv(normalized_records, str(csv_path), column_list)
    
    logger.info(f"CSV build complete. Output: {csv_path}")


def run_command(args):
    """Run complete pipeline: discover -> scrape -> normalize -> CSV."""
    logger.info("Running complete pipeline...")
    
    # Step 1: Discover and scrape
    scrape_command(args)
    
    # Step 2: Build CSV
    build_csv_command(args)
    
    logger.info("Pipeline complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TennisAbstract 2025 ATP Match Scraper"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Scrape tournaments')
    scrape_parser.add_argument('--year', type=int, default=2025)
    scrape_parser.add_argument('--output-dir', type=str, default='data/raw')
    scrape_parser.set_defaults(func=scrape_command)
    
    # Build CSV command
    build_parser = subparsers.add_parser('build-csv', help='Build CSV from JSONL')
    build_parser.add_argument('--year', type=int, default=2025)
    build_parser.add_argument('--data-dir', type=str, default='data')
    build_parser.add_argument('--mappings-dir', type=str, default='mappings')
    build_parser.add_argument('--column-list-file', type=str, default=None,
                             help='Path to file with column list (one per line)')
    build_parser.set_defaults(func=build_csv_command)
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run complete pipeline')
    run_parser.add_argument('--year', type=int, default=2025)
    run_parser.add_argument('--output-dir', type=str, default='data/raw')
    run_parser.add_argument('--data-dir', type=str, default='data')
    run_parser.add_argument('--mappings-dir', type=str, default='mappings')
    run_parser.add_argument('--column-list-file', type=str, default=None,
                           help='Path to file with column list (one per line)')
    run_parser.set_defaults(func=run_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()

