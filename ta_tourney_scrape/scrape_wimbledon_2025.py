"""
Script to scrape Wimbledon 2025 matches from player profiles.
This is a fallback approach when TennisAbstract doesn't have tournament page data.
"""

import pandas as pd
import sys
import re
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ta_tourney_scrape.parse_profile_classic import scrape_player_classic_matches
from ta_tourney_scrape.dedupe import deduplicate_profile_matches
from ta_tourney_scrape.http import get_session

# Initialize session
get_session()

# List of Wimbledon 2025 players (parsed from user input)
WIMBLEDON_2025_PLAYERS = [
    # Champion & Runner-up
    "Jannik Sinner", "Carlos Alcaraz",
    # Semifinals
    "Novak Djokovic", "Taylor Fritz",
    # Quarterfinals
    "Ben Shelton", "Flavio Cobolli", "Karen Khachanov", "Cameron Norrie",
    # 4th round
    "Grigor Dimitrov", "Lorenzo Sonego", "Marin Cilic", "Alex de Minaur",
    "Jordan Thompson", "Kamil Majchrzak", "Nicolas Jarry", "Andrey Rublev",
    # 3rd round
    "Pedro Martinez", "Sebastian Ofner", "Marton Fucsovics", "Brandon Nakashima",
    "Jaume Munar", "Jakub Mensik", "August Homgren", "Miomir Kecmanovic",
    "Alejandro Davidovich Fokina", "Luciano Darderi", "Nuno Borges", "Arthur Rinderknech",
    "Joao Fonseca", "Mattia Bellucci", "Adrian Mannarino", "Jan-Lennard Struff",
    # 2nd round
    "Aleksandar Vukic", "Mariano Navone", "Corentin Moutet", "Tommy Paul",
    "Rinky Hijikata", "Gael Monfils", "Reilly Opelka", "Nikoloz Basilashvili",
    "Jack Draper", "Fabian Marozsan", "Jack Pinnington Jones", "Marcos Giron",
    "Arthur Cazaux", "Tomas Machac", "Jesper de Jong", "Dan Evans",
    "Gabriel Diallo", "Botic van de Zandschulp", "Arthur Fery", "Benjamin Bonzi",
    "Billy Harris", "Shintaro Mochizuki", "Ethan Quinn", "Cristian Garin",
    "Learner Tien", "Jenson Brooksby", "Jiri Lehecka", "Frances Tiafoe",
    "Lloyd Harris", "Valentin Royer", "Felix Auger-Aliassime", "Oliver Tarvet",
    # 1st round
    "Luca Nardi", "Tseng Chun-hsin", "George Loffhagen", "Denis Shapovalov",
    "Yoshihito Nishioka", "Francisco Comesana", "Hamad Medjedovic", "Johannus Monday",
    "Alex Bolt", "David Goffin", "Aleksandar Kovacevic", "Ugo Humbert",
    "Bu Yunchaokete", "Alexander Shevchenko", "Jaime Faria", "Lorenzo Musetti",
    "Sebastian Baez", "Raphael Collignon", "James McCabe", "Alexander Bublik",
    "Beibit Zhukayev", "Tomas Martin Etcheverry", "Camilo Ugo Carabelli", "Hugo Gaston",
    "Roberto Carballes Baena", "Adam Walton", "Quentin Halys", "Damir Dzumhur",
    "Alex Michelsen", "Christopher Eubanks", "Jay Clarke", "Alexandre Muller",
    "Giovanni Mpetshi Perricard", "Daniel Altmaier", "Matteo Arnaldi", "Brandon Holt",
    "Alexei Popyrin", "Roman Safiullin", "Vit Kopriva", "Daniil Medvedev",
    "Francisco Cerundolo", "Dusan Lajovic", "Giulio Zeppieri", "Mackenzie McDonald",
    "Matteo Berrettini", "Henry Searle", "Chris Rodesch", "Alexander Zverev",
    "Holger Rune", "Nishesh Basavareddy", "Jacob Fearnley", "Tallon Griekspoor",
    "Hugo Dellien", "Oliver Crawford", "Roberto Bautista Agut", "Elmer Moller",
    "Laslo Djere", "Zizou Bergs", "Christopher O'Connell", "Stefanos Tsitsipas",
    "James Duckworth", "Filip Misolic", "Leandro Riedi", "Fabio Fognini"
]

def main():
    print(f"🎾 Scraping Wimbledon 2025 from {len(WIMBLEDON_2025_PLAYERS)} player profiles...")
    print(f"⏱️  This will take approximately {len(WIMBLEDON_2025_PLAYERS) * 2.5 / 60:.1f} minutes with rate limiting.\n")
    
    all_wimbledon_matches = []
    failed_players = []
    
    for player_name in tqdm(WIMBLEDON_2025_PLAYERS, desc="Scraping profiles"):
        try:
            # Scrape this player's matches from Wimbledon 2025 using classic player pages
            # The function will filter for Wimbledon in the tournament name
            matches = scrape_player_classic_matches(
                player_name=player_name,
                year=2025,
                tournament_filter="Wimbledon"  # Filter for Wimbledon specifically
            )
            
            if matches:
                all_wimbledon_matches.extend(matches)
                print(f"  ✓ {player_name}: {len(matches)} matches")
            else:
                print(f"  ⚠ {player_name}: No Wimbledon matches found")
                
        except Exception as e:
            print(f"  ✗ {player_name}: Error - {e}")
            failed_players.append(player_name)
    
    print(f"\n📊 Scraping complete:")
    print(f"  Total raw matches scraped: {len(all_wimbledon_matches)}")
    print(f"  Failed players: {len(failed_players)}")
    
    if failed_players:
        print(f"  Failed: {', '.join(failed_players[:10])}" + ("..." if len(failed_players) > 10 else ""))
    
    if not all_wimbledon_matches:
        print("\n❌ No Wimbledon matches scraped. Exiting.")
        return
    
    # Deduplicate (each match appears on both players' profiles)
    print(f"\n🔄 Deduplicating matches...")
    deduplicated_matches = deduplicate_profile_matches(all_wimbledon_matches, year=2025)
    print(f"  Deduplicated to {len(deduplicated_matches)} unique matches")
    
    # Convert to DataFrame
    wimbledon_df = pd.DataFrame(deduplicated_matches)

    # Standardize tournament metadata (do not invent IDs; keep blanks where unknown)
    wimbledon_df['tourney_name'] = 'Wimbledon'
    wimbledon_df['surface'] = 'Grass'
    wimbledon_df['tourney_level'] = 'G'  # Grand Slam
    wimbledon_df['draw_size'] = 128
    wimbledon_df['year'] = 2025
    
    # Save Wimbledon-only data
    wimbledon_output_path = Path('data/atp_matches_2025_wimbledon_only.csv')
    wimbledon_df.to_csv(wimbledon_output_path, index=False)
    print(f"\n💾 Saved Wimbledon-only matches to: {wimbledon_output_path}")
    print(f"   {len(wimbledon_df)} matches")
    
    # Load existing dataset and merge
    existing_data_path = Path('data/atp_matches_2025_ULTIMATE_COMPLETE.csv')
    if existing_data_path.exists():
        print(f"\n🔗 Merging with existing dataset...")
        existing_df = pd.read_csv(existing_data_path)
        print(f"  Existing dataset: {len(existing_df)} matches from {existing_df['tourney_name'].nunique()} tournaments")
        
        # Combine datasets
        # Align columns (Wimbledon data may have different columns)
        # Use columns from existing dataset as the master list
        for col in existing_df.columns:
            if col not in wimbledon_df.columns:
                wimbledon_df[col] = ''
        
        # Keep only columns that exist in the existing dataset
        wimbledon_df = wimbledon_df[existing_df.columns]
        
        # Append only; NEVER delete existing rows.
        # If Wimbledon already exists in the base, only drop duplicates between existing and new Wimbledon rows.
        key_cols = ['tourney_name', 'winner_name', 'loser_name', 'round', 'score']
        existing_keys = set()
        if all(c in existing_df.columns for c in key_cols):
            existing_keys = set(tuple(x) for x in existing_df[key_cols].fillna('').astype(str).to_numpy())

        if all(c in wimbledon_df.columns for c in key_cols) and existing_keys:
            wimbledon_df['_k'] = list(map(tuple, wimbledon_df[key_cols].fillna('').astype(str).to_numpy()))
            wimbledon_df = wimbledon_df[~wimbledon_df['_k'].isin(existing_keys)].drop(columns=['_k'])

        combined_df = pd.concat([existing_df, wimbledon_df], ignore_index=True)
        
        print(f"  Combined dataset: {len(combined_df)} matches from {combined_df['tourney_name'].nunique()} tournaments")
        
        # Save combined dataset
        combined_output_path = Path('data/atp_matches_2025_WITH_WIMBLEDON.csv')
        combined_df.to_csv(combined_output_path, index=False)
        print(f"\n✅ Saved combined dataset to: {combined_output_path}")
        print(f"   Total: {len(combined_df)} matches")
        print(f"   Added: {len(wimbledon_df)} new Wimbledon matches")
        
        # Summary by Grand Slam
        print(f"\n📊 Grand Slam summary in combined dataset:")
        for slam in ['Australian', 'Garros', 'Wimbledon', 'US Open']:
            slam_matches = combined_df[combined_df['tourney_name'].str.contains(slam, case=False, na=False)]
            print(f"   {slam}: {len(slam_matches)} matches")
    else:
        print(f"\n⚠️  No existing dataset found at {existing_data_path}")
        print(f"   Wimbledon-only data saved to {wimbledon_output_path}")

if __name__ == "__main__":
    main()

