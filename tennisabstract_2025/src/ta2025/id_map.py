"""
Player ID mapping using Jeff Sackmann's atp_players.csv.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from .utils import normalize_name

logger = logging.getLogger(__name__)


class PlayerIDMapper:
    """Maps player names to IDs using Sackmann's player database."""
    
    def __init__(self, players_csv_path: str, overrides_csv_path: Optional[str] = None):
        """
        Initialize the player ID mapper.
        
        Args:
            players_csv_path: Path to atp_players.csv
            overrides_csv_path: Optional path to name_overrides.csv
        """
        self.players_csv_path = Path(players_csv_path)
        self.overrides_csv_path = Path(overrides_csv_path) if overrides_csv_path else None
        self.player_db: Dict[str, str] = {}  # normalized_name -> player_id
        self.overrides: Dict[str, str] = {}  # original_name -> player_id
        self.missing_players: Dict[str, int] = {}  # player_name -> count
        
        self._load_players()
        self._load_overrides()
    
    def _load_players(self):
        """Load player database from atp_players.csv."""
        if not self.players_csv_path.exists():
            logger.warning(f"Player database not found: {self.players_csv_path}")
            return
        
        try:
            with open(self.players_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    player_id = row.get('player_id', '').strip()
                    name = row.get('name', '').strip()
                    
                    if player_id and name:
                        # Store normalized name -> ID mapping
                        normalized = normalize_name(name)
                        if normalized:
                            self.player_db[normalized] = player_id
        except Exception as e:
            logger.error(f"Error loading player database: {e}")
    
    def _load_overrides(self):
        """Load manual name overrides."""
        if not self.overrides_csv_path or not self.overrides_csv_path.exists():
            return
        
        try:
            with open(self.overrides_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    original_name = row.get('original_name', '').strip()
                    player_id = row.get('player_id', '').strip()
                    
                    if original_name and player_id:
                        self.overrides[original_name] = player_id
        except Exception as e:
            logger.warning(f"Error loading overrides: {e}")
    
    def find_player_id(self, player_name: str) -> Optional[str]:
        """
        Find player ID for a given name.
        
        Args:
            player_name: Player name to look up
            
        Returns:
            Player ID or None if not found
        """
        if not player_name:
            return None
        
        # Check overrides first
        if player_name in self.overrides:
            return self.overrides[player_name]
        
        # Try exact normalized match
        normalized = normalize_name(player_name)
        if normalized in self.player_db:
            return self.player_db[normalized]
        
        # Try last name + first initial
        parts = normalized.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            first_initial = parts[0][0] if parts[0] else ''
            partial_key = f"{last_name} {first_initial}"
            
            # Find matches
            for key, player_id in self.player_db.items():
                if key.startswith(partial_key):
                    return player_id
        
        # Not found
        self.missing_players[player_name] = self.missing_players.get(player_name, 0) + 1
        return None
    
    def get_missing_players(self) -> Dict[str, int]:
        """Get dictionary of missing players and their counts."""
        return self.missing_players.copy()
    
    def save_missing_players(self, output_path: str):
        """Save missing players to CSV."""
        if not self.missing_players:
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['player_name', 'count'])
            for name, count in sorted(self.missing_players.items(), key=lambda x: -x[1]):
                writer.writerow([name, count])
        
        logger.info(f"Saved {len(self.missing_players)} missing players to {output_path}")

