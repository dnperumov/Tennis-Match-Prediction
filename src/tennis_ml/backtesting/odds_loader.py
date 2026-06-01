"""
Historical odds loading and matching helpers.
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def normalize_player_name(name: str) -> str:
    """Normalize player names for cross-source matching."""
    if pd.isna(name):
        return ''
    normalized = unicodedata.normalize('NFKD', str(name))
    normalized = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r'[^a-z0-9]+', ' ', normalized)
    return re.sub(r'\s+', ' ', normalized).strip()


def player_name_key_variants(name: str) -> List[str]:
    """Return likely key variants for full and abbreviated tennis names."""
    normalized = normalize_player_name(name)
    tokens = normalized.split()
    if len(tokens) <= 1:
        return [normalized]

    variants = {normalized}
    for surname_length in (1, 2, 3):
        if len(tokens) <= surname_length:
            continue
        surname = ' '.join(tokens[-surname_length:])
        initials = ' '.join(token[0] for token in tokens[:-surname_length] if token)
        variants.add(f'{surname} {initials}'.strip())

    return sorted(variants)


@dataclass
class OddsColumns:
    winner: str
    loser: str
    date: str
    winner_odds: str
    loser_odds: str


class OddsLoader:
    """Loads historical tennis odds and matches them to prediction rows."""

    ODDS_COLUMN_PRIORITY = (
        ('AvgW', 'AvgL'),
        ('MaxW', 'MaxL'),
        ('PSW', 'PSL'),
        ('B365W', 'B365L'),
        ('EXW', 'EXL'),
        ('LBW', 'LBL'),
    )

    def load(self, odds_file: str, preferred_columns: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """Load odds CSV/Excel and return a standardized winner/loser odds frame."""
        path = Path(odds_file)
        if path.suffix.lower() in {'.xls', '.xlsx'}:
            raw = pd.read_excel(path)
        else:
            raw = pd.read_csv(path, low_memory=False)

        columns = self._resolve_columns(raw, preferred_columns)
        keep_columns = [columns.date, columns.winner, columns.loser, columns.winner_odds, columns.loser_odds]
        optional_columns = ['AvgW', 'AvgL', 'MaxW', 'MaxL', 'PSW', 'PSL', 'B365W', 'B365L', 'EXW', 'EXL', 'BFEW', 'BFEL']
        keep_columns.extend(column for column in optional_columns if column in raw.columns and column not in keep_columns)
        odds = raw[keep_columns].copy()
        odds = odds.rename(columns={
            columns.date: 'match_date',
            columns.winner: 'winner_name',
            columns.loser: 'loser_name',
            columns.winner_odds: 'winner_odds',
            columns.loser_odds: 'loser_odds',
        })
        odds['match_date'] = self._parse_dates(odds['match_date'])
        odds['winner_key'] = odds['winner_name'].map(normalize_player_name)
        odds['loser_key'] = odds['loser_name'].map(normalize_player_name)
        for column in ['winner_odds', 'loser_odds'] + optional_columns:
            if column in odds.columns:
                odds[column] = pd.to_numeric(odds[column], errors='coerce')
        if columns.winner_odds == 'AvgW' and columns.loser_odds == 'AvgL':
            odds['closing_winner_odds'] = odds['winner_odds']
            odds['closing_loser_odds'] = odds['loser_odds']
        elif {'AvgW', 'AvgL'}.issubset(odds.columns):
            odds['closing_winner_odds'] = odds['AvgW']
            odds['closing_loser_odds'] = odds['AvgL']
        elif {'MaxW', 'MaxL'}.issubset(odds.columns):
            odds['closing_winner_odds'] = odds['MaxW']
            odds['closing_loser_odds'] = odds['MaxL']
        else:
            odds['closing_winner_odds'] = odds['winner_odds']
            odds['closing_loser_odds'] = odds['loser_odds']
        odds = odds.dropna(subset=['match_date', 'winner_odds', 'loser_odds'])
        odds['execution_odds_source'] = f'{columns.winner_odds}/{columns.loser_odds}'
        odds['closing_odds_source'] = 'AvgW/AvgL' if {'AvgW', 'AvgL'}.issubset(odds.columns) else 'selected'
        return odds.drop_duplicates(subset=['match_date', 'winner_key', 'loser_key'], keep='last')

    def align_match_dates(self, matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
        """
        Add exact Tennis-Data match dates to Jeff Sackmann match rows.

        Jeff data dates are tournament start dates. Tennis-Data odds rows have
        actual match dates, so this alignment keeps the original tournament date
        in ``tourney_start_date`` and adds/uses ``match_date`` for chronological
        feature construction and backtesting.
        """
        aligned = matches.copy().reset_index(drop=True)
        aligned['match_id'] = aligned.index
        aligned['tourney_start_date'] = pd.to_datetime(aligned['tourney_date'], errors='coerce')
        aligned['match_date'] = aligned['tourney_start_date']
        aligned['winner_key'] = aligned['winner_name'].map(player_name_key_variants)
        aligned['loser_key'] = aligned['loser_name'].map(player_name_key_variants)
        exploded = aligned[['match_id', 'tourney_start_date', 'winner_key', 'loser_key']].explode('winner_key')
        exploded = exploded.explode('loser_key')

        odds_to_match = odds[['match_date', 'winner_key', 'loser_key']].copy()
        odds_to_match = odds_to_match.rename(columns={'match_date': 'odds_date'})
        matched = exploded.merge(odds_to_match, on=['winner_key', 'loser_key'], how='left')
        matched = matched.dropna(subset=['odds_date'])
        matched['days_from_tourney_start'] = (
            pd.to_datetime(matched['odds_date']) - pd.to_datetime(matched['tourney_start_date'])
        ).dt.days
        matched = matched[
            (matched['days_from_tourney_start'] >= 0) &
            (matched['days_from_tourney_start'] <= 21)
        ]
        matched = matched.sort_values(by=['match_id', 'days_from_tourney_start'])
        matched = matched.drop_duplicates(subset=['match_id'], keep='first')
        aligned = aligned.drop(columns=['winner_key', 'loser_key']).merge(
            matched[['match_id', 'odds_date', 'days_from_tourney_start']],
            on='match_id',
            how='left'
        )
        matched_date = pd.to_datetime(aligned['odds_date'], errors='coerce')
        aligned['match_date'] = matched_date.fillna(aligned['match_date'])
        aligned['match_date_aligned'] = aligned['odds_date'].notna()
        aligned = aligned.drop(columns=['odds_date'])
        sort_columns = [col for col in ['match_date', 'tourney_date', 'tourney_id', 'match_num'] if col in aligned.columns]
        return aligned.sort_values(by=sort_columns).reset_index(drop=True)

    def attach_odds(self, predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
        """Attach winner/loser odds to prediction rows by date and player names."""
        predicted = predictions.copy().reset_index(drop=True)
        predicted['prediction_id'] = predicted.index
        source_date = 'match_date' if 'match_date' in predicted.columns else 'tourney_date'
        predicted['match_date'] = pd.to_datetime(predicted[source_date], errors='coerce').dt.date
        predicted['winner_key'] = predicted['winner_name'].map(player_name_key_variants)
        predicted['loser_key'] = predicted['loser_name'].map(player_name_key_variants)
        exploded = predicted[['prediction_id', 'match_date', 'winner_key', 'loser_key']].explode('winner_key')
        exploded = exploded.explode('loser_key')

        odds_columns = [
            'match_date', 'winner_key', 'loser_key',
            'winner_odds', 'loser_odds',
            'closing_winner_odds', 'closing_loser_odds',
            'execution_odds_source', 'closing_odds_source',
        ]
        odds_to_match = odds[[column for column in odds_columns if column in odds.columns]].copy()
        odds_to_match = odds_to_match.rename(columns={'match_date': 'odds_date'})
        matched_odds = exploded.merge(
            odds_to_match,
            on=['winner_key', 'loser_key'],
            how='left'
        )
        matched_odds = matched_odds.dropna(subset=['winner_odds', 'loser_odds'])
        matched_odds['days_from_tourney_start'] = (
            pd.to_datetime(matched_odds['odds_date']) -
            pd.to_datetime(matched_odds['match_date'])
        ).dt.days
        max_window = 0 if source_date == 'match_date' else 21
        matched_odds = matched_odds[
            (matched_odds['days_from_tourney_start'] >= 0) &
            (matched_odds['days_from_tourney_start'] <= max_window)
        ]
        matched_odds = matched_odds.sort_values(
            by=['prediction_id', 'days_from_tourney_start']
        )
        matched_odds = matched_odds.drop_duplicates(subset=['prediction_id'], keep='first')
        merged = predicted.drop(columns=['winner_key', 'loser_key']).merge(
            matched_odds[[column for column in [
                'prediction_id', 'odds_date', 'winner_odds', 'loser_odds',
                'closing_winner_odds', 'closing_loser_odds',
                'execution_odds_source', 'closing_odds_source',
            ] if column in matched_odds.columns]],
            on='prediction_id',
            how='left'
        )
        merged['player1_odds'] = merged.apply(
            lambda row: row['winner_odds'] if row['player1'] == row['winner_name'] else row['loser_odds'],
            axis=1
        )
        merged['player2_odds'] = merged.apply(
            lambda row: row['winner_odds'] if row['player2'] == row['winner_name'] else row['loser_odds'],
            axis=1
        )
        if {'closing_winner_odds', 'closing_loser_odds'}.issubset(merged.columns):
            merged['player1_closing_odds'] = merged.apply(
                lambda row: row['closing_winner_odds'] if row['player1'] == row['winner_name'] else row['closing_loser_odds'],
                axis=1
            )
            merged['player2_closing_odds'] = merged.apply(
                lambda row: row['closing_winner_odds'] if row['player2'] == row['winner_name'] else row['closing_loser_odds'],
                axis=1
            )
        merged['player1_market_probability'] = 1 / merged['player1_odds']
        merged['player2_market_probability'] = 1 / merged['player2_odds']
        merged['market_overround'] = (
            merged['player1_market_probability'] + merged['player2_market_probability']
        )
        merged['player1_market_probability_novig'] = (
            merged['player1_market_probability'] / merged['market_overround']
        )
        merged['player2_market_probability_novig'] = (
            merged['player2_market_probability'] / merged['market_overround']
        )
        if {'player1_probability', 'player2_probability'}.issubset(merged.columns):
            merged['player1_model_market_edge'] = (
                merged['player1_probability'] - merged['player1_market_probability_novig']
            )
            merged['player2_model_market_edge'] = (
                merged['player2_probability'] - merged['player2_market_probability_novig']
            )
        merged['market_probability_diff'] = (
            merged['player1_market_probability_novig'] - merged['player2_market_probability_novig']
        )
        return merged

    def _resolve_columns(
        self,
        df: pd.DataFrame,
        preferred_columns: Optional[Tuple[str, str]]
    ) -> OddsColumns:
        required = {'Winner', 'Loser', 'Date'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Odds file missing required columns: {sorted(missing)}")

        if preferred_columns is not None:
            winner_odds, loser_odds = preferred_columns
            if winner_odds not in df.columns or loser_odds not in df.columns:
                raise ValueError(f"Preferred odds columns not found: {preferred_columns}")
            return OddsColumns('Winner', 'Loser', 'Date', winner_odds, loser_odds)

        for winner_odds, loser_odds in self.ODDS_COLUMN_PRIORITY:
            if winner_odds in df.columns and loser_odds in df.columns:
                return OddsColumns('Winner', 'Loser', 'Date', winner_odds, loser_odds)

        raise ValueError(
            "Could not find supported odds columns. Expected one of "
            f"{self.ODDS_COLUMN_PRIORITY}"
        )

    @staticmethod
    def _parse_dates(values: pd.Series) -> pd.Series:
        text_values = values.astype(str)
        iso_mask = text_values.str.contains('-', na=False)
        parsed = pd.Series(pd.NaT, index=values.index, dtype='datetime64[ns]')
        parsed.loc[iso_mask] = pd.to_datetime(values.loc[iso_mask], errors='coerce')
        parsed.loc[~iso_mask] = pd.to_datetime(values.loc[~iso_mask], errors='coerce', dayfirst=True)
        return parsed.dt.date
