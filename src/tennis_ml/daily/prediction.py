"""Daily model training and future-match prediction service."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tennis_ml.backtesting.odds_loader import OddsLoader
from tennis_ml.backtesting.pnl_backtester import PnLBacktester
from tennis_ml.backtesting.strategy_rules import DEFAULT_STRATEGY
from tennis_ml.data import DataLoader
from tennis_ml.data.player import Player
from tennis_ml.features import BettingFeatureEngineer, EloTracker, FeatureEngineer
from tennis_ml.features.columns import FEATURE_COLUMNS
from tennis_ml.live_stats.database import DEFAULT_DB_PATH, insert_model_run, insert_prediction, latest_model_run
from tennis_ml.models.stacked import StackedModel

from .model_artifacts import ensure_latest_model_artifact, latest_model_dir


DEFAULT_ODDS_FILES = ('data/odds/atp_odds_2013_2026.csv',)


TOURNEY_IMPORTANCE = {'G': 5, 'M': 4, 'A': 3, 'C': 2, 'F': 1, 'D': 1}
SURFACE_ENCODING = {'Clay': 0, 'Grass': 1, 'Hard': 2, 'Carpet': 3, 'Unknown': 3}
ROUND_ORDER = {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7, 'BR': 6, 'RR': 3}


@dataclass
class MatchPredictionRequest:
    player1: str
    player2: str
    surface: str = 'Hard'
    tournament_level: str = 'A'
    round: str = 'R32'
    match_date: str | None = None
    player1_odds: float | None = None
    player2_odds: float | None = None
    kalshi_yes_price: float | None = None  # Kalshi YES price (0-1) for player1 winning
    model_dir: str | Path | None = None
    friction_bps: float = 200.0

    def validate(self) -> None:
        if not self.player1 or not self.player2:
            raise ValueError('Both player1 and player2 are required.')
        if self.player1.strip().lower() == self.player2.strip().lower():
            raise ValueError('player1 and player2 must be different players.')
        if self.player1_odds is not None and self.player1_odds <= 1:
            raise ValueError('player1_odds must be greater than 1.0.')
        if self.player2_odds is not None and self.player2_odds <= 1:
            raise ValueError('player2_odds must be greater than 1.0.')
        if self.kalshi_yes_price is not None and not 0 < self.kalshi_yes_price < 1:
            raise ValueError('kalshi_yes_price must be between 0 and 1 (exclusive).')

    def market_probabilities(self) -> tuple[float, float] | None:
        """No-vig market probability pair (player1, player2), if any market is supplied."""
        if self.player1_odds and self.player2_odds:
            raw1, raw2 = 1 / self.player1_odds, 1 / self.player2_odds
            overround = raw1 + raw2
            return raw1 / overround, raw2 / overround
        if self.kalshi_yes_price is not None:
            return float(self.kalshi_yes_price), float(1 - self.kalshi_yes_price)
        return None


def train_daily_model(
    as_of_date: str | pd.Timestamp | None = None,
    data_dir: str | Path = 'tennis_datav2',
    model_root: str | Path = 'models/daily',
    db_path: str | Path = DEFAULT_DB_PATH,
    start_year: int = 2000,
    odds_files: tuple[str, ...] | list[str] = DEFAULT_ODDS_FILES,
) -> dict[str, Any]:
    """Train the daily stacked (market-aware) model from historical ATP data.

    Historical book odds are attached the same way ``walk_forward.py`` does
    (``OddsLoader.align_match_dates`` + ``attach_odds``), so the classifier
    sees the no-vig market probability features and a per-submodel logit
    blend weight is selected on a recent validation slice.
    """
    as_of = pd.to_datetime(as_of_date or pd.Timestamp.utcnow().date())
    artifact_dir = Path(model_root) / as_of.strftime('%Y-%m-%d')
    artifact_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(data_dir=str(data_dir))
    matches = loader.load_data(range(start_year, as_of.year + 1), download_missing=False)
    date_column = 'match_date' if 'match_date' in matches.columns else 'tourney_date'
    matches = matches[pd.to_datetime(matches[date_column], errors='coerce') <= as_of].copy()
    if len(matches) < 500:
        raise ValueError(f'Not enough historical rows to train daily model: {len(matches)}')

    odds_loader = OddsLoader()
    odds = _load_historical_odds(odds_loader, odds_files, as_of)
    if odds is not None:
        matches = odds_loader.align_match_dates(matches, odds)

    sort_columns = [col for col in ['match_date', 'tourney_date', 'tourney_id', 'match_num'] if col in matches.columns]
    matches = matches.sort_values(sort_columns).reset_index(drop=True)
    warmup_end = max(100, int(len(matches) * 0.45))
    warmup = matches.iloc[:warmup_end].copy()
    training_source = matches.iloc[warmup_end:].copy()

    loader.initialize_players(warmup)
    elo = EloTracker()
    elo.initialize(warmup)
    features = FeatureEngineer().create_rolling_features(
        training_source,
        loader.players,
        loader,
        betting_feature_engineer=BettingFeatureEngineer(),
        elo_tracker=elo,
        random_state=99,
    )
    if odds is not None:
        features = odds_loader.attach_odds(features, odds)
    features = features.dropna(subset=['result'])

    model = StackedModel()
    model.fit(features, FEATURE_COLUMNS)
    model.save(artifact_dir)

    submodel_meta = model.metadata_.get('submodels', {})
    metadata = {
        'model_run_id': f'daily-{as_of.strftime("%Y-%m-%d")}',
        'model_kind': 'stacked',
        'trained_at': model.metadata_.get('trained_at', datetime.now(timezone.utc).isoformat()),
        'as_of_date': as_of.strftime('%Y-%m-%d'),
        'artifact_dir': str(artifact_dir),
        'training_rows': int(len(features)),
        'validation_rows': int(sum(meta.get('validation_rows', 0) for meta in submodel_meta.values())),
        'training_window': model.metadata_.get('training_window'),
        'market_features_used': model.metadata_.get('market_features_used'),
        'metrics': {
            f'{name}_blend_weight': meta.get('blend_weight')
            for name, meta in submodel_meta.items()
        },
        'features': model.feature_columns_,
        'odds_files': [str(path) for path in (odds_files or [])] if odds is not None else [],
        'notes': 'Daily stacked retrain (XGBoost + no-vig market blend) from historical ATP data with Tennis-Data odds features.',
    }
    (artifact_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')
    insert_model_run(metadata, db_path=db_path)
    return metadata


def _load_historical_odds(
    odds_loader: OddsLoader,
    odds_files: tuple[str, ...] | list[str],
    as_of: pd.Timestamp,
) -> pd.DataFrame | None:
    """Load and concatenate historical odds files; None when unavailable."""
    frames = []
    for path in odds_files or []:
        candidate = Path(path)
        if not candidate.exists():
            continue
        try:
            frames.append(odds_loader.load(str(candidate)))
        except Exception:
            continue
    if not frames:
        return None
    odds = pd.concat(frames, ignore_index=True)
    odds = odds[pd.to_datetime(odds['match_date'], errors='coerce') <= as_of]
    return odds if not odds.empty else None


class TennisPredictionService:
    """Loads the latest daily model and scores player-vs-player requests."""

    def __init__(
        self,
        data_dir: str | Path = 'tennis_datav2',
        db_path: str | Path = DEFAULT_DB_PATH,
        model_dir: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self.model_dir = Path(model_dir) if model_dir else self._latest_model_dir()
        self.bundle = self._load_bundle(self.model_dir)

    def predict(self, request: MatchPredictionRequest, persist: bool = True) -> dict[str, Any]:
        request.validate()
        as_of = pd.to_datetime(request.match_date or pd.Timestamp.utcnow().date())
        context = self._build_context(as_of)
        feature_row = self._feature_row(request, context, as_of)
        market = request.market_probabilities()
        if market is not None:
            feature_row['player1_market_probability_novig'] = market[0]
            feature_row['player2_market_probability_novig'] = market[1]
            feature_row['market_probability_diff'] = market[0] - market[1]
            if request.player1_odds and request.player2_odds:
                feature_row['market_overround'] = 1 / request.player1_odds + 1 / request.player2_odds
            else:
                feature_row['market_overround'] = 1.0

        if self.bundle.get('kind') == 'stacked':
            stacked: StackedModel = self.bundle['stacked']
            frame = pd.DataFrame([feature_row])
            p1 = float(stacked.predict_proba(frame).iloc[0])
            p2 = 1 - p1
        else:
            features = self.bundle['features']
            frame = pd.DataFrame([feature_row]).reindex(columns=features, fill_value=np.nan)
            if self.bundle.get('imputer') is not None:
                values = pd.DataFrame(
                    self.bundle['imputer'].transform(frame),
                    columns=features,
                )
            else:
                values = frame.fillna(0)
            values = self.bundle['scaler'].transform(values)
            probabilities = self.bundle['model'].predict_proba(values)[0]
            class_probabilities = dict(zip(self.bundle['model'].classes_, probabilities))
            p1 = float(class_probabilities.get(1, 0.0))
            p2 = float(class_probabilities.get(2, 1 - p1))
        result = self._decision(request, p1, p2, feature_row)
        result.update({
            'model_kind': self.bundle.get('kind', 'legacy'),
            'market_probability_player1': market[0] if market is not None else None,
            'market_probability_player2': market[1] if market is not None else None,
            'kalshi_yes_price': request.kalshi_yes_price,
            'prediction_id': self._prediction_id(request, p1, p2),
            'prediction_time': datetime.now(timezone.utc).isoformat(),
            'model_run_id': self.bundle.get('metadata', {}).get('model_run_id'),
            'match_date': as_of.strftime('%Y-%m-%d'),
            'player1': request.player1,
            'player2': request.player2,
            'surface': request.surface,
            'tournament_level': request.tournament_level,
            'round': request.round,
            'player1_probability': p1,
            'player2_probability': p2,
            'fair_odds_player1': fair_odds(p1),
            'fair_odds_player2': fair_odds(p2),
            'confidence': self._confidence_label(feature_row, p1),
        })
        if persist:
            insert_prediction(result, db_path=self.db_path)
        return result

    def _build_context(self, as_of: pd.Timestamp) -> dict[str, Any]:
        loader = DataLoader(data_dir=str(self.data_dir))
        matches = loader.load_data(range(2000, as_of.year + 1), download_missing=False)
        date_column = 'match_date' if 'match_date' in matches.columns else 'tourney_date'
        matches = matches[pd.to_datetime(matches[date_column], errors='coerce') < as_of].copy()
        loader.initialize_players(matches)
        elo = EloTracker()
        elo.initialize(matches)
        return {'loader': loader, 'players': loader.players, 'elo': elo}

    def _feature_row(self, request: MatchPredictionRequest, context: dict[str, Any], as_of: pd.Timestamp) -> dict[str, Any]:
        players = context['players']
        p1 = players.get(request.player1, Player(request.player1))
        p2 = players.get(request.player2, Player(request.player2))
        surface = normalize_surface(request.surface)
        p1_rank = last_numeric(p1.ranks, 999)
        p2_rank = last_numeric(p2.ranks, 999)
        p1_age = last_numeric(p1.ages, 27)
        p2_age = last_numeric(p2.ages, 27)
        p1_seed = last_numeric(p1.seeds, np.nan)
        p2_seed = last_numeric(p2.seeds, np.nan)
        p1_height = last_numeric(p1.heights, 185)
        p2_height = last_numeric(p2.heights, 185)
        p1_hand = last_value(p1.hands, 'R')
        p2_hand = last_value(p2.hands, 'R')
        betting = BettingFeatureEngineer()
        row = {
            'player1_rank': p1_rank,
            'player2_rank': p2_rank,
            'rank_diff': p1_rank - p2_rank,
            'log_rank_diff': np.log1p(abs(p1_rank - p2_rank)),
            'top_10_vs_not': ((p1_rank <= 10) and (p2_rank > 10)) or ((p2_rank <= 10) and (p1_rank > 10)),
            'age_diff': p1_age - p2_age,
            'young_vs_old': ((p1_age < 25) and (p2_age > 30)) or ((p2_age < 25) and (p1_age > 30)),
            'seed_diff': p1_seed - p2_seed,
            'seeded_vs_unseeded': (pd.notna(p1_seed) and pd.isna(p2_seed)) or (pd.notna(p2_seed) and pd.isna(p1_seed)),
            'height_diff': p1_height - p2_height,
            'tall_vs_short': ((p1_height > 190) and (p2_height < 180)) or ((p2_height > 190) and (p1_height < 180)),
            'same_hand': p1_hand == p2_hand,
            'left_vs_right': ((p1_hand == 'L') and (p2_hand == 'R')) or ((p2_hand == 'L') and (p1_hand == 'R')),
            'player1_last_5_win_percentage': p1.last_5_win_percentage(),
            'player2_last_5_win_percentage': p2.last_5_win_percentage(),
            'player1_last_10_win_percentage': p1.last_10_win_percentage(),
            'player2_last_10_win_percentage': p2.last_10_win_percentage(),
            'player1_surface_last_10_win_percentage': p1.surface_last_10_win_percentage(surface),
            'player2_surface_last_10_win_percentage': p2.surface_last_10_win_percentage(surface),
            'player1_surface_match': surface == p1.preferred_surface(),
            'player2_surface_match': surface == p2.preferred_surface(),
            'surface_preference_diff': int(surface == p1.preferred_surface()) - int(surface == p2.preferred_surface()),
            'head_to_head_wins_p1': p1.head_to_head_stats(request.player2).get('wins', 0),
            'head_to_head_wins_p2': p2.head_to_head_stats(request.player1).get('wins', 0),
            'surface_encoded': SURFACE_ENCODING.get(surface, 3),
            'tournament_importance': TOURNEY_IMPORTANCE.get(request.tournament_level, 1),
            'player1_momentum': betting._calculate_weighted_form(request.player1, players),
            'player2_momentum': betting._calculate_weighted_form(request.player2, players),
            'player1_confidence': betting._calculate_confidence(request.player1, players),
            'player2_confidence': betting._calculate_confidence(request.player2, players),
            'player1_surface_advantage': p1.win_percentage(surface),
            'player2_surface_advantage': p2.win_percentage(surface),
            'round_encoded': ROUND_ORDER.get(request.round, 0),
            'best_of': 5 if request.tournament_level == 'G' else 3,
            'draw_size': 128 if request.tournament_level == 'G' else 56,
            'is_grand_slam': request.tournament_level == 'G',
            'is_davis_cup': request.tournament_level == 'D',
            'is_final': request.round == 'F',
            'is_early_round': request.round in {'R128', 'R64', 'R32'},
            'player1_days_since_last_match': p1.days_since_last_match(as_of),
            'player2_days_since_last_match': p2.days_since_last_match(as_of),
            'days_since_last_match_diff': p1.days_since_last_match(as_of) - p2.days_since_last_match(as_of),
            'player1_matches_last_7d': p1.matches_last_days(as_of, 7),
            'player2_matches_last_7d': p2.matches_last_days(as_of, 7),
            'matches_last_7d_diff': p1.matches_last_days(as_of, 7) - p2.matches_last_days(as_of, 7),
            'player1_matches_last_14d': p1.matches_last_days(as_of, 14),
            'player2_matches_last_14d': p2.matches_last_days(as_of, 14),
            'matches_last_14d_diff': p1.matches_last_days(as_of, 14) - p2.matches_last_days(as_of, 14),
            'player1_matches_last_30d': p1.matches_last_days(as_of, 30),
            'player2_matches_last_30d': p2.matches_last_days(as_of, 30),
            'matches_last_30d_diff': p1.matches_last_days(as_of, 30) - p2.matches_last_days(as_of, 30),
            'player1_avg_minutes_last_3': p1.avg_minutes(3),
            'player2_avg_minutes_last_3': p2.avg_minutes(3),
            'avg_minutes_last_3_diff': p1.avg_minutes(3) - p2.avg_minutes(3),
            'player1_serve_points_won_l10': p1.recent_serve_points_won(),
            'player2_serve_points_won_l10': p2.recent_serve_points_won(),
            'serve_points_won_l10_diff': p1.recent_serve_points_won() - p2.recent_serve_points_won(),
            'player1_return_points_won_l10': p1.recent_return_points_won(),
            'player2_return_points_won_l10': p2.recent_return_points_won(),
            'return_points_won_l10_diff': p1.recent_return_points_won() - p2.recent_return_points_won(),
            'player1_ace_rate_l10': p1.recent_ace_rate(),
            'player2_ace_rate_l10': p2.recent_ace_rate(),
            'ace_rate_l10_diff': p1.recent_ace_rate() - p2.recent_ace_rate(),
            'player1_double_fault_rate_l10': p1.recent_double_fault_rate(),
            'player2_double_fault_rate_l10': p2.recent_double_fault_rate(),
            'double_fault_rate_l10_diff': p1.recent_double_fault_rate() - p2.recent_double_fault_rate(),
        }
        row.update(context['elo'].features(request.player1, request.player2, surface))
        return row

    def _decision(self, request: MatchPredictionRequest, p1: float, p2: float, feature_row: dict[str, Any]) -> dict[str, Any]:
        odds = {'player1': request.player1_odds, 'player2': request.player2_odds}
        candidates = []
        for side, probability in [('player1', p1), ('player2', p2)]:
            decimal_odds = odds[side]
            if decimal_odds is None:
                continue
            adjusted_odds = PnLBacktester(friction_bps=request.friction_bps)._apply_friction(decimal_odds)
            market_probability = 1 / decimal_odds
            edge = probability - market_probability
            ev = probability * (adjusted_odds - 1) - (1 - probability)
            candidates.append((side, probability, decimal_odds, edge, ev))

        if not candidates:
            return {
                'market_odds_player1': request.player1_odds,
                'market_odds_player2': request.player2_odds,
                'recommended_side': None,
                'strategy_decision': 'fair-odds-only',
                'edge': None,
                'ev': None,
                'notes': 'No market odds supplied; showing model fair odds only.',
            }

        side, probability, decimal_odds, edge, ev = max(candidates, key=lambda item: item[4])
        no_slam = feature_row.get('tournament_importance') != 5
        in_favorite_band = 1.5 <= decimal_odds < 2.0
        decision = 'bet' if ev > 0 and in_favorite_band and no_slam else 'pass'
        notes = f'Default strategy={DEFAULT_STRATEGY}; requires positive friction-adjusted EV, odds 1.50-2.00, and non-Slam.'
        return {
            'market_odds_player1': request.player1_odds,
            'market_odds_player2': request.player2_odds,
            'recommended_side': side,
            'strategy_decision': decision,
            'edge': float(edge),
            'ev': float(ev),
            'notes': notes,
        }

    def _confidence_label(self, feature_row: dict[str, Any], p1: float) -> str:
        low_data = min(feature_row.get('player1_elo_matches', 0), feature_row.get('player2_elo_matches', 0)) < 10
        if low_data:
            return 'low-data'
        if abs(p1 - 0.5) >= 0.12:
            return 'high'
        if abs(p1 - 0.5) >= 0.06:
            return 'medium'
        return 'low'

    def _latest_model_dir(self) -> Path:
        candidates: list[Path] = []
        run = latest_model_run(self.db_path)
        if run and run.get('artifact_dir'):
            path = Path(run['artifact_dir'])
            if StackedModel.exists(path) or (path / 'daily_ensemble_model.pkl').exists():
                candidates.append(path)
        local = latest_model_dir('models/daily')
        if local is not None:
            candidates.append(local)
        if candidates:
            # Prefer stacked artifacts (production model) over legacy ensembles,
            # then the newest by directory name.
            stacked = [path for path in candidates if StackedModel.exists(path)]
            pool = stacked or candidates
            return sorted(pool, key=lambda path: path.name)[-1]
        try:
            return ensure_latest_model_artifact('models/daily')
        except Exception as exc:
            raise FileNotFoundError(
                'No daily model found locally, and the latest release artifact could not be downloaded. '
                'Run the GitHub Actions daily model workflow or use Model Setup in the sidebar.'
            ) from exc

    @staticmethod
    def _load_bundle(model_dir: Path) -> dict[str, Any]:
        metadata_path = model_dir / 'metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
        if StackedModel.exists(model_dir):
            return {
                'kind': 'stacked',
                'stacked': StackedModel.load(model_dir),
                'metadata': metadata,
            }
        bundle = {
            'kind': 'legacy',
            'model': joblib.load(model_dir / 'daily_ensemble_model.pkl'),
            'scaler': joblib.load(model_dir / 'daily_ensemble_scaler.pkl'),
            'features': joblib.load(model_dir / 'daily_ensemble_features.pkl'),
            'metadata': metadata,
        }
        imputer_path = model_dir / 'daily_ensemble_imputer.pkl'
        if imputer_path.exists():
            bundle['imputer'] = joblib.load(imputer_path)
        return bundle

    @staticmethod
    def _prediction_id(request: MatchPredictionRequest, p1: float, p2: float) -> str:
        payload = '|'.join([
            request.match_date or '',
            request.player1,
            request.player2,
            request.surface,
            str(round(p1, 6)),
            str(round(p2, 6)),
        ])
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]


def fair_odds(probability: float) -> float | None:
    if probability <= 0:
        return None
    return float(1 / probability)


def normalize_surface(surface: str) -> str:
    mapping = {'hard': 'Hard', 'clay': 'Clay', 'grass': 'Grass', 'carpet': 'Carpet'}
    return mapping.get(str(surface).strip().lower(), 'Unknown')


def last_numeric(values: list[Any], default: float) -> float:
    for value in reversed(values):
        converted = pd.to_numeric(value, errors='coerce')
        if pd.notna(converted):
            return float(converted)
    return float(default)


def last_value(values: list[Any], default: Any) -> Any:
    for value in reversed(values):
        if value is not None and pd.notna(value):
            return value
    return default
