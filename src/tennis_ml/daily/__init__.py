"""Daily training and prediction services."""

from .pipeline import run_daily_update, run_strategy_step, scrape_daily_matches
from .matchups import build_matchup_board, fetch_matchups, infer_surface
from .model_artifacts import ensure_latest_model_artifact, latest_model_dir
from .prediction import MatchPredictionRequest, TennisPredictionService, train_daily_model

__all__ = [
    'MatchPredictionRequest',
    'build_matchup_board',
    'fetch_matchups',
    'infer_surface',
    'TennisPredictionService',
    'ensure_latest_model_artifact',
    'latest_model_dir',
    'run_daily_update',
    'run_strategy_step',
    'scrape_daily_matches',
    'train_daily_model',
]
