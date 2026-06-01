"""
Strategy analysis, edge-trust filtering, and research promotion gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

from .pnl_backtester import PnLBacktester


DEFAULT_THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
TRUST_FEATURES = [
    'model_probability',
    'market_probability',
    'odds',
    'edge',
    'ev',
    'clv',
    'tournament_importance',
    'probability_bucket',
    'odds_bucket',
    'rank_band',
    'model_type',
    'surface',
    'round',
]


@dataclass(frozen=True)
class StrategyConfig:
    edge_threshold: float
    model_type: str | None = None
    max_odds: float | None = None
    exclude_slams: bool = False
    surface: str | None = None
    min_edge_trust: float | None = None

    def to_dict(self) -> dict:
        return {
            'edge_threshold': self.edge_threshold,
            'model_type': self.model_type,
            'max_odds': self.max_odds,
            'exclude_slams': self.exclude_slams,
            'surface': self.surface,
            'min_edge_trust': self.min_edge_trust,
        }


def candidate_configs(thresholds: Iterable[float] = DEFAULT_THRESHOLDS) -> list[StrategyConfig]:
    configs = []
    for threshold in thresholds:
        for model_type in [None, 'top_10', 'other']:
            for max_odds in [3.0, 5.0, 10.0, None]:
                for exclude_slams in [False, True]:
                    for surface in [None, 'Hard', 'Clay', 'Grass']:
                        configs.append(StrategyConfig(
                            edge_threshold=threshold,
                            model_type=model_type,
                            max_odds=max_odds,
                            exclude_slams=exclude_slams,
                            surface=surface,
                        ))
    return configs


def strategy_candidate_report(
    predictions: pd.DataFrame,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    stake: float = 1.0,
    friction_bps: float = 200,
) -> pd.DataFrame:
    rows = []
    for config in candidate_configs(thresholds):
        backtester = PnLBacktester(
            edge_threshold=config.edge_threshold,
            stake=stake,
            friction_bps=friction_bps,
        )
        bets = backtester.generate_bets(
            predictions,
            max_odds=config.max_odds,
            model_type=config.model_type,
            exclude_slams=config.exclude_slams,
            surface=config.surface,
        )
        summary = backtester.summarize(bets)
        summary.update(config.to_dict())
        rows.append(summary)
    return pd.DataFrame(rows).sort_values(
        by=['friction_adjusted_roi', 'bets'],
        ascending=[False, False],
    )


def apply_friction_to_bets(bets: pd.DataFrame, friction_bps: float = 200, stake: float = 1.0) -> pd.DataFrame:
    if bets.empty:
        return bets.copy()
    frame = bets.copy()
    if 'friction_bps' not in frame.columns:
        frame['friction_bps'] = friction_bps
    if 'friction_adjusted_odds' not in frame.columns and 'odds' in frame.columns:
        frame['friction_adjusted_odds'] = 1 + (frame['odds'] - 1) * (1 - max(0, friction_bps) / 10000)
    if 'friction_adjusted_profit' not in frame.columns and {'won', 'friction_adjusted_odds'}.issubset(frame.columns):
        frame['friction_adjusted_profit'] = [
            stake * (odds - 1) if bool(won) else -stake
            for odds, won in zip(frame['friction_adjusted_odds'], frame['won'])
        ]
    if 'friction_adjusted_ev' not in frame.columns and {'model_probability', 'friction_adjusted_odds'}.issubset(frame.columns):
        frame['friction_adjusted_ev'] = (
            frame['model_probability'] * (frame['friction_adjusted_odds'] - 1) -
            (1 - frame['model_probability'])
        )
    return frame


def segment_stability_report(bets: pd.DataFrame, stake: float = 1.0) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    rows = []
    frame = bets.copy()
    frame['year'] = pd.to_datetime(frame.get('match_date', frame['tourney_date'])).dt.year
    for column in ['year', 'model_type', 'surface', 'tournament_importance']:
        for value, group in frame.groupby(column, dropna=False):
            summary = PnLBacktester(stake=stake).summarize(group)
            summary['segment'] = column
            summary['value'] = value
            rows.append(summary)
    return pd.DataFrame(rows)


def gate_summary(
    bets: pd.DataFrame,
    calibration: pd.DataFrame | None = None,
    stake: float = 1.0,
    min_bets: int = 300,
    min_friction_roi: float = 0.0,
    min_average_clv: float = 0.0,
    min_positive_clv_rate: float = 0.52,
    max_single_year_profit_share: float = 0.50,
) -> pd.DataFrame:
    summary = PnLBacktester(stake=stake).summarize(bets)
    year_share = single_year_profit_share(bets)
    ece = expected_calibration_error(calibration)
    rows = [
        gate_row('minimum_bets', summary['bets'] >= min_bets, summary['bets'], min_bets),
        gate_row('friction_adjusted_roi', summary['friction_adjusted_roi'] > min_friction_roi, summary['friction_adjusted_roi'], min_friction_roi),
        gate_row('average_clv', summary['average_clv'] > min_average_clv, summary['average_clv'], min_average_clv),
        gate_row('positive_clv_rate', summary['clv_positive_rate'] > min_positive_clv_rate, summary['clv_positive_rate'], min_positive_clv_rate),
        gate_row('single_year_profit_share', year_share <= max_single_year_profit_share, year_share, max_single_year_profit_share, comparison='<='),
        gate_row('calibration_ece', ece <= 0.02, ece, 0.02, comparison='<='),
    ]
    passed = all(row['passed'] for row in rows)
    rows.append({
        'gate': 'overall',
        'passed': passed,
        'value': float(passed),
        'threshold': 1.0,
        'comparison': 'all',
    })
    return pd.DataFrame(rows)


def gate_row(gate: str, passed: bool, value, threshold, comparison: str = '>') -> dict:
    return {
        'gate': gate,
        'passed': bool(passed),
        'value': float(value) if pd.notna(value) else 0.0,
        'threshold': float(threshold),
        'comparison': comparison,
    }


def single_year_profit_share(bets: pd.DataFrame) -> float:
    if bets.empty:
        return 0.0
    frame = bets.copy()
    frame['year'] = pd.to_datetime(frame.get('match_date', frame['tourney_date'])).dt.year
    yearly = frame.groupby('year')['friction_adjusted_profit' if 'friction_adjusted_profit' in frame.columns else 'profit'].sum()
    positive_total = yearly[yearly > 0].sum()
    if positive_total <= 0:
        return 1.0
    return float(yearly.max() / positive_total)


def expected_calibration_error(calibration: pd.DataFrame | None) -> float:
    if calibration is None or calibration.empty or 'ece_component' not in calibration.columns:
        return 0.0
    return float(calibration['ece_component'].sum())


def make_candidate_bets(
    predictions: pd.DataFrame,
    edge_threshold: float,
    stake: float = 1.0,
    friction_bps: float = 200,
) -> pd.DataFrame:
    return PnLBacktester(
        edge_threshold=edge_threshold,
        stake=stake,
        friction_bps=friction_bps,
    ).generate_bets(predictions)


def add_edge_trust(
    predictions: pd.DataFrame,
    edge_threshold: float = 0.03,
    stake: float = 1.0,
    friction_bps: float = 200,
    min_prior_bets: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if 'test_year' not in predictions.columns:
        return predictions.copy(), pd.DataFrame()

    years = sorted(predictions['test_year'].dropna().unique())
    predictions_with_trust = predictions.copy()
    reports = []
    for year in years:
        prior_predictions = predictions[predictions['test_year'] < year]
        current_predictions = predictions[predictions['test_year'] == year]
        if prior_predictions.empty or current_predictions.empty:
            continue
        prior_bets = prepare_trust_frame(make_candidate_bets(prior_predictions, edge_threshold, stake, friction_bps))
        current_bets = prepare_trust_frame(make_candidate_bets(current_predictions, edge_threshold, stake, friction_bps))
        if len(prior_bets) < min_prior_bets or current_bets.empty:
            continue
        model = fit_edge_trust_model(prior_bets)
        probabilities = pd.Series(model.predict_proba(current_bets[TRUST_FEATURES])[:, 1], index=current_bets.index)
        current_bets = current_bets.copy()
        current_bets['edge_trust_probability'] = probabilities
        reports.append({
            'test_year': year,
            'prior_bets': int(len(prior_bets)),
            'scored_bets': int(len(current_bets)),
            'average_edge_trust_probability': float(probabilities.mean()),
            'min_prior_year': int(prior_predictions['test_year'].min()),
            'max_prior_year': int(prior_predictions['test_year'].max()),
        })
        predictions_with_trust = assign_trust_to_predictions(predictions_with_trust, current_bets)
    return predictions_with_trust, pd.DataFrame(reports)


def prepare_trust_frame(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return bets
    frame = bets.copy()
    frame['clv'] = frame['clv'].fillna(0)
    frame['edge_trust_target'] = (
        (frame['friction_adjusted_profit'] > 0) |
        (frame['clv'] > 0)
    ).astype(int)
    frame['probability_bucket'] = pd.cut(frame['model_probability'], bins=10, labels=False, include_lowest=True).astype(float)
    frame['odds_bucket'] = pd.cut(frame['odds'], bins=[1, 1.5, 2, 3, 5, 10, 100], labels=False, include_lowest=True).astype(float)
    frame['rank_band'] = frame['model_type'].fillna('unknown')
    for column in TRUST_FEATURES:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.dropna(subset=['edge_trust_target'])


def fit_edge_trust_model(frame: pd.DataFrame) -> Pipeline:
    categorical = ['model_type', 'surface', 'round', 'rank_band']
    numeric = [column for column in TRUST_FEATURES if column not in categorical]
    preprocessor = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numeric),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ]), categorical),
    ])
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=42,
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])
    pipeline.fit(frame[TRUST_FEATURES], frame['edge_trust_target'])
    return pipeline


def assign_trust_to_predictions(predictions: pd.DataFrame, scored_bets: pd.DataFrame) -> pd.DataFrame:
    updated = predictions.copy()
    key_columns = ['test_year', 'match_date', 'player', 'opponent', 'side']
    side_probability = {
        'player1': 'player1_edge_trust_probability',
        'player2': 'player2_edge_trust_probability',
    }
    for _, bet in scored_bets.iterrows():
        mask = (
            (updated['test_year'] == bet.get('test_year')) &
            (pd.to_datetime(updated['match_date']) == pd.to_datetime(bet.get('match_date'))) &
            (
                ((updated['player1'] == bet['player']) & (bet['side'] == 'player1')) |
                ((updated['player2'] == bet['player']) & (bet['side'] == 'player2'))
            )
        )
        column = side_probability.get(bet['side'])
        if column is not None:
            updated.loc[mask, column] = bet['edge_trust_probability']
    return updated
