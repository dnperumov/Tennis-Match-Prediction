#!/usr/bin/env python3
"""
Yearly walk-forward training and betting backtest.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backtest import (
    add_model_market_edges,
    print_summary,
    score_matches,
    summarize_by_model_type,
    summarize_by_year,
    summarize_segments,
    summarize_threshold_grid,
)
from tennis_ml.backtesting import (
    OddsLoader,
    PnLBacktester,
    add_edge_trust,
    gate_summary,
    segment_stability_report,
    strategy_candidate_report,
)
from tennis_ml.data import DataLoader
from tennis_ml.features import BettingFeatureEngineer, EloTracker, FeatureEngineer
from tennis_ml.features.columns import FEATURE_COLUMNS
from tennis_ml.models import ModelTrainer


CACHE_VERSION = 'walk-forward-cache-v1'
ALIGNED_CACHE_VERSION = 'aligned-matches-v1'
FOLD_FEATURE_CACHE_VERSION = 'fold-features-v1'


def main():
    parser = argparse.ArgumentParser(description='Walk-forward tennis model backtest')
    parser.add_argument('--data-dir', default='tennis_datav2')
    parser.add_argument('--odds-file', nargs='+', required=True)
    parser.add_argument('--winner-odds-col', help='Optional execution odds column, e.g. B365W')
    parser.add_argument('--loser-odds-col', help='Optional execution odds column, e.g. B365L')
    parser.add_argument('--train-start-year', type=int, default=2013)
    parser.add_argument('--test-start-year', type=int, default=2019)
    parser.add_argument('--test-end-year', type=int, default=2027,
                        help='End-exclusive test year')
    parser.add_argument('--edge-threshold', type=float, default=0.03)
    parser.add_argument('--threshold-grid', type=float, nargs='*',
                        default=[0.01, 0.02, 0.03, 0.05, 0.075, 0.10])
    parser.add_argument('--stake', type=float, default=1.0)
    parser.add_argument('--friction-bps', type=float, default=200,
                        help='Odds haircut in basis points for friction-adjusted PnL')
    parser.add_argument('--min-odds', type=float)
    parser.add_argument('--max-odds', type=float)
    parser.add_argument('--model-type', choices=['top_10', 'other'])
    parser.add_argument('--exclude-slams', action='store_true')
    parser.add_argument('--surface', choices=['Hard', 'Clay', 'Grass'])
    parser.add_argument('--output-dir', default='walk_forward_results')
    parser.add_argument('--probability-model', choices=['residual', 'classifier'], default='residual',
                        help='Use market-residual probabilities or the original classifiers')
    parser.add_argument('--cache-dir', default='.cache/walk_forward',
                        help='Directory for aligned match and fold feature caches')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable reading and writing walk-forward caches')
    parser.add_argument('--refresh-cache', action='store_true',
                        help='Rebuild cache entries even when matching cache files exist')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print('[1/4] Loading match data and odds...')
    data_loader = DataLoader(data_dir=args.data_dir)
    matches = data_loader.load_data(years=range(2000, args.test_end_year))
    odds_loader = OddsLoader()
    odds_columns = None
    if args.winner_odds_col or args.loser_odds_col:
        if not args.winner_odds_col or not args.loser_odds_col:
            raise ValueError('Pass both --winner-odds-col and --loser-odds-col, or neither.')
        odds_columns = (args.winner_odds_col, args.loser_odds_col)
    odds = pd.concat([odds_loader.load(path, preferred_columns=odds_columns) for path in args.odds_file], ignore_index=True)
    alignment_key = alignment_cache_key(
        data_dir=args.data_dir,
        end_year=args.test_end_year,
        odds_files=args.odds_file,
        odds_columns=odds_columns,
    )
    matches = align_matches_with_cache(
        matches=matches,
        odds=odds,
        odds_loader=odds_loader,
        cache_dir=cache_dir,
        cache_key=alignment_key,
        use_cache=use_cache,
        refresh_cache=args.refresh_cache,
    )
    aligned_rate = float(matches['match_date_aligned'].mean()) if 'match_date_aligned' in matches.columns else 0.0
    print(f'Aligned exact match dates for {aligned_rate:.1%} of rows.')

    fold_predictions = []
    fold_metrics = []

    for test_year in range(args.test_start_year, args.test_end_year):
        print(f'[2/4] Fold {test_year}: train {args.train_start_year}-{test_year - 1}, test {test_year}')
        train_features, test_features = build_fold_features_cached(
            matches=matches,
            odds=odds,
            odds_loader=odds_loader,
            train_start_year=args.train_start_year,
            test_year=test_year,
            alignment_key=alignment_key,
            cache_dir=cache_dir,
            use_cache=use_cache,
            refresh_cache=args.refresh_cache,
        )

        if train_features.empty or test_features.empty:
            print(f'  Skipping {test_year}: empty train or test features after filtering.')
            continue

        available_features = [col for col in FEATURE_COLUMNS if col in train_features.columns]
        X_train = train_features[available_features]
        y_train = train_features['result']
        X_test = test_features[available_features]
        y_test = test_features['result']
        top_10_train_mask = (train_features['player1_rank'] <= 10) | (train_features['player2_rank'] <= 10)
        top_10_test_mask = (test_features['player1_rank'] <= 10) | (test_features['player2_rank'] <= 10)

        if args.probability_model == 'residual':
            results = train_residual_models(
                train_features,
                test_features,
                available_features,
                top_10_train_mask,
                top_10_test_mask,
            )
            predictions = results.pop('predictions')
        else:
            trainer = ModelTrainer(model_dir=str(output_dir / '_models'))
            results = trainer.train_separate_models(
                X_train,
                y_train,
                top_10_train_mask,
                X_test,
                y_test,
                top_10_test_mask,
            )
            top_10_bundle = results['top_10']
            other_bundle = results['other']
            predictions = score_matches(test_features, available_features, top_10_bundle, other_bundle)
        predictions = add_model_market_edges(predictions)
        predictions['test_year'] = test_year
        predictions['probability_model'] = args.probability_model
        fold_predictions.append(predictions)

        for model_type, result in results.items():
            row = {'test_year': test_year, 'model_type': model_type}
            row.update(result['metrics'])
            fold_metrics.append(row)

    if not fold_predictions:
        raise ValueError('No walk-forward predictions were generated.')

    print('[3/4] Combining fold predictions...')
    predictions = pd.concat(fold_predictions, ignore_index=True)
    predictions, edge_trust_summary = add_edge_trust(
        predictions,
        edge_threshold=args.edge_threshold,
        stake=args.stake,
        friction_bps=args.friction_bps,
    )
    predictions_path = output_dir / 'predictions.csv'
    metrics_path = output_dir / 'fold_metrics.csv'
    predictions.to_csv(predictions_path, index=False)
    pd.DataFrame(fold_metrics).to_csv(metrics_path, index=False)
    edge_trust_summary.to_csv(output_dir / 'edge_trust_summary.csv', index=False)

    matched = predictions[['player1_odds', 'player2_odds']].notna().all(axis=1).sum()
    print(f'Matched odds for {matched}/{len(predictions)} predictions.')

    print('[4/4] Running PnL summaries...')
    backtester = PnLBacktester(
        edge_threshold=args.edge_threshold,
        stake=args.stake,
        friction_bps=args.friction_bps,
    )
    bets = backtester.generate_bets(
        predictions,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        model_type=args.model_type,
        exclude_slams=args.exclude_slams,
        surface=args.surface,
    )
    summary = backtester.summarize(bets)

    bets.to_csv(output_dir / 'bets.csv', index=False)
    pd.DataFrame([summary]).to_csv(output_dir / 'summary.csv', index=False)
    calibration = calibration_report(predictions)
    summarize_threshold_grid(
        predictions,
        thresholds=sorted(set(args.threshold_grid + [args.edge_threshold])),
        stake=args.stake,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        model_type=args.model_type,
        exclude_slams=args.exclude_slams,
        surface=args.surface,
    ).to_csv(output_dir / 'threshold_summary.csv', index=False)
    summarize_by_model_type(bets, stake=args.stake).to_csv(output_dir / 'model_type_summary.csv', index=False)
    summarize_by_year(bets, stake=args.stake).to_csv(output_dir / 'yearly_summary.csv', index=False)
    summarize_segments(bets, stake=args.stake).to_csv(output_dir / 'segment_summary.csv', index=False)
    calibration.to_csv(output_dir / 'calibration_summary.csv', index=False)
    probability_bucket_roi(bets).to_csv(output_dir / 'probability_bucket_roi.csv', index=False)
    strategy_candidate_report(
        predictions,
        thresholds=sorted(set(args.threshold_grid + [args.edge_threshold])),
        stake=args.stake,
        friction_bps=args.friction_bps,
    ).to_csv(output_dir / 'strategy_candidates.csv', index=False)
    segment_stability_report(bets, stake=args.stake).to_csv(output_dir / 'segment_stability.csv', index=False)
    gate_summary(bets, calibration=calibration, stake=args.stake).to_csv(output_dir / 'gate_summary.csv', index=False)
    nested_walk_forward_strategy_selection(
        predictions,
        thresholds=sorted(set(args.threshold_grid + [args.edge_threshold])),
        stake=args.stake,
        friction_bps=args.friction_bps,
    ).to_csv(output_dir / 'nested_strategy_summary.csv', index=False)

    print_summary(summary)
    print(f'Wrote walk-forward reports to: {output_dir}')


def align_matches_with_cache(
    matches: pd.DataFrame,
    odds: pd.DataFrame,
    odds_loader: OddsLoader,
    cache_dir: Path,
    cache_key: str,
    use_cache: bool,
    refresh_cache: bool,
) -> pd.DataFrame:
    cache_file = cache_dir / f'aligned_matches_{cache_key}.pkl'
    if use_cache and not refresh_cache and cache_file.exists():
        print(f'  Loaded aligned matches cache: {cache_file}')
        return pd.read_pickle(cache_file)

    aligned = odds_loader.align_match_dates(matches, odds)
    if use_cache:
        write_pickle_cache(aligned, cache_file)
        print(f'  Wrote aligned matches cache: {cache_file}')
    return aligned


def build_fold_features_cached(
    matches: pd.DataFrame,
    odds: pd.DataFrame,
    odds_loader: OddsLoader,
    train_start_year: int,
    test_year: int,
    alignment_key: str,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cache_key = fold_cache_key(
        alignment_key=alignment_key,
        train_start_year=train_start_year,
        test_year=test_year,
    )
    cache_file = cache_dir / f'fold_features_{cache_key}.pkl'
    if use_cache and not refresh_cache and cache_file.exists():
        print(f'  Loaded fold feature cache: {cache_file.name}')
        cached = pd.read_pickle(cache_file)
        return cached['train_features'], cached['test_features']

    train_features, test_features = build_fold_features(
        matches=matches,
        odds=odds,
        odds_loader=odds_loader,
        train_start_year=train_start_year,
        test_year=test_year,
    )
    if use_cache:
        write_pickle_cache(
            {'train_features': train_features, 'test_features': test_features},
            cache_file,
        )
        print(f'  Wrote fold feature cache: {cache_file.name}')
    return train_features, test_features


def alignment_cache_key(
    data_dir: str,
    end_year: int,
    odds_files: list[str],
    odds_columns: Tuple[str, str] | None,
) -> str:
    payload = {
        'cache_version': CACHE_VERSION,
        'aligned_cache_version': ALIGNED_CACHE_VERSION,
        'data_files': data_file_signatures(data_dir, end_year),
        'end_year': end_year,
        'odds_files': [file_signature(path) for path in odds_files],
        'odds_columns': odds_columns,
    }
    return stable_hash(payload)


def fold_cache_key(
    alignment_key: str,
    train_start_year: int,
    test_year: int,
) -> str:
    payload = {
        'cache_version': CACHE_VERSION,
        'fold_feature_cache_version': FOLD_FEATURE_CACHE_VERSION,
        'feature_columns': FEATURE_COLUMNS,
        'alignment_key': alignment_key,
        'train_start_year': train_start_year,
        'test_year': test_year,
        'train_random_state': 42 + test_year,
        'test_random_state': 43 + test_year,
    }
    return stable_hash(payload)


def data_file_signatures(data_dir: str, end_year: int) -> list[dict]:
    directory = Path(data_dir)
    signatures = []
    for year in range(2000, end_year):
        path = directory / f'atp_matches_{year}.csv'
        signatures.append(file_signature(path))
    return signatures


def file_signature(path: str | Path) -> dict:
    resolved = Path(path).resolve()
    if not resolved.exists():
        return {
            'path': str(resolved),
            'exists': False,
        }
    stat = resolved.stat()
    return {
        'path': str(resolved),
        'exists': True,
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:20]


def write_pickle_cache(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    pd.to_pickle(value, temp_path)
    temp_path.replace(path)


def build_fold_features(
    matches: pd.DataFrame,
    odds: pd.DataFrame,
    odds_loader: OddsLoader,
    train_start_year: int,
    test_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    date_series = pd.to_datetime(matches['match_date'] if 'match_date' in matches.columns else matches['tourney_date'])
    warmup = matches[date_series.dt.year < train_start_year]
    train = matches[
        (date_series.dt.year >= train_start_year) &
        (date_series.dt.year < test_year)
    ]
    test = matches[date_series.dt.year == test_year]

    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    betting_feature_engineer = BettingFeatureEngineer()
    elo_tracker = EloTracker()
    data_loader.initialize_players(warmup)
    elo_tracker.initialize(warmup)

    train_features = feature_engineer.create_rolling_features(
        train,
        data_loader.players,
        data_loader,
        betting_feature_engineer=betting_feature_engineer,
        elo_tracker=elo_tracker,
        random_state=42 + test_year,
    )
    test_features = feature_engineer.create_rolling_features(
        test,
        data_loader.players,
        data_loader,
        betting_feature_engineer=betting_feature_engineer,
        elo_tracker=elo_tracker,
        random_state=43 + test_year,
    )

    train_features = odds_loader.attach_odds(train_features, odds)
    test_features = odds_loader.attach_odds(test_features, odds)

    player_match_count = train_features['player1'].value_counts().add(
        train_features['player2'].value_counts(),
        fill_value=0
    )
    eligible_players = player_match_count[player_match_count >= 10].index
    train_features = train_features[
        train_features['player1'].isin(eligible_players) &
        train_features['player2'].isin(eligible_players)
    ].copy()
    test_features = test_features[
        test_features['player1'].isin(eligible_players) &
        test_features['player2'].isin(eligible_players)
    ].copy()
    return train_features, test_features


def train_residual_models(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_columns: list[str],
    top_10_train_mask: pd.Series,
    top_10_test_mask: pd.Series,
) -> dict:
    predictions = test_features.copy()
    predictions['top_10_match'] = top_10_test_mask
    predictions['model_type'] = 'other'
    predictions.loc[top_10_test_mask, 'model_type'] = 'top_10'
    predictions['player1_probability'] = pd.NA
    predictions['player2_probability'] = pd.NA
    results = {}

    for model_name, train_mask, test_mask in [
        ('top_10', top_10_train_mask, top_10_test_mask),
        ('other', ~top_10_train_mask, ~top_10_test_mask),
    ]:
        train_mask = train_mask & train_features['player1_market_probability_novig'].notna()
        test_mask = test_mask & test_features['player1_market_probability_novig'].notna()
        if not train_mask.any() or not test_mask.any():
            continue
        bundle = fit_residual_model(
            train_features.loc[train_mask, feature_columns],
            (train_features.loc[train_mask, 'result'] == 1).astype(float) -
            train_features.loc[train_mask, 'player1_market_probability_novig'].astype(float),
        )
        residual = predict_residual(bundle, test_features.loc[test_mask, feature_columns])
        baseline = test_features.loc[test_mask, 'player1_market_probability_novig'].astype(float)
        player1_probability = (baseline + residual).clip(0.02, 0.98)
        predictions.loc[test_mask, 'player1_probability'] = player1_probability
        predictions.loc[test_mask, 'player2_probability'] = 1 - player1_probability
        results[model_name] = {
            'metrics': probability_metrics(
                test_features.loc[test_mask, 'result'],
                player1_probability.astype(float),
            )
        }

    predictions['winner_probability'] = predictions.apply(
        lambda row: row['player1_probability'] if row['player1'] == row['winner_name']
        else row['player2_probability'],
        axis=1
    )
    predictions['predicted_winner'] = predictions.apply(predicted_winner_from_probability, axis=1)
    predictions['prediction_correct'] = predictions['predicted_winner'] == predictions['winner_name']
    results['predictions'] = predictions
    return results


def predicted_winner_from_probability(row: pd.Series):
    if pd.isna(row['player1_probability']) or pd.isna(row['player2_probability']):
        return pd.NA
    return row['player1'] if row['player1_probability'] >= row['player2_probability'] else row['player2']


def fit_residual_model(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    valid = y_train.notna()
    X_train = X_train.loc[valid]
    y_train = y_train.loc[valid]
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imputed)
    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=42,
    )
    model.fit(X_scaled, y_train)
    return {'model': model, 'imputer': imputer, 'scaler': scaler}


def predict_residual(bundle: dict, X: pd.DataFrame) -> pd.Series:
    X_imputed = bundle['imputer'].transform(X)
    X_scaled = bundle['scaler'].transform(X_imputed)
    return pd.Series(bundle['model'].predict(X_scaled), index=X.index)


def probability_metrics(y_true_result: pd.Series, player1_probability: pd.Series) -> dict:
    y_binary = (y_true_result == 1).astype(int)
    proba = pd.DataFrame({1: player1_probability, 2: 1 - player1_probability}, index=y_binary.index)
    predicted = (player1_probability >= 0.5).map({True: 1, False: 2})
    return {
        'accuracy': accuracy_score(y_true_result, predicted),
        'precision': accuracy_score(y_true_result, predicted),
        'recall': accuracy_score(y_true_result, predicted),
        'f1_score': accuracy_score(y_true_result, predicted),
        'log_loss': log_loss(y_true_result, proba[[1, 2]]),
        'brier_score': brier_score_loss(y_binary, player1_probability),
    }


def calibration_report(predictions: pd.DataFrame, buckets: int = 10) -> pd.DataFrame:
    rows = []
    priced = predictions.dropna(subset=['player1_probability', 'player1_market_probability_novig']).copy()
    for side in ['player1', 'player2']:
        frame = priced.copy()
        frame['probability'] = frame[f'{side}_probability'].astype(float)
        frame['market_probability'] = frame[f'{side}_market_probability_novig'].astype(float)
        frame['won'] = frame['result'] == (1 if side == 'player1' else 2)
        frame['bucket'] = pd.cut(frame['probability'], bins=buckets, labels=False, include_lowest=True)
        for bucket, group in frame.groupby('bucket', dropna=True):
            rows.append({
                'side': side,
                'bucket': int(bucket),
                'bets': int(len(group)),
                'average_probability': float(group['probability'].mean()),
                'actual_win_rate': float(group['won'].mean()),
                'average_market_probability': float(group['market_probability'].mean()),
                'average_residual_edge': float((group['probability'] - group['market_probability']).mean()),
                'ece_component': float(len(group) / len(frame) * abs(group['won'].mean() - group['probability'].mean())),
            })
    return pd.DataFrame(rows)


def probability_bucket_roi(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    frame = bets.copy()
    frame['probability_bucket'] = pd.cut(frame['model_probability'], bins=10, labels=False, include_lowest=True)
    rows = []
    for bucket, group in frame.groupby('probability_bucket', dropna=True):
        summary = PnLBacktester(stake=float(group['stake'].iloc[0])).summarize(group)
        summary['probability_bucket'] = int(bucket)
        summary['min_probability'] = float(group['model_probability'].min())
        summary['max_probability'] = float(group['model_probability'].max())
        rows.append(summary)
    return pd.DataFrame(rows)


def nested_walk_forward_strategy_selection(
    predictions: pd.DataFrame,
    thresholds: list[float],
    stake: float,
    friction_bps: float = 0,
    min_prior_bets: int = 150,
) -> pd.DataFrame:
    candidates = []
    for threshold in thresholds:
        for model_type in [None, 'top_10', 'other']:
            for max_odds in [3.0, 5.0, 10.0, None]:
                for exclude_slams in [False, True]:
                    for surface in [None, 'Hard', 'Clay', 'Grass']:
                        candidates.append({
                            'edge_threshold': threshold,
                            'model_type': model_type,
                            'max_odds': max_odds,
                            'exclude_slams': exclude_slams,
                            'surface': surface,
                        })

    years = sorted(predictions['test_year'].dropna().unique())
    rows = []
    for year in years:
        prior = predictions[predictions['test_year'] < year]
        current = predictions[predictions['test_year'] == year]
        if prior.empty or current.empty:
            continue
        scored = []
        for config in candidates:
            backtester = PnLBacktester(edge_threshold=config['edge_threshold'], stake=stake, friction_bps=friction_bps)
            prior_bets = backtester.generate_bets(
                prior,
                max_odds=config['max_odds'],
                model_type=config['model_type'],
                exclude_slams=config['exclude_slams'],
                surface=config['surface'],
            )
            summary = backtester.summarize(prior_bets)
            if summary['bets'] >= min_prior_bets:
                scored.append((summary['friction_adjusted_roi'], summary['bets'], config))
        if not scored:
            continue
        _, _, best_config = max(scored, key=lambda item: (item[0], item[1]))
        backtester = PnLBacktester(edge_threshold=best_config['edge_threshold'], stake=stake, friction_bps=friction_bps)
        test_bets = backtester.generate_bets(
            current,
            max_odds=best_config['max_odds'],
            model_type=best_config['model_type'],
            exclude_slams=best_config['exclude_slams'],
            surface=best_config['surface'],
        )
        summary = backtester.summarize(test_bets)
        summary.update(best_config)
        summary['test_year'] = year
        rows.append(summary)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main()
