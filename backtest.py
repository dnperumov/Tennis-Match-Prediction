#!/usr/bin/env python3
"""
No-look-ahead probability scoring and betting PnL backtest.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.backtesting import OddsLoader, PnLBacktester
from tennis_ml.data import DataLoader
from tennis_ml.features import BettingFeatureEngineer, EloTracker, FeatureEngineer
from tennis_ml.features.columns import FEATURE_COLUMNS
from tennis_ml.models import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description='Tennis model PnL backtest')
    parser.add_argument('--data-dir', default='tennis_datav2')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--years', type=int, nargs=2, default=[2000, 2027],
                        help='Year range for data (start end-exclusive)')
    parser.add_argument('--odds-file', nargs='+',
                        help='One or more CSV/XLS/XLSX odds files with Winner, Loser, Date, and odds columns')
    parser.add_argument('--winner-odds-col', help='Optional winner odds column override, e.g. B365W')
    parser.add_argument('--loser-odds-col', help='Optional loser odds column override, e.g. B365L')
    parser.add_argument('--edge-threshold', type=float, default=0.03,
                        help='Minimum expected value per $1 staked, e.g. 0.03 for 3 cents')
    parser.add_argument('--threshold-grid', type=float, nargs='*',
                        default=[0.01, 0.02, 0.03, 0.05, 0.075, 0.10],
                        help='EV thresholds to summarize after the main backtest')
    parser.add_argument('--stake', type=float, default=1.0)
    parser.add_argument('--min-odds', type=float)
    parser.add_argument('--max-odds', type=float)
    parser.add_argument('--model-type', choices=['top_10', 'other'])
    parser.add_argument('--exclude-slams', action='store_true')
    parser.add_argument('--surface', choices=['Hard', 'Clay', 'Grass'])
    parser.add_argument('--output-dir', default='backtest_results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    odds_loader = None
    odds = None
    if args.odds_file:
        print('[1/4] Loading historical odds...')
        odds_columns = None
        if args.winner_odds_col or args.loser_odds_col:
            if not args.winner_odds_col or not args.loser_odds_col:
                raise ValueError('Pass both --winner-odds-col and --loser-odds-col, or neither.')
            odds_columns = (args.winner_odds_col, args.loser_odds_col)

        odds_loader = OddsLoader()
        odds = pd.concat(
            [odds_loader.load(path, preferred_columns=odds_columns) for path in args.odds_file],
            ignore_index=True
        )

    print('[2/4] Building no-look-ahead test features...')
    features = build_no_lookahead_features(args.data_dir, args.years, odds_loader=odds_loader, odds=odds)

    if args.odds_file:
        print('[2/4] Matching historical odds...')
        features = odds_loader.attach_odds(features, odds)

    available_features = [col for col in FEATURE_COLUMNS if col in features.columns]

    print('[3/4] Loading models and scoring matches...')
    trainer = ModelTrainer(model_dir=args.model_dir)
    top_10_model = trainer.load_model('top_10')
    other_model = trainer.load_model('other')
    predictions = score_matches(features, available_features, top_10_model, other_model)
    predictions = add_model_market_edges(predictions)
    predictions_path = output_dir / 'predictions.csv'
    predictions.to_csv(predictions_path, index=False)
    print(f'Wrote scored matches: {predictions_path}')

    if not args.odds_file:
        print('No odds file supplied, so PnL was not calculated.')
        print('Re-run with --odds-file path/to/odds.csv to simulate bets.')
        return

    priced_predictions = predictions
    priced_path = output_dir / 'predictions_with_odds.csv'
    priced_predictions.to_csv(priced_path, index=False)
    matched = priced_predictions[['player1_odds', 'player2_odds']].notna().all(axis=1).sum()
    print(f'Matched odds for {matched}/{len(priced_predictions)} matches.')

    print('[4/4] Running PnL backtest...')
    backtester = PnLBacktester(edge_threshold=args.edge_threshold, stake=args.stake)
    bets = backtester.generate_bets(
        priced_predictions,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        model_type=args.model_type,
        exclude_slams=args.exclude_slams,
        surface=args.surface
    )
    summary = backtester.summarize(bets)
    bets_path = output_dir / 'bets.csv'
    summary_path = output_dir / 'summary.csv'
    threshold_summary_path = output_dir / 'threshold_summary.csv'
    model_summary_path = output_dir / 'model_type_summary.csv'
    yearly_summary_path = output_dir / 'yearly_summary.csv'
    segment_summary_path = output_dir / 'segment_summary.csv'
    bets.to_csv(bets_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    threshold_summary = summarize_threshold_grid(
        priced_predictions,
        thresholds=sorted(set(args.threshold_grid + [args.edge_threshold])),
        stake=args.stake,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        model_type=args.model_type,
        exclude_slams=args.exclude_slams,
        surface=args.surface
    )
    threshold_summary.to_csv(threshold_summary_path, index=False)
    model_type_summary = summarize_by_model_type(bets, stake=args.stake)
    model_type_summary.to_csv(model_summary_path, index=False)
    yearly_summary = summarize_by_year(bets, stake=args.stake)
    yearly_summary.to_csv(yearly_summary_path, index=False)
    segment_summary = summarize_segments(bets, stake=args.stake)
    segment_summary.to_csv(segment_summary_path, index=False)

    print_summary(summary)
    print(f'Wrote bets: {bets_path}')
    print(f'Wrote summary: {summary_path}')
    print(f'Wrote threshold summary: {threshold_summary_path}')
    print(f'Wrote model type summary: {model_summary_path}')
    print(f'Wrote yearly summary: {yearly_summary_path}')
    print(f'Wrote segment summary: {segment_summary_path}')


def build_no_lookahead_features(data_dir: str, years: List[int], odds_loader=None, odds=None) -> pd.DataFrame:
    data_loader = DataLoader(data_dir=data_dir)
    feature_engineer = FeatureEngineer()
    betting_feature_engineer = BettingFeatureEngineer()
    elo_tracker = EloTracker()

    atp_matches = data_loader.load_data(years=range(years[0], years[1]))
    if odds_loader is not None and odds is not None:
        atp_matches = odds_loader.align_match_dates(atp_matches, odds)
    initial_50, next_25, final_25 = data_loader.split_data_chronologically(atp_matches)
    data_loader.initialize_players(initial_50)
    elo_tracker.initialize(initial_50)

    train_features = feature_engineer.create_rolling_features(
        next_25,
        data_loader.players,
        data_loader,
        betting_feature_engineer=betting_feature_engineer,
        elo_tracker=elo_tracker,
        random_state=42
    )
    test_features = feature_engineer.create_rolling_features(
        final_25,
        data_loader.players,
        data_loader,
        betting_feature_engineer=betting_feature_engineer,
        elo_tracker=elo_tracker,
        random_state=43
    )

    player_match_count = train_features['player1'].value_counts().add(
        train_features['player2'].value_counts(),
        fill_value=0
    )
    eligible_players = player_match_count[player_match_count >= 10].index
    return test_features[
        test_features['player1'].isin(eligible_players) &
        test_features['player2'].isin(eligible_players)
    ].copy()


def score_matches(
    features: pd.DataFrame,
    feature_columns: list[str],
    top_10_model: dict,
    other_model: dict
) -> pd.DataFrame:
    predictions = features.copy()
    predictions['top_10_match'] = (
        (predictions['player1_rank'] <= 10) |
        (predictions['player2_rank'] <= 10)
    )
    predictions['model_type'] = 'other'
    predictions.loc[predictions['top_10_match'], 'model_type'] = 'top_10'
    predictions['player1_probability'] = pd.NA
    predictions['player2_probability'] = pd.NA

    for model_name, bundle in [('top_10', top_10_model), ('other', other_model)]:
        mask = predictions['model_type'] == model_name
        if not mask.any():
            continue

        model_features = bundle['features']
        for column in model_features:
            if column not in predictions.columns:
                predictions[column] = pd.NA
        X = predictions.loc[mask, model_features]
        if bundle.get('imputer') is not None:
            X = pd.DataFrame(bundle['imputer'].transform(X), columns=model_features, index=X.index)
        else:
            X = X.fillna(0)
        X_scaled = bundle['scaler'].transform(X)
        probabilities = bundle['model'].predict_proba(X_scaled)
        probability_frame = pd.DataFrame(
            probabilities,
            columns=bundle['model'].classes_,
            index=X.index
        )
        predictions.loc[mask, 'player1_probability'] = probability_frame[1]
        predictions.loc[mask, 'player2_probability'] = probability_frame[2]

    predictions['winner_probability'] = predictions.apply(
        lambda row: row['player1_probability'] if row['player1'] == row['winner_name']
        else row['player2_probability'],
        axis=1
    )
    predictions['predicted_winner'] = predictions.apply(
        lambda row: row['player1'] if row['player1_probability'] >= row['player2_probability']
        else row['player2'],
        axis=1
    )
    predictions['prediction_correct'] = predictions['predicted_winner'] == predictions['winner_name']
    return predictions


def add_model_market_edges(predictions: pd.DataFrame) -> pd.DataFrame:
    if {
        'player1_probability',
        'player2_probability',
        'player1_market_probability_novig',
        'player2_market_probability_novig'
    }.issubset(predictions.columns):
        predictions['player1_model_market_edge'] = (
            predictions['player1_probability'] - predictions['player1_market_probability_novig']
        )
        predictions['player2_model_market_edge'] = (
            predictions['player2_probability'] - predictions['player2_market_probability_novig']
        )
    return predictions


def print_summary(summary: dict):
    print('\nBACKTEST SUMMARY')
    print('=' * 60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f'{key}: {value:.4f}')
        else:
            print(f'{key}: {value}')


def summarize_threshold_grid(
    priced_predictions: pd.DataFrame,
    thresholds: List[float],
    stake: float,
    min_odds: float = None,
    max_odds: float = None,
    model_type: str = None,
    exclude_slams: bool = False,
    surface: str = None
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        backtester = PnLBacktester(edge_threshold=threshold, stake=stake)
        summary = backtester.summarize(
            backtester.generate_bets(
                priced_predictions,
                min_odds=min_odds,
                max_odds=max_odds,
                model_type=model_type,
                exclude_slams=exclude_slams,
                surface=surface
            )
        )
        summary['edge_threshold'] = threshold
        rows.append(summary)
    return pd.DataFrame(rows)


def summarize_by_model_type(bets: pd.DataFrame, stake: float) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    rows = []
    for model_type, group in bets.groupby('model_type'):
        summary = PnLBacktester(stake=stake).summarize(group)
        summary['model_type'] = model_type
        rows.append(summary)
    return pd.DataFrame(rows)


def summarize_by_year(bets: pd.DataFrame, stake: float) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    dated = bets.copy()
    date_column = 'match_date' if 'match_date' in dated.columns else 'tourney_date'
    dated['year'] = pd.to_datetime(dated[date_column]).dt.year
    rows = []
    for year, group in dated.groupby('year'):
        summary = PnLBacktester(stake=stake).summarize(group)
        summary['year'] = year
        rows.append(summary)
    return pd.DataFrame(rows)


def summarize_segments(bets: pd.DataFrame, stake: float) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    rows = []
    for column in ['model_type', 'surface', 'tournament_importance']:
        for value, group in bets.groupby(column, dropna=False):
            summary = PnLBacktester(stake=stake).summarize(group)
            summary['segment'] = column
            summary['value'] = value
            rows.append(summary)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main()
