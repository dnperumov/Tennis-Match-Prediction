#!/usr/bin/env python3
"""
Main ML Pipeline for Tennis Match Prediction

This script orchestrates the complete machine learning pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Model training
4. Model evaluation
5. Model saving
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.data import DataLoader
from tennis_ml.backtesting import OddsLoader
from tennis_ml.features import FeatureEngineer, BettingFeatureEngineer, EloTracker
from tennis_ml.features.columns import FEATURE_COLUMNS
from tennis_ml.models import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description='Tennis Match Prediction ML Pipeline')
    parser.add_argument('--data-dir', type=str, default='tennis_datav2',
                       help='Directory containing ATP match data')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--years', type=int, nargs=2, default=[2000, 2027],
                       help='Year range for data (start end-exclusive)')
    parser.add_argument('--train', action='store_true',
                       help='Train new models')
    parser.add_argument('--predict', nargs=4, metavar=('PLAYER1', 'PLAYER2', 'SURFACE', 'TOURNAMENT'),
                       help='Make a prediction: player1 player2 surface tournament_importance')
    parser.add_argument('--odds-file', nargs='+',
                       help='Optional odds files for market-aware training features')
    
    args = parser.parse_args()
    
    # Initialize components
    data_loader = DataLoader(data_dir=args.data_dir)
    feature_engineer = FeatureEngineer()
    betting_feature_engineer = BettingFeatureEngineer()
    elo_tracker = EloTracker()
    trainer = ModelTrainer(model_dir=args.model_dir)
    
    if args.train:
        print("=" * 60)
        print("TENNIS MATCH PREDICTION ML PIPELINE")
        print("=" * 60)
        
        # Step 1: Load data
        print("\n[1/5] Loading data...")
        years = range(args.years[0], args.years[1])
        atp_matches = data_loader.load_data(years=years)
        if args.odds_file:
            odds_loader = OddsLoader()
            odds = pd.concat(
                [odds_loader.load(path) for path in args.odds_file],
                ignore_index=True
            )
            atp_matches = odds_loader.align_match_dates(atp_matches, odds)
        else:
            odds_loader = None
            odds = None
        print(f"Loaded {len(atp_matches)} matches from {args.years[0]}-{args.years[1]-1}")
        
        # Step 2: Split data chronologically
        print("\n[2/5] Splitting data chronologically...")
        initial_50, next_25, final_25 = data_loader.split_data_chronologically(atp_matches)
        print(f"Initial 50%: {len(initial_50)} matches")
        print(f"Next 25%: {len(next_25)} matches")
        print(f"Final 25%: {len(final_25)} matches")
        
        # Step 3: Initialize player statistics
        print("\n[3/5] Initializing player statistics...")
        data_loader.initialize_players(initial_50)
        elo_tracker.initialize(initial_50)
        print(f"Initialized {len(data_loader.players)} players")
        
        # Step 4: Feature engineering
        print("\n[4/5] Engineering features...")
        
        # Create features in match order. Each match is featurized before its
        # result is used to update player histories, which prevents look-ahead.
        next_25_features = feature_engineer.create_rolling_features(
            next_25,
            data_loader.players,
            data_loader,
            betting_feature_engineer=betting_feature_engineer,
            elo_tracker=elo_tracker,
            random_state=42
        )
        final_25_features = feature_engineer.create_rolling_features(
            final_25,
            data_loader.players,
            data_loader,
            betting_feature_engineer=betting_feature_engineer,
            elo_tracker=elo_tracker,
            random_state=43
        )

        if args.odds_file:
            print("Attaching market odds features...")
            next_25_features = odds_loader.attach_odds(next_25_features, odds)
            final_25_features = odds_loader.attach_odds(final_25_features, odds)
        
        # Filter players with minimum matches
        player_match_count = (
            next_25_features['player1'].value_counts().add(
                next_25_features['player2'].value_counts(),
                fill_value=0
            )
        )
        eligible_players = player_match_count[player_match_count >= 10].index
        next_25_features = next_25_features[
            next_25_features['player1'].isin(eligible_players) &
            next_25_features['player2'].isin(eligible_players)
        ]
        final_25_features = final_25_features[
            final_25_features['player1'].isin(eligible_players) &
            final_25_features['player2'].isin(eligible_players)
        ]
        
        available_features = [col for col in FEATURE_COLUMNS if col in next_25_features.columns]
        print(f"Using {len(available_features)} features")
        
        # Prepare training data
        top_10_train_mask = (next_25_features['player1_rank'] <= 10) | (next_25_features['player2_rank'] <= 10)
        top_10_val_mask = (final_25_features['player1_rank'] <= 10) | (final_25_features['player2_rank'] <= 10)
        X_train = next_25_features[available_features]
        y_train = next_25_features['result']
        X_val = final_25_features[available_features]
        y_val = final_25_features['result']
        print(f"Training matches after filter: {len(X_train)}")
        print(f"No-look-ahead test matches after filter: {len(X_val)}")
        
        # Step 5: Train models
        print("\n[5/5] Training models...")
        results = trainer.train_separate_models(
            X_train, y_train, top_10_train_mask, X_val, y_val, top_10_val_mask
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)
        print("\nTop 10 Players Model:")
        for metric, value in results['top_10']['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nOther Players Model:")
        for metric, value in results['other']['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        # Save models
        print("\nSaving models...")
        trainer.save_model('top_10', results['top_10']['model'], 
                          results['top_10']['scaler'], results['top_10']['features'],
                          results['top_10'].get('imputer'))
        trainer.save_model('other', results['other']['model'],
                          results['other']['scaler'], results['other']['features'],
                          results['other'].get('imputer'))
        
        print("\n✅ Pipeline completed successfully!")
        print(f"Models saved to: {args.model_dir}/")
    
    elif args.predict:
        from tennis_ml.models import MatchPredictor
        
        player1_name, player2_name, surface, tournament = args.predict
        tournament_importance = int(tournament)
        
        print(f"\nMaking prediction: {player1_name} vs {player2_name}")
        print(f"Surface: {surface}, Tournament: {tournament_importance}")
        
        # Load models (simplified - in production, you'd load from saved files)
        print("Note: For predictions, models must be trained first.")
        print("Run with --train flag to train models.")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
