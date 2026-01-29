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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tennis_ml.data import DataLoader
from tennis_ml.features import FeatureEngineer, BettingFeatureEngineer
from tennis_ml.models import ModelTrainer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description='Tennis Match Prediction ML Pipeline')
    parser.add_argument('--data-dir', type=str, default='tennis_datav2',
                       help='Directory containing ATP match data')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--years', type=int, nargs=2, default=[2000, 2025],
                       help='Year range for data (start end)')
    parser.add_argument('--train', action='store_true',
                       help='Train new models')
    parser.add_argument('--predict', nargs=4, metavar=('PLAYER1', 'PLAYER2', 'SURFACE', 'TOURNAMENT'),
                       help='Make a prediction: player1 player2 surface tournament_importance')
    
    args = parser.parse_args()
    
    # Initialize components
    data_loader = DataLoader(data_dir=args.data_dir)
    feature_engineer = FeatureEngineer()
    betting_feature_engineer = BettingFeatureEngineer()
    trainer = ModelTrainer(model_dir=args.model_dir)
    
    if args.train:
        print("=" * 60)
        print("TENNIS MATCH PREDICTION ML PIPELINE")
        print("=" * 60)
        
        # Step 1: Load data
        print("\n[1/5] Loading data...")
        years = range(args.years[0], args.years[1])
        atp_matches = data_loader.load_data(years=years)
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
        print(f"Initialized {len(data_loader.players)} players")
        
        # Step 4: Feature engineering
        print("\n[4/5] Engineering features...")
        
        # Create player vs player datasets
        next_25_pvp = feature_engineer.create_player_vs_player_dataset(next_25)
        final_25_pvp = feature_engineer.create_player_vs_player_dataset(final_25)
        
        # Update players with next_25 data
        data_loader.update_players_incremental(next_25)
        
        # Create features
        next_25_features = feature_engineer.create_features(
            next_25_pvp, data_loader.players
        )
        
        # Add betting features
        next_25_features = betting_feature_engineer.add_betting_features(
            next_25_features, data_loader.players
        )
        
        # Filter players with minimum matches
        player_match_count = (
            next_25_features['player1'].value_counts() +
            next_25_features['player2'].value_counts()
        )
        eligible_players = player_match_count[player_match_count >= 10].index
        next_25_features = next_25_features[
            next_25_features['player1'].isin(eligible_players) &
            next_25_features['player2'].isin(eligible_players)
        ]
        
        # Define features
        feature_columns = [
            'rank_diff', 'log_rank_diff', 'top_10_vs_not',
            'age_diff', 'young_vs_old', 'seed_diff', 'seeded_vs_unseeded',
            'height_diff', 'tall_vs_short', 'same_hand', 'left_vs_right',
            'player1_last_5_win_percentage', 'player2_last_5_win_percentage',
            'player1_last_10_win_percentage', 'player2_last_10_win_percentage',
            'player1_surface_last_10_win_percentage', 'player2_surface_last_10_win_percentage',
            'player1_preferred_surface', 'player2_preferred_surface',
            'player1_surface_match', 'player2_surface_match',
            'surface_preference_diff', 'head_to_head_wins_p1', 'head_to_head_wins_p2',
            'surface_encoded', 'tournament_importance',
            'player1_momentum', 'player2_momentum',
            'player1_confidence', 'player2_confidence',
            'player1_surface_advantage', 'player2_surface_advantage'
        ]
        
        available_features = [col for col in feature_columns if col in next_25_features.columns]
        print(f"Using {len(available_features)} features")
        
        # Prepare training data
        X = next_25_features[available_features].fillna(0)
        y = next_25_features['result']
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create top 10 mask
        top_10_mask = (X_train['player1_rank'] <= 10) | (X_train['player2_rank'] <= 10)
        
        # Step 5: Train models
        print("\n[5/5] Training models...")
        results = trainer.train_separate_models(
            X_train, y_train, top_10_mask, X_val, y_val
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

