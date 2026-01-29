"""
Model training for tennis match prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, brier_score_loss
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib
import os


class ModelTrainer:
    """Trains and evaluates tennis match prediction models."""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_lists = {}
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def train_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame = None, y_val: pd.Series = None,
                            use_calibration: bool = True) -> Dict[str, Any]:
        """
        Train ensemble model with multiple algorithms.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            use_calibration: Whether to calibrate probabilities
            
        Returns:
            Dictionary with trained model and metrics
        """
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        
        # Create base models
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('lr', lr_model)
            ],
            voting='soft'
        )
        
        # Train
        ensemble.fit(X_train_scaled, y_train)
        
        # Calibrate if requested
        if use_calibration:
            calibrated = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
            calibrated.fit(X_train_scaled, y_train)
            model = calibrated
        else:
            model = ensemble
        
        # Evaluate if validation data provided
        metrics = {}
        if X_val is not None and y_val is not None:
            X_val_imputed = pd.DataFrame(
                imputer.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            X_val_scaled = scaler.transform(X_val_imputed)
            
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted'),
                'recall': recall_score(y_val, y_pred, average='weighted'),
                'f1_score': f1_score(y_val, y_pred, average='weighted'),
                'log_loss': log_loss(y_val, y_pred_proba),
                'brier_score': brier_score_loss(y_val, y_pred_proba[:, 1])
            }
        
        # Store components
        self.models['ensemble'] = model
        self.scalers['ensemble'] = scaler
        self.feature_lists['ensemble'] = list(X_train.columns)
        
        return {
            'model': model,
            'scaler': scaler,
            'imputer': imputer,
            'features': list(X_train.columns),
            'metrics': metrics
        }
    
    def train_separate_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                             top_10_mask: pd.Series, X_val: pd.DataFrame = None,
                             y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Train separate models for top 10 players vs others.
        
        Args:
            X_train: Training features
            y_train: Training labels
            top_10_mask: Boolean mask for top 10 players
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with trained models and metrics
        """
        # Split data
        top_10_data = X_train[top_10_mask]
        other_data = X_train[~top_10_mask]
        top_10_labels = y_train[top_10_mask]
        other_labels = y_train[~top_10_mask]
        
        # Train top 10 model
        top_10_result = self.train_ensemble_model(
            top_10_data, top_10_labels,
            X_val[top_10_mask] if X_val is not None else None,
            y_val[top_10_mask] if y_val is not None else None
        )
        
        # Train other model
        other_result = self.train_ensemble_model(
            other_data, other_labels,
            X_val[~top_10_mask] if X_val is not None else None,
            y_val[~top_10_mask] if y_val is not None else None
        )
        
        self.models['top_10'] = top_10_result['model']
        self.models['other'] = other_result['model']
        self.scalers['top_10'] = top_10_result['scaler']
        self.scalers['other'] = other_result['scaler']
        self.feature_lists['top_10'] = top_10_result['features']
        self.feature_lists['other'] = other_result['features']
        
        return {
            'top_10': top_10_result,
            'other': other_result
        }
    
    def save_model(self, model_name: str, model: Any, scaler: Any, features: List[str],
                   imputer: Any = None):
        """Save trained model and components."""
        model_path = os.path.join(self.model_dir, f'{model_name}_model.pkl')
        scaler_path = os.path.join(self.model_dir, f'{model_name}_scaler.pkl')
        features_path = os.path.join(self.model_dir, f'{model_name}_features.pkl')
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(features, features_path)
        
        if imputer is not None:
            imputer_path = os.path.join(self.model_dir, f'{model_name}_imputer.pkl')
            joblib.dump(imputer, imputer_path)
        
        print(f"Model saved: {model_name}")
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load trained model and components."""
        model_path = os.path.join(self.model_dir, f'{model_name}_model.pkl')
        scaler_path = os.path.join(self.model_dir, f'{model_name}_scaler.pkl')
        features_path = os.path.join(self.model_dir, f'{model_name}_features.pkl')
        imputer_path = os.path.join(self.model_dir, f'{model_name}_imputer.pkl')
        
        result = {
            'model': joblib.load(model_path),
            'scaler': joblib.load(scaler_path),
            'features': joblib.load(features_path)
        }
        
        if os.path.exists(imputer_path):
            result['imputer'] = joblib.load(imputer_path)
        
        return result

