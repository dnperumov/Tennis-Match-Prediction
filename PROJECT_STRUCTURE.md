# Project Structure

## Cleaned Up ML Pipeline Structure

This document describes the cleaned up and organized structure of the Tennis Match Prediction ML Pipeline.

## Directory Structure

```
Tennis-Match-Prediction/
├── src/                          # Source code
│   └── tennis_ml/               # Main package
│       ├── __init__.py
│       ├── data/                # Data loading and processing
│       │   ├── __init__.py
│       │   ├── loader.py       # DataLoader class
│       │   └── player.py       # Player statistics class
│       ├── features/            # Feature engineering
│       │   ├── __init__.py
│       │   ├── engineering.py  # Core feature engineering
│       │   └── betting_features.py  # Betting-specific features
│       ├── models/              # Model training and prediction
│       │   ├── __init__.py
│       │   ├── trainer.py      # ModelTrainer class
│       │   └── predictor.py    # MatchPredictor class
│       └── utils/               # Utilities
│           └── __init__.py
│
├── notebooks/                    # Jupyter notebooks
│   └── Tennis_Prediction_Model_9_16_24.ipynb
│
├── data/                         # Data directory
│   └── raw/                     # Raw data files (if needed)
│
├── models/                       # Saved trained models
│   ├── top_10_model.pkl
│   ├── top_10_scaler.pkl
│   ├── top_10_features.pkl
│   ├── other_model.pkl
│   ├── other_scaler.pkl
│   └── other_features.pkl
│
├── tests/                        # Unit tests
│
├── tennis_datav2/               # ATP match data (CSV files)
│   ├── atp_matches_2000.csv
│   ├── atp_matches_2001.csv
│   └── ... (2000-2024)
│
├── pipeline.py                  # Main pipeline script
├── setup.py                     # Package setup
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore rules
└── PROJECT_STRUCTURE.md         # This file
```

## Removed Files

The following old/unnecessary files have been removed:

- ❌ `tennis_games_prediction.py` - Old prediction script
- ❌ `tennis_prediction.py` - Old prediction script
- ❌ `tournament_predictor.py` - Old tournament predictor
- ❌ `predict_indian_wells.py` - Specific tournament script
- ❌ `betting_improvements.py` - Integrated into main pipeline
- ❌ `enhanced_features.py` - Integrated into features module
- ❌ `real_time_integration.py` - Not part of core pipeline
- ❌ `advanced_models.py` - Integrated into models module
- ❌ `complete_betting_system.py` - Integrated into models module
- ❌ `integrate_improvements.py` - Temporary integration file
- ❌ `Tennis_Prediction_Model_9_4_24.ipynb` - Old notebook version
- ❌ `tennis_env_39/` - Old virtual environment
- ❌ `wheels/` - Installer files
- ❌ `Miniconda3-latest-MacOSX-arm64.sh` - Installer

## Key Components

### 1. Data Module (`src/tennis_ml/data/`)
- **DataLoader**: Loads and preprocesses ATP match data
- **Player**: Tracks comprehensive player statistics over time

### 2. Features Module (`src/tennis_ml/features/`)
- **FeatureEngineer**: Creates core features for prediction
- **BettingFeatureEngineer**: Adds betting-specific features (momentum, confidence, etc.)

### 3. Models Module (`src/tennis_ml/models/`)
- **ModelTrainer**: Trains ensemble models with calibration
- **MatchPredictor**: Makes predictions using trained models

### 4. Pipeline Script (`pipeline.py`)
- Main orchestration script
- Handles data loading, feature engineering, training, and evaluation
- Command-line interface for easy usage

## Usage

### Training Models
```bash
python pipeline.py --train
```

### Making Predictions
```python
from tennis_ml.models import MatchPredictor
from tennis_ml.models.trainer import ModelTrainer

# Load model and make predictions
```

## Benefits of New Structure

1. **Modularity**: Clear separation of concerns
2. **Maintainability**: Easy to update individual components
3. **Scalability**: Easy to add new features or models
4. **Production Ready**: Proper package structure
5. **Clean Codebase**: Removed redundant files
6. **Documentation**: Clear README and structure docs

## Next Steps

1. Add unit tests in `tests/` directory
2. Add model versioning
3. Create API wrapper for predictions
4. Add monitoring and logging
5. Set up CI/CD pipeline

