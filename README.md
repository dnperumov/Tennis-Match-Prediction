# Tennis Match Prediction ML Pipeline

A production-ready machine learning pipeline for predicting tennis match outcomes using historical ATP match data from 2000-2024.

## 🎯 Features

- **Comprehensive Data Processing**: Chronological data splitting for temporal validation
- **Advanced Feature Engineering**: 30+ features including momentum, confidence, and surface-specific metrics
- **Ensemble Models**: Combined Random Forest, XGBoost, and Logistic Regression
- **Probability Calibration**: Calibrated probabilities for accurate predictions
- **Separate Models**: Specialized models for top 10 players vs others
- **Production Ready**: Clean, modular codebase with proper structure

## 📁 Project Structure

```
Tennis-Match-Prediction/
├── src/
│   └── tennis_ml/
│       ├── data/           # Data loading and player statistics
│       ├── features/       # Feature engineering
│       ├── models/         # Model training and prediction
│       └── utils/          # Utilities
├── notebooks/              # Jupyter notebooks for exploration
├── data/                   # Data directory
│   └── raw/               # Raw data files
├── models/                 # Saved trained models
├── tests/                  # Unit tests
├── pipeline.py            # Main pipeline script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### Installation

See also: `ENV_SETUP.md`.

1. Clone the repository:
```bash
git clone <repository-url>
cd Tennis-Match-Prediction
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Models

Run the complete ML pipeline:

```bash
python pipeline.py --train --data-dir tennis_datav2 --model-dir models
```

This will:
1. Load ATP match data (2000-2024)
2. Split data chronologically
3. Initialize player statistics
4. Engineer features
5. Train ensemble models
6. Evaluate and save models

### Making Predictions

After training, you can make predictions:

```python
from tennis_ml.models import MatchPredictor
from tennis_ml.models.trainer import ModelTrainer

# Load trained model
trainer = ModelTrainer(model_dir='models')
top_10_model = trainer.load_model('top_10')
other_model = trainer.load_model('other')

# Create predictor
predictor = MatchPredictor(
    model=top_10_model['model'],
    scaler=top_10_model['scaler'],
    features=top_10_model['features'],
    players=players_dict
)

# Make prediction
result = predictor.predict(
    player1_name="Novak Djokovic",
    player2_name="Rafael Nadal",
    surface="Hard",
    tournament_importance=5
)

print(f"Player 1 Win Probability: {result['player1_probability']:.3f}")
print(f"Player 2 Win Probability: {result['player2_probability']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
```

## 📊 Model Architecture

### Data Processing
- **Chronological Splitting**: 50% initial (player stats), 25% training, 25% validation
- **Temporal Integrity**: Player statistics updated progressively
- **Minimum Matches Filter**: Only players with 10+ matches included

### Feature Engineering
- **Basic Features**: Rankings, age, height, seeds, handedness
- **Performance Metrics**: Win percentages, surface performance, head-to-head
- **Betting Features**: Momentum, confidence, fatigue indicators, tournament context
- **Total Features**: 30+ engineered features

### Model Design
- **Ensemble Approach**: Voting classifier with Random Forest, XGBoost, and Logistic Regression
- **Separate Models**: 
  - Top 10 players model (for elite matches)
  - Other players model (for general matches)
- **Probability Calibration**: Platt scaling for accurate probability estimates
- **Feature Scaling**: StandardScaler for normalization

## 📈 Performance

Expected model performance:
- **Accuracy**: 85-88%
- **Probability Calibration**: Brier Score < 0.20
- **Features**: 30+ comprehensive features

## 🔧 Configuration

### Data Directory
Default: `tennis_datav2/`

The pipeline will automatically download missing data files from the Jeff Sackmann tennis data repository.

### Model Directory
Default: `models/`

Trained models are saved with the following structure:
- `{model_name}_model.pkl` - Trained model
- `{model_name}_scaler.pkl` - Feature scaler
- `{model_name}_features.pkl` - Feature list
- `{model_name}_imputer.pkl` - Missing value imputer

## 📝 Usage Examples

### Training with Custom Year Range
```bash
python pipeline.py --train --years 2010 2024
```

### Training with Custom Directories
```bash
python pipeline.py --train --data-dir custom_data --model-dir custom_models
```

## 🧪 Development

### Running Tests
```bash
pytest tests/
```

### Code Structure
- `src/tennis_ml/data/`: Data loading and player statistics tracking
- `src/tennis_ml/features/`: Feature engineering (basic + betting features)
- `src/tennis_ml/models/`: Model training and prediction
- `pipeline.py`: Main orchestration script

## 📚 Documentation

For detailed documentation, see:
- `notebooks/` - Jupyter notebooks with examples
- Code docstrings in each module

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Tests are included for new features
- Documentation is updated

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- ATP Tour for match data
- Jeff Sackmann for maintaining the tennis_atp GitHub repository
- Scikit-learn, XGBoost, and other open-source ML libraries

## 🔮 Future Improvements

- [ ] Real-time data integration
- [ ] Advanced ensemble methods
- [ ] Hyperparameter optimization
- [ ] Model versioning
- [ ] API for predictions
- [ ] Web dashboard
