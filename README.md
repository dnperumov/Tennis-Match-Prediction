# Tennis Match Prediction ML Pipeline

A production-ready machine learning pipeline for predicting tennis match outcomes using historical ATP match data from 2000-2026.

## 🎯 Features

- **Comprehensive Data Processing**: Chronological data splitting with no-look-ahead rolling features
- **Advanced Feature Engineering**: rolling form, head-to-head, serve/return strength, fatigue, event context, and surface-specific metrics
- **Rolling Elo Ratings**: dynamic-K overall, surface, recent-form, uncertainty, and inactivity-decayed Elo features are built without look-ahead
- **Market-Aware Training**: Optional no-vig odds features for probability modeling
- **Market-Residual Walk-Forward Model**: Learns adjustments around the no-vig market baseline instead of free-floating edges
- **Betting Diagnostics**: PnL, CLV proxy, calibration buckets, probability-bucket ROI, and nested walk-forward strategy selection
- **Ensemble Models**: Combined Random Forest, gradient boosting, and Logistic Regression
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

For market-aware training, pass historical Tennis-Data odds:

```bash
python pipeline.py --train \
  --data-dir tennis_datav2 \
  --model-dir models \
  --odds-file data/odds/atp_odds_2013_2026.csv
```

This will:
1. Load ATP match data (2000-2026)
2. Split data chronologically
3. Initialize player statistics from the first 50% warmup window
4. Engineer rolling features for each later match before updating player histories
5. Train ensemble models on the next 25%
6. Evaluate on the final 25% and save models

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
- **Chronological Splitting**: 50% initial warmup, 25% training, 25% no-look-ahead testing
- **Temporal Integrity**: Each match is featurized before its result updates player statistics
- **Minimum Matches Filter**: Only players with 10+ matches included

### Feature Engineering
- **Basic Features**: Rankings, age, height, seeds, handedness
- **Performance Metrics**: Win percentages, surface performance, head-to-head
- **Betting Features**: Momentum, confidence, fatigue indicators, tournament context
- **Total Features**: 30+ engineered features

### Model Design
- **Ensemble Approach**: Voting classifier with Random Forest, gradient boosting, and Logistic Regression
- **Separate Models**: 
  - Top 10 players model (for elite matches)
  - Other players model (for general matches)
- **Probability Calibration**: Platt scaling for accurate probability estimates
- **Feature Scaling**: StandardScaler for normalization

## 📈 Performance

Market-aware no-look-ahead performance on 2000-2026 ATP data:
- **Top 10 model accuracy**: 75.72%
- **Other players model accuracy**: 65.03%
- **Probability Calibration**: Brier Score 0.1668 for the top 10 model and 0.2163 for the other players model
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
The year range is end-exclusive, so this trains on 2010-2026:

```bash
python pipeline.py --train --years 2010 2027
```

### Training with Custom Directories
```bash
python pipeline.py --train --data-dir custom_data --model-dir custom_models
```

### Backtesting PnL
First score the no-look-ahead test set:

```bash
python backtest.py --data-dir tennis_datav2 --model-dir models
```

This writes `backtest_results/predictions.csv`. To run a PnL simulation, pass a historical odds file:

```bash
python backtest.py \
  --data-dir tennis_datav2 \
  --model-dir models \
  --odds-file data/odds/atp_odds_2013_2026.csv \
  --edge-threshold 0.03 \
  --stake 1
```

The odds file should include `Date`, `Winner`, `Loser`, and decimal odds columns. Tennis-data style columns such as `AvgW`/`AvgL`, `MaxW`/`MaxL`, `PSW`/`PSL`, or `B365W`/`B365L` are supported. Use `--winner-odds-col` and `--loser-odds-col` to force a specific bookmaker.

To combine multiple Tennis-Data season spreadsheets into one CSV:

```bash
python scripts/combine_tennis_data_odds.py \
  data/odds/atp_odds_2019.xlsx data/odds/atp_odds_2020.xlsx \
  --output data/odds/atp_odds_2019_2026.csv
```

Backtest reports are written to `backtest_results/`, including scored predictions, selected bets, threshold summaries, and model-type summaries.

Useful strategy filters:

```bash
python backtest.py \
  --data-dir tennis_datav2 \
  --model-dir models \
  --odds-file data/odds/atp_odds_2013_2026.csv \
  --model-type top_10 \
  --max-odds 5 \
  --edge-threshold 0.03
```

Recent market-aware backtest results using average odds:
- **All bets, EV >= 0.03**: 3,108 bets, +3.24% ROI
- **Top-10 model, max odds 5**: 886 bets, +10.73% ROI
- **Top-10 model, no slams, max odds 10**: 787 bets, +10.23% ROI

### Walk-Forward Validation
The stricter validation retrains fresh models for each test year using only prior seasons. When odds are supplied, Tennis-Data match dates are used as exact match dates and Jeff Sackmann `tourney_date` is preserved as the tournament start date:

```bash
python walk_forward.py \
  --data-dir tennis_datav2 \
  --odds-file data/odds/atp_odds_2013_2026.csv \
  --winner-odds-col B365W \
  --loser-odds-col B365L \
  --train-start-year 2013 \
  --test-start-year 2019 \
  --test-end-year 2027 \
  --edge-threshold 0.03
```

By default, `walk_forward.py` uses a market-residual probability model:

```
final_probability = no_vig_market_probability + learned_residual
```

This makes the model prove it can improve on the market rather than hallucinating large raw edges. Pass `--probability-model classifier` to reproduce the older classifier-style walk-forward.

This writes reports to `walk_forward_results/`, including:
- `predictions.csv`: out-of-sample probabilities
- `bets.csv`: selected flat-stake bets
- `summary.csv`: PnL, ROI, drawdown, and CLV proxy
- `calibration_summary.csv`: probability bucket calibration and market residual calibration
- `probability_bucket_roi.csv`: ROI by model probability bucket
- `nested_strategy_summary.csv`: strategy filters selected only from prior walk-forward years

Walk-forward caching is enabled by default. It stores exact-date aligned matches and per-fold no-lookahead feature frames under `.cache/walk_forward/`, keyed by ATP file stats, odds file stats, odds column choice, fold year, and cache-version strings. Useful cache controls:

```bash
python walk_forward.py ... --cache-dir .cache/walk_forward
python walk_forward.py ... --refresh-cache
python walk_forward.py ... --no-cache
```

This makes repeated 2019-2026 runs much faster because the expensive chronological feature construction is reused while model training, scoring, and report generation still run fresh.

Profitability research gates are also emitted by walk-forward runs:
- `gate_summary.csv`: pass/fail promotion gates using friction-adjusted ROI, CLV, calibration, minimum bets, and year stability
- `strategy_candidates.csv`: candidate filter grid ranked by friction-adjusted ROI
- `edge_trust_summary.csv`: second-stage edge-trust model diagnostics trained only on prior years
- `segment_stability.csv`: year, model-type, surface, and event-level stability checks

The default betting friction is a 2% odds haircut. Override it with:

```bash
python walk_forward.py ... --friction-bps 200
```

To analyze an existing walk-forward run without retraining:

```bash
python analyze_strategy.py \
  --predictions walk_forward_results_residual/predictions.csv \
  --bets walk_forward_results_residual/bets.csv \
  --calibration walk_forward_results_residual/calibration_summary.csv \
  --output-dir strategy_analysis \
  --friction-bps 200
```

To start a paper-trading ledger from scored opportunities:

```bash
python paper_trade.py \
  --odds-input walk_forward_results_residual/predictions.csv \
  --output-dir paper_trading \
  --edge-threshold 0.03 \
  --friction-bps 200
```

To mine which historical bet types made money:

```bash
python mine_profitable_trades.py \
  --bets walk_forward_results_current/bets.csv \
  --output-dir trade_mining_current \
  --friction-bps 200 \
  --min-bets 30 \
  --clusters 8
```

This writes filter, rule, cluster, and clustered-bet diagnostics. These are exploratory only; any pocket found here must be promoted through nested walk-forward validation before paper trading.

To fetch current Kalshi tennis market quotes:

```bash
python fetch_kalshi_tennis.py \
  --output data/kalshi/tennis_markets.csv \
  --max-pages 10
```

The Kalshi connector currently uses public event/market quote fields. Full order book snapshots and order placement require authenticated API access and should only be added after paper-trading gates pass.

Authenticated Kalshi calls read credentials from environment variables only:

```bash
export KALSHI_API_KEY_ID="..."
export KALSHI_PRIVATE_KEY_PEM="$(cat /path/to/kalshi_private_key.pem)"
```

Do not commit API keys or private keys to this repository.

To ingest new match stats from a CSV/HTML table into the local research database:

```bash
python scrape_match_stats.py \
  --source path/or/url/to/match_stats.csv \
  --source-name tennis_abstract \
  --database data/live/match_stats.csv
```

The scraper normalizes common columns such as date, tournament, round, surface, winner/player1, loser/player2, score, minutes, and serve/return stat percentages. It upserts by `source_match_id`, so re-running the same scrape updates the row instead of duplicating it.

Current default paper-trade strategy:

```bash
python evaluate_strategy.py \
  --bets walk_forward_results_current/bets.csv \
  --strategy favorite_band_no_slams_v1 \
  --output-dir strategy_eval_current \
  --friction-bps 200
```

`favorite_band_no_slams_v1` accepts model-selected bets with decimal odds from 1.50 to 2.00 and excludes Grand Slams. In the 2019-2026 B365 walk-forward ledger it produced 152 bets, +13.84% friction-adjusted ROI, and positive friction-adjusted profit in every test year. It still fails CLV gates, so it is suitable for paper trading, not live staking.

To generate a local dashboard:

```bash
python dashboard_app.py \
  --walk-forward-dir walk_forward_results_current \
  --strategy-dir strategy_eval_current \
  --mining-dir trade_mining_current \
  --kalshi-file data/kalshi/tennis_markets.csv \
  --output dashboard/tennis_trading_dashboard.html
```

The original stricter classifier walk-forward result was materially weaker than the single split:
- **All bets, EV >= 0.03**: 4,307 bets, -9.30% ROI
- **Top-10 model, max odds 5**: 1,167 bets, -1.59% ROI
- **Other clay only**: 704 bets, +16.80% ROI, but most profit comes from 2022 and should be treated as unstable

Residual-model smoke test for 2019-2020 using `B365W`/`B365L` execution odds and average-odds CLV proxy:
- **All bets, EV >= 0.03**: 427 bets, -3.61% ROI
- **Top/other yearly metrics**: Top-10 Brier 0.1750 in 2019 and 0.1481 in 2020; other-player Brier 0.2193 in 2019 and 0.2091 in 2020
- **CLV caveat**: true opening/closing timestamps are not in Tennis-Data. With bookmaker columns such as `B365W`/`B365L`, the CLV report is a proxy against average odds; this smoke run had average CLV -0.12% and a 43.09% positive-CLV rate.

Use the walk-forward reports as the main decision source. The single-split backtests are useful for research, but too optimistic for strategy selection.

## 🧪 Development

### Running Tests
```bash
PYTHONPATH=src python -m unittest discover tests
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
- Scikit-learn, XGBoost or scikit-learn gradient boosting, and other open-source ML libraries

## 🔮 Future Improvements

- [ ] Real-time data integration
- [ ] Advanced ensemble methods
- [ ] Hyperparameter optimization
- [ ] Model versioning
- [ ] API for predictions
- [ ] Web dashboard
