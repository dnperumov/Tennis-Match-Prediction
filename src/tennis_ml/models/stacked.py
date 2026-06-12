"""Reusable stacked market-aware model.

This packages the production "stacked" approach from ``walk_forward.py``:
an XGBoost classifier over the full feature set (including no-vig market
probability features), blended with the no-vig market probability in logit
space. The blend weight is selected on the most recent chronological slice of
the training window, so when the classifier adds no information beyond the
market the final probabilities collapse toward the market line instead of
fighting it.

Two submodels are trained — ``top_10`` (either player ranked inside the top
10) and ``other`` — mirroring the walk-forward fold logic exactly (same
hyperparameters, same blend-weight grid, same chronological validation
slice with early stopping).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


MARKET_PROBABILITY_COLUMN = 'player1_market_probability_novig'
SUBMODEL_NAMES = ('top_10', 'other')
DEFAULT_BLEND_GRID = [round(0.05 * step, 2) for step in range(0, 21)]
ARTIFACT_FILENAME = 'stacked_model.joblib'
METADATA_FILENAME = 'metadata.json'


def logit(p: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    clipped = np.clip(p, 0.01, 0.99)
    return np.log(clipped / (1 - clipped))


def blend_probabilities(model_prob, market_prob, weight: float):
    """Blend model and market probabilities in logit space."""
    blended_logit = weight * logit(model_prob) + (1 - weight) * logit(market_prob)
    blended = 1 / (1 + np.exp(-blended_logit))
    return np.clip(blended, 0.02, 0.98)


def choose_blend_weight(
    model_prob,
    market_prob,
    y_true,
    grid: list[float] | None = None,
) -> float:
    """Pick the model-share blend weight minimizing validation log loss."""
    grid = grid or DEFAULT_BLEND_GRID
    best_weight, best_loss = 0.0, float('inf')
    for weight in grid:
        blended = blend_probabilities(model_prob, market_prob, weight)
        loss = log_loss(y_true, blended, labels=[0, 1])
        if loss < best_loss:
            best_weight, best_loss = weight, loss
    return best_weight


def fit_stacked_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Fit the stacked XGBoost classifier with early stopping on a validation slice."""
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=5,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            eval_metric='logloss',
            early_stopping_rounds=75,
            random_state=42,
        )
        model.fit(X_imputed, y_train, eval_set=[(X_val_imputed, y_val)], verbose=False)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=1500,
            learning_rate=0.02,
            max_leaf_nodes=31,
            l2_regularization=2.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )
        model.fit(X_imputed, y_train)
    return {'model': model, 'imputer': imputer}


def predict_stacked_classifier(bundle: dict, X: pd.DataFrame) -> pd.Series:
    X_imputed = bundle['imputer'].transform(X)
    proba = bundle['model'].predict_proba(X_imputed)[:, 1]
    return pd.Series(proba, index=X.index)


def _fit_calibrator(raw_prob: np.ndarray, y_true: np.ndarray, min_isotonic_rows: int = 300):
    """Calibrate raw classifier probabilities on the validation slice.

    Uses isotonic regression when the slice is large enough, otherwise Platt
    (sigmoid) scaling on the logit. Returns None when calibration cannot be
    fit (degenerate labels), in which case raw probabilities are used.
    """
    raw_prob = np.asarray(raw_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2 or len(y_true) < 10:
        return None
    if len(y_true) >= min_isotonic_rows:
        calibrator = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds='clip')
        calibrator.fit(raw_prob, y_true)
        return {'method': 'isotonic', 'model': calibrator}
    platt = LogisticRegression(C=1e6, max_iter=1000)
    platt.fit(np.asarray(logit(raw_prob)).reshape(-1, 1), y_true)
    return {'method': 'sigmoid', 'model': platt}


def _apply_calibrator(calibrator, raw_prob: np.ndarray) -> np.ndarray:
    raw_prob = np.asarray(raw_prob, dtype=float)
    if calibrator is None:
        return np.clip(raw_prob, 0.02, 0.98)
    if calibrator['method'] == 'isotonic':
        return np.clip(calibrator['model'].predict(raw_prob), 0.02, 0.98)
    calibrated = calibrator['model'].predict_proba(np.asarray(logit(raw_prob)).reshape(-1, 1))[:, 1]
    return np.clip(calibrated, 0.02, 0.98)


class StackedModel:
    """Market-aware stacked classifier with logit-space market blending.

    ``fit`` trains ``top_10`` and ``other`` XGBoost classifiers using an
    internal chronological validation slice (early stopping + blend-weight
    grid search per submodel, mirroring ``walk_forward.py``).

    ``predict_proba`` returns the player1 win probability. Rows with
    ``player1_market_probability_novig`` present are logit-blended with the
    selected weight; rows without market data use the calibrated pure
    classifier probability.
    """

    def __init__(self, blend_grid: list[float] | None = None):
        self.blend_grid = blend_grid or DEFAULT_BLEND_GRID
        self.feature_columns_: list[str] = []
        self.submodels_: dict[str, dict[str, Any]] = {}
        self.metadata_: dict[str, Any] = {}

    # ------------------------------------------------------------------ fit

    def fit(self, train_features: pd.DataFrame, feature_columns: Iterable[str]) -> 'StackedModel':
        self.feature_columns_ = [col for col in feature_columns if col in train_features.columns]
        if not self.feature_columns_:
            raise ValueError('None of the requested feature columns are present in train_features.')
        if 'result' not in train_features.columns:
            raise ValueError("train_features must contain a 'result' column (1 = player1 won).")

        top_10_mask = self._top_10_mask(train_features)
        market_available = (
            MARKET_PROBABILITY_COLUMN in train_features.columns
            and train_features[MARKET_PROBABILITY_COLUMN].notna().any()
        )

        self.submodels_ = {}
        submodel_meta: dict[str, Any] = {}
        for model_name, mask in [('top_10', top_10_mask), ('other', ~top_10_mask)]:
            if market_available:
                mask = mask & train_features[MARKET_PROBABILITY_COLUMN].notna()
            if not mask.any():
                continue

            fold_train = train_features.loc[mask].sort_values(self._date_column(train_features))
            split_at = max(int(len(fold_train) * 0.85), len(fold_train) - 1500)
            split_at = min(max(split_at, 1), len(fold_train) - 1) if len(fold_train) > 1 else 1
            fit_slice = fold_train.iloc[:split_at]
            val_slice = fold_train.iloc[split_at:]
            if val_slice.empty:
                val_slice = fit_slice.tail(min(250, len(fit_slice)))

            bundle = fit_stacked_classifier(
                fit_slice[self.feature_columns_],
                (fit_slice['result'] == 1).astype(int),
                val_slice[self.feature_columns_],
                (val_slice['result'] == 1).astype(int),
            )
            val_model_prob = predict_stacked_classifier(bundle, val_slice[self.feature_columns_])
            y_val = (val_slice['result'] == 1).astype(int)

            if market_available and val_slice[MARKET_PROBABILITY_COLUMN].notna().all():
                blend_weight = choose_blend_weight(
                    val_model_prob,
                    val_slice[MARKET_PROBABILITY_COLUMN].astype(float),
                    y_val,
                    grid=self.blend_grid,
                )
            else:
                blend_weight = 1.0

            calibrator = _fit_calibrator(val_model_prob.to_numpy(), y_val.to_numpy())

            self.submodels_[model_name] = {
                'model': bundle['model'],
                'imputer': bundle['imputer'],
                'blend_weight': float(blend_weight),
                'calibrator': calibrator,
            }
            submodel_meta[model_name] = {
                'blend_weight': float(blend_weight),
                'fit_rows': int(len(fit_slice)),
                'validation_rows': int(len(val_slice)),
                'calibration': calibrator['method'] if calibrator else None,
            }

        if not self.submodels_:
            raise ValueError('No submodels could be trained (empty training masks).')

        dates = pd.to_datetime(train_features[self._date_column(train_features)], errors='coerce')
        self.metadata_ = {
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'training_rows': int(len(train_features)),
            'training_window': {
                'start': str(dates.min().date()) if dates.notna().any() else None,
                'end': str(dates.max().date()) if dates.notna().any() else None,
            },
            'market_features_used': bool(market_available),
            'feature_count': len(self.feature_columns_),
            'submodels': submodel_meta,
        }
        return self

    # -------------------------------------------------------------- predict

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Player1 win probability for each row of ``features``."""
        if not self.submodels_:
            raise ValueError('StackedModel is not fitted. Call fit() or load() first.')
        frame = features.reindex(columns=self.feature_columns_, fill_value=np.nan)
        top_10_mask = self._top_10_mask(features)
        result = pd.Series(np.nan, index=features.index, dtype=float)

        for model_name, mask in [('top_10', top_10_mask), ('other', ~top_10_mask)]:
            submodel = self.submodels_.get(model_name) or self._fallback_submodel(model_name)
            rows = frame.loc[mask]
            if rows.empty:
                continue
            raw = predict_stacked_classifier(submodel, rows)

            if MARKET_PROBABILITY_COLUMN in features.columns:
                market = features.loc[mask, MARKET_PROBABILITY_COLUMN].astype(float)
            else:
                market = pd.Series(np.nan, index=rows.index, dtype=float)
            has_market = market.notna()

            blended = pd.Series(np.nan, index=rows.index, dtype=float)
            if has_market.any():
                blended.loc[has_market] = blend_probabilities(
                    raw.loc[has_market],
                    market.loc[has_market],
                    submodel['blend_weight'],
                )
            if (~has_market).any():
                blended.loc[~has_market] = _apply_calibrator(
                    submodel['calibrator'],
                    raw.loc[~has_market].to_numpy(),
                )
            result.loc[mask] = blended

        return result

    def blend_weights(self) -> dict[str, float]:
        return {name: submodel['blend_weight'] for name, submodel in self.submodels_.items()}

    # -------------------------------------------------------- save / load

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            'format_version': 1,
            'feature_columns': self.feature_columns_,
            'blend_grid': self.blend_grid,
            'submodels': self.submodels_,
            'metadata': self.metadata_,
        }
        joblib.dump(payload, directory / ARTIFACT_FILENAME)
        (directory / METADATA_FILENAME).write_text(
            json.dumps({'model_kind': 'stacked', **self.metadata_}, indent=2, sort_keys=True, default=str),
            encoding='utf-8',
        )
        return directory / ARTIFACT_FILENAME

    @classmethod
    def load(cls, directory: str | Path) -> 'StackedModel':
        directory = Path(directory)
        payload = joblib.load(directory / ARTIFACT_FILENAME)
        instance = cls(blend_grid=payload.get('blend_grid'))
        instance.feature_columns_ = payload['feature_columns']
        instance.submodels_ = payload['submodels']
        instance.metadata_ = payload.get('metadata', {})
        return instance

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / ARTIFACT_FILENAME).exists()

    # ------------------------------------------------------------ internals

    @staticmethod
    def _top_10_mask(features: pd.DataFrame) -> pd.Series:
        if 'player1_rank' in features.columns and 'player2_rank' in features.columns:
            p1 = pd.to_numeric(features['player1_rank'], errors='coerce')
            p2 = pd.to_numeric(features['player2_rank'], errors='coerce')
            return (p1 <= 10) | (p2 <= 10)
        return pd.Series(False, index=features.index)

    @staticmethod
    def _date_column(features: pd.DataFrame) -> str:
        return 'match_date' if 'match_date' in features.columns else 'tourney_date'

    def _fallback_submodel(self, missing_name: str) -> dict[str, Any]:
        for name in SUBMODEL_NAMES:
            if name != missing_name and name in self.submodels_:
                return self.submodels_[name]
        raise ValueError(f'No trained submodel available to score {missing_name} rows.')
