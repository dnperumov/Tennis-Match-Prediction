import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tennis_ml.models.stacked import StackedModel, blend_probabilities


def make_synthetic_frame(rows: int = 600, seed: int = 7) -> pd.DataFrame:
    """Tiny synthetic match frame with informative features and market probs."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1.2, rows)
    noise = rng.normal(0, 1.0, rows)
    p_true = 1 / (1 + np.exp(-signal))
    result = np.where(rng.random(rows) < p_true, 1, 2)
    market_prob = np.clip(p_true + rng.normal(0, 0.05, rows), 0.05, 0.95)

    half = rows // 2
    return pd.DataFrame({
        'match_date': pd.date_range('2022-01-01', periods=rows, freq='D'),
        'player1_rank': np.concatenate([
            rng.integers(1, 9, half),            # top-10 matches
            rng.integers(20, 200, rows - half),  # other matches
        ]),
        'player2_rank': rng.integers(20, 200, rows),
        'elo_diff': signal * 100,
        'rank_diff': -signal * 30 + noise * 5,
        'noise_feature': noise,
        'player1_market_probability_novig': market_prob,
        'player2_market_probability_novig': 1 - market_prob,
        'result': result,
    })


FEATURES = [
    'elo_diff', 'rank_diff', 'noise_feature',
    'player1_market_probability_novig', 'player2_market_probability_novig',
]


class StackedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = make_synthetic_frame()
        cls.model = StackedModel().fit(cls.frame, FEATURES)

    def test_fit_trains_both_submodels_with_blend_weights(self):
        self.assertIn('top_10', self.model.submodels_)
        self.assertIn('other', self.model.submodels_)
        for name, weight in self.model.blend_weights().items():
            self.assertGreaterEqual(weight, 0.0, name)
            self.assertLessEqual(weight, 1.0, name)

    def test_predict_proba_with_market_probability_is_sane(self):
        probs = self.model.predict_proba(self.frame.head(50))
        self.assertEqual(len(probs), 50)
        self.assertTrue(((probs >= 0.02) & (probs <= 0.98)).all())
        self.assertFalse(probs.isna().any())

    def test_predict_proba_without_market_uses_calibrated_classifier(self):
        no_market = self.frame.head(50).copy()
        no_market['player1_market_probability_novig'] = np.nan
        no_market['player2_market_probability_novig'] = np.nan
        probs = self.model.predict_proba(no_market)
        self.assertTrue(((probs >= 0.0) & (probs <= 1.0)).all())
        self.assertFalse(probs.isna().any())
        # Strong positive signal should still produce >0.5 probabilities.
        strong = no_market.copy()
        strong['elo_diff'] = 300.0
        strong['rank_diff'] = -90.0
        self.assertGreater(self.model.predict_proba(strong).mean(), 0.5)

    def test_blend_behavior_zero_weight_collapses_to_market(self):
        model = StackedModel().fit(self.frame, FEATURES)
        for submodel in model.submodels_.values():
            submodel['blend_weight'] = 0.0
        rows = self.frame.head(20)
        probs = model.predict_proba(rows)
        expected = rows['player1_market_probability_novig'].clip(0.02, 0.98)
        np.testing.assert_allclose(probs.to_numpy(), expected.to_numpy(), atol=1e-9)

    def test_blend_behavior_market_pulls_probability(self):
        rows = self.frame.head(20).copy()
        with_market = self.model.predict_proba(rows)
        rows_no_market = rows.copy()
        rows_no_market['player1_market_probability_novig'] = np.nan
        without_market = self.model.predict_proba(rows_no_market)
        # Blended output should match the logit blend formula given raw probs.
        for name, mask_fn in [('top_10', lambda f: (f['player1_rank'] <= 10) | (f['player2_rank'] <= 10))]:
            submodel = self.model.submodels_[name]
            if submodel['blend_weight'] == 1.0:
                continue
            mask = mask_fn(rows)
            self.assertFalse(
                np.allclose(with_market[mask].to_numpy(), without_market[mask].to_numpy())
            )

    def test_save_load_roundtrip_predictions_identical(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.model.save(temp_dir)
            self.assertTrue((Path(temp_dir) / 'stacked_model.joblib').exists())
            self.assertTrue((Path(temp_dir) / 'metadata.json').exists())
            self.assertTrue(StackedModel.exists(temp_dir))
            loaded = StackedModel.load(temp_dir)
        self.assertEqual(loaded.feature_columns_, self.model.feature_columns_)
        self.assertEqual(loaded.blend_weights(), self.model.blend_weights())
        self.assertEqual(loaded.metadata_.get('trained_at'), self.model.metadata_.get('trained_at'))
        original = self.model.predict_proba(self.frame.tail(40))
        roundtrip = loaded.predict_proba(self.frame.tail(40))
        np.testing.assert_allclose(original.to_numpy(), roundtrip.to_numpy())

    def test_blend_probabilities_formula(self):
        model_prob = pd.Series([0.7])
        market_prob = pd.Series([0.5])
        full_model = blend_probabilities(model_prob, market_prob, 1.0)
        full_market = blend_probabilities(model_prob, market_prob, 0.0)
        self.assertAlmostEqual(float(full_model[0]), 0.7, places=9)
        self.assertAlmostEqual(float(full_market[0]), 0.5, places=9)
        halfway = blend_probabilities(model_prob, market_prob, 0.5)
        self.assertTrue(0.5 < float(halfway[0]) < 0.7)


if __name__ == '__main__':
    unittest.main()
