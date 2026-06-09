import tempfile
import tarfile
import unittest
from pathlib import Path

import pandas as pd

from tennis_ml.daily import MatchPredictionRequest, scrape_daily_matches
from tennis_ml.daily.prediction import fair_odds
from tennis_ml.daily.model_artifacts import ensure_latest_model_artifact, latest_model_dir
from tennis_ml.live_stats.database import (
    init_live_db,
    read_table,
    upsert_normalized_matches,
    upsert_odds_snapshots,
)
from tennis_ml.live_stats.scraper import normalize_match_stats


class DailyPipelineTests(unittest.TestCase):
    def test_sqlite_match_upsert_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / 'live.db'
            rows = normalize_match_stats(pd.DataFrame([{
                'Date': '2026-05-31',
                'Tournament': 'Test Open',
                'Round': 'R32',
                'Surface': 'Hard',
                'Winner': 'Player A',
                'Loser': 'Player B',
                'Score': '6-4 6-4',
            }]), source='unit')

            init_live_db(db_path)
            upsert_normalized_matches(rows, db_path)
            upsert_normalized_matches(rows, db_path)

            self.assertEqual(len(read_table('matches', db_path)), 1)
            self.assertEqual(len(read_table('match_stats', db_path)), 1)

    def test_scrape_daily_matches_filters_source_date(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / 'ta.csv'
            db_path = Path(temp_dir) / 'live.db'
            pd.DataFrame([
                {'Date': '2026-05-30', 'Tournament': 'Old Open', 'Winner': 'A', 'Loser': 'B'},
                {'Date': '2026-05-31', 'Tournament': 'New Open', 'Winner': 'C', 'Loser': 'D'},
            ]).to_csv(source, index=False)

            rows = scrape_daily_matches('2026-05-31', db_path=db_path, source=str(source))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows.loc[0, 'tournament'], 'New Open')
            self.assertEqual(len(read_table('matches', db_path)), 1)

    def test_odds_snapshot_does_not_store_secret_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / 'live.db'
            rows = pd.DataFrame([{
                'snapshot_time': '2026-05-31T12:00:00Z',
                'source': 'kalshi',
                'event_ticker': 'EVT',
                'market_ticker': 'MKT',
                'yes_ask': 0.4,
                'no_ask': 0.62,
                'yes_decimal_odds': 2.5,
                'no_decimal_odds': 1.6129,
                'private_key': 'do-not-store',
            }])

            upsert_odds_snapshots(rows, db_path)
            stored = read_table('odds_snapshots', db_path)

            self.assertEqual(len(stored), 1)
            self.assertNotIn('do-not-store', stored.loc[0, 'raw_json'])

    def test_prediction_request_validation_and_fair_odds(self):
        request = MatchPredictionRequest(player1='A', player2='A')
        with self.assertRaises(ValueError):
            request.validate()
        self.assertAlmostEqual(fair_odds(0.4), 2.5)

    def test_model_artifact_download_and_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_model = temp_path / '2026-06-08'
            source_model.mkdir()
            (source_model / 'daily_ensemble_model.pkl').write_text('placeholder')
            archive = temp_path / 'model.tar.gz'
            with tarfile.open(archive, 'w:gz') as handle:
                handle.add(source_model, arcname=source_model.name)

            model_root = temp_path / 'models'
            found = ensure_latest_model_artifact(model_root, artifact_url=archive.as_uri())

            self.assertEqual(found.name, '2026-06-08')
            self.assertEqual(latest_model_dir(model_root).name, '2026-06-08')


if __name__ == '__main__':
    unittest.main()
