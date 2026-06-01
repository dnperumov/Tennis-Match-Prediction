import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tennis_ml.live_stats import scrape_match_stats, upsert_match_stats


class LiveStatsScraperTests(unittest.TestCase):
    def test_scrape_csv_and_upsert_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / 'matches.csv'
            database = Path(temp_dir) / 'db.csv'
            pd.DataFrame([
                {
                    'Date': '2026-05-23',
                    'Tournament': 'Test Open',
                    'Round': 'R32',
                    'Surface': 'Hard',
                    'Winner': 'Player A',
                    'Loser': 'Player B',
                    'Score': '6-4 6-4',
                    'minutes': 82,
                }
            ]).to_csv(source, index=False)

            rows = scrape_match_stats(str(source), source_name='unit_test')
            db = upsert_match_stats(rows, database)
            db_again = upsert_match_stats(rows, database)

            self.assertEqual(len(rows), 1)
            self.assertEqual(db.loc[0, 'player1'], 'Player A')
            self.assertEqual(len(db_again), 1)


if __name__ == '__main__':
    unittest.main()
