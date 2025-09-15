import io
import unittest
import pandas as pd

from twitter_sentiment.data import load_datasets


class TestData(unittest.TestCase):
    def test_load_datasets_success(self):
        train_csv = io.StringIO("label,tweet\n0,hello\n1,world\n")
        test_csv = io.StringIO("tweet\nhello again\n")
        # pandas accepts file-like objects directly
        train_df, test_df = load_datasets(train_csv, test_csv)
        self.assertEqual(list(train_df.columns), ['label', 'tweet'])
        self.assertEqual(list(test_df.columns), ['tweet'])

    def test_load_datasets_missing_cols(self):
        train_csv = io.StringIO("tweet\nhello\n")
        test_csv = io.StringIO("tweet\nhello\n")
        with self.assertRaises(ValueError):
            load_datasets(train_csv, test_csv)


if __name__ == '__main__':
    unittest.main()
