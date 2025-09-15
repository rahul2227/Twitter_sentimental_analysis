import unittest
import numpy as np

from twitter_sentiment.train import _balance_data


class TestBalance(unittest.TestCase):
    def test_downsample(self):
        X = np.array([f"t{i}" for i in range(12)])
        y = np.array([0]*10 + [1]*2)
        Xb, yb = _balance_data(X, y, 'downsample', np.random.default_rng(0))
        # balanced counts
        _, counts = np.unique(yb, return_counts=True)
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[0], 2)

    def test_upsample(self):
        X = np.array([f"t{i}" for i in range(12)])
        y = np.array([0]*10 + [1]*2)
        Xb, yb = _balance_data(X, y, 'upsample', np.random.default_rng(0))
        _, counts = np.unique(yb, return_counts=True)
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[0], 10)


if __name__ == '__main__':
    unittest.main()

