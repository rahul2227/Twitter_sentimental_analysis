import unittest
import numpy as np

from twitter_sentiment.model import build_pipeline, VectorizerConfig


class TestModels(unittest.TestCase):
    def _toy(self):
        X = np.array([
            "good and nice",
            "pleasant and great",
            "awesome and wonderful",
            "bad and awful",
            "terrible and nasty",
            "horrible and poor",
        ])
        y = np.array([1, 1, 1, 0, 0, 0])
        return X, y

    def test_cnb_pipeline(self):
        X, y = self._toy()
        pipe = build_pipeline('cnb', VectorizerConfig(word_ngrams=(1, 2)))
        pipe.fit(X, y)
        preds = pipe.predict(X)
        self.assertEqual(preds.shape[0], X.shape[0])

    def test_linearsvm_calibrated_with_selectk(self):
        X, y = self._toy()
        pipe = build_pipeline('linearsvm', VectorizerConfig(word_ngrams=(1, 2)), select_k=5, calibrate_probabilities=True)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        self.assertEqual(preds.shape[0], X.shape[0])


if __name__ == '__main__':
    unittest.main()
