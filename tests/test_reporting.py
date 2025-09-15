import unittest
import numpy as np

from twitter_sentiment.model import build_pipeline, VectorizerConfig
from twitter_sentiment.evaluate import compute_metrics, get_prediction_scores


class TestReporting(unittest.TestCase):
    def test_scores_and_roc_auc(self):
        # Binary toy set with clear separation
        X = np.array([
            "good and nice",
            "pleasant and great",
            "bad and awful",
            "terrible and nasty",
        ])
        y = np.array([1, 1, 0, 0])

        pipe = build_pipeline('nb', VectorizerConfig(word_ngrams=(1, 2)))
        pipe.fit(X, y)
        preds = pipe.predict(X)
        scores = get_prediction_scores(pipe, X)
        m = compute_metrics(y, preds, scores=scores)
        self.assertIn('roc_auc', m)
        self.assertGreaterEqual(m['roc_auc'], 0.5)
        self.assertLessEqual(m['roc_auc'], 1.0)


if __name__ == '__main__':
    unittest.main()
