import unittest
import numpy as np

from twitter_sentiment.evaluate import compute_metrics


class TestEvaluate(unittest.TestCase):
    def test_metrics_correct_ordering(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        m = compute_metrics(y_true, y_pred)
        self.assertIn('accuracy', m)
        self.assertIn('f1_macro', m)
        self.assertIn('classification_report', m)
        self.assertIn('confusion_matrix', m)
        # Confusion matrix should reflect TN=2, FP=0, FN=1, TP=1
        cm = np.array(m['confusion_matrix'])
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(int(cm[0, 0]), 2)
        self.assertEqual(int(cm[0, 1]), 0)
        self.assertEqual(int(cm[1, 0]), 1)
        self.assertEqual(int(cm[1, 1]), 1)


if __name__ == '__main__':
    unittest.main()
