import unittest

from twitter_sentiment.preprocess import preprocess_and_tokenize


class TestPreprocess(unittest.TestCase):
    def test_basic_normalization_and_tokenization(self):
        text = "Check out https://example.com @user!!! Sooooo cooool 😎"
        tokens = preprocess_and_tokenize(text)
        self.assertIn('<url>', tokens)
        self.assertIn('<user>', tokens)
        # repeated chars collapsed to 2
        self.assertIn('sooo', tokens)
        self.assertIn('coool', tokens)

    def test_hashtags_and_apostrophes(self):
        text = "#Happy I'm loving it!"  # keep hashtag word and contraction
        tokens = preprocess_and_tokenize(text)
        self.assertIn('happy', tokens)
        self.assertIn("i'm", tokens)


if __name__ == '__main__':
    unittest.main()
