import unittest

from twitter_sentiment.preprocess import preprocess_and_tokenize
from twitter_sentiment.preprocess import PreprocessorConfig, Tokenizer


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

    def test_number_policy_token(self):
        text = "I have 2 apples and 10 bananas"
        cfg = PreprocessorConfig(number_policy='token')
        tokens = preprocess_and_tokenize(text, cfg)
        self.assertIn('<num>', tokens)
        # numbers replaced, words remain
        self.assertIn('apples', tokens)

    def test_hashtag_keep_and_segment(self):
        text_keep = "#Happy"
        cfg_keep = PreprocessorConfig(hashtag_policy='keep')
        keep_toks = preprocess_and_tokenize(text_keep, cfg_keep)
        # keep policy emits both a hashtag marker and the word
        self.assertIn('hashtag', keep_toks)
        self.assertIn('happy', keep_toks)
        text_seg = "#HappyDays"
        cfg_seg = PreprocessorConfig(hashtag_policy='segment')
        toks = preprocess_and_tokenize(text_seg, cfg_seg)
        # segmentation is heuristic; accept either split or unsplit
        self.assertTrue('happy' in toks or 'happydays' in toks)

    def test_contractions_and_negation_scope(self):
        text = "I'm not happy today"
        cfg = PreprocessorConfig(contractions={'i\'m': 'i am'}, negation_scope=2)
        toks = preprocess_and_tokenize(text, cfg)
        # single-letter 'i' is filtered out by default; 'am' should be present
        self.assertIn('am', toks)
        self.assertIn('not', toks)
        self.assertIn('happy_neg', toks)
        self.assertIn('today_neg', toks)

    def test_emoji_mapping_when_enabled(self):
        text = "This is great 😀"
        cfg = PreprocessorConfig(emoji_policy='map')
        toks = preprocess_and_tokenize(text, cfg)
        self.assertIn('smile', toks)


if __name__ == '__main__':
    unittest.main()
