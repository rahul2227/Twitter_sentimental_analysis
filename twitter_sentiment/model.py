from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from .preprocess import preprocess_and_tokenize


@dataclass
class VectorizerConfig:
    word_ngrams: Tuple[int, int] = (1, 2)
    use_char_ngrams: bool = False
    char_ngrams: Tuple[int, int] = (3, 5)


def build_vectorizer(cfg: VectorizerConfig) -> Pipeline:
    """Create a vectorization pipeline combining word and optional char n-grams."""
    word_vec = CountVectorizer(analyzer=preprocess_and_tokenize, ngram_range=cfg.word_ngrams)
    if cfg.use_char_ngrams:
        char_vec = CountVectorizer(analyzer='char', ngram_range=cfg.char_ngrams)
        union = FeatureUnion([('word', word_vec), ('char', char_vec)])
        return Pipeline([('union', union), ('tfidf', TfidfTransformer())])
    else:
        return Pipeline([('bow', word_vec), ('tfidf', TfidfTransformer())])


def build_classifier(name: str):
    name = name.lower()
    if name in {"nb", "naive_bayes", "multinomialnb"}:
        return MultinomialNB()
    if name in {"logreg", "logistic", "logistic_regression"}:
        return LogisticRegression(max_iter=1000, n_jobs=1, class_weight='balanced')
    if name in {"linearsvm", "svm", "linsvm"}:
        return LinearSVC(class_weight='balanced')
    raise ValueError(f"Unknown model '{name}'. Expected one of: nb, logreg, linearsvm")


def build_pipeline(model_name: str, vec_cfg: VectorizerConfig) -> Pipeline:
    """Build full pipeline: vectorizer(s) + classifier."""
    vec = build_vectorizer(vec_cfg)
    clf = build_classifier(model_name)
    return Pipeline([('vec', vec), ('clf', clf)])


def default_param_grid(model_name: str) -> Dict[str, list]:
    """Return a small, safe parameter grid for quick tuning."""
    model_name = model_name.lower()
    if model_name in {"nb", "naive_bayes", "multinomialnb"}:
        return {
            'clf__alpha': [0.3, 0.7, 1.0, 1.5],
        }
    if model_name in {"logreg", "logistic", "logistic_regression"}:
        return {
            'clf__C': [0.25, 1.0, 4.0],
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear'],
        }
    if model_name in {"linearsvm", "svm", "linsvm"}:
        return {
            'clf__C': [0.5, 1.0, 2.0],
        }
    return {}

