from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, chi2

from .preprocess import Tokenizer, PreprocessorConfig


@dataclass
class VectorizerConfig:
    word_ngrams: Tuple[int, int] = (1, 2)
    use_char_ngrams: bool = False
    char_ngrams: Tuple[int, int] = (3, 5)
    use_hashing: bool = False
    hashing_features: int = 2 ** 20


def build_vectorizer(cfg: VectorizerConfig, pre_cfg: PreprocessorConfig | None = None) -> Pipeline:
    """Create a vectorization pipeline combining word and optional char n-grams.

    Uses a picklable `Tokenizer` callable to allow configurable preprocessing.
    """
    tokenizer = Tokenizer(pre_cfg)
    if cfg.use_hashing:
        word_vec = HashingVectorizer(
            analyzer='word',
            tokenizer=tokenizer,
            token_pattern=None,
            ngram_range=cfg.word_ngrams,
            n_features=cfg.hashing_features,
            alternate_sign=False,
        )
        if cfg.use_char_ngrams:
            char_vec = HashingVectorizer(
                analyzer='char',
                ngram_range=cfg.char_ngrams,
                n_features=cfg.hashing_features,
                alternate_sign=False,
            )
            union = FeatureUnion([('word', word_vec), ('char', char_vec)])
            return Pipeline([('union', union), ('tfidf', TfidfTransformer())])
        else:
            return Pipeline([('bow', word_vec), ('tfidf', TfidfTransformer())])
    else:
        # When providing a custom tokenizer, explicitly set token_pattern=None
        # to avoid scikit-learn warnings about it being ignored.
        word_vec = CountVectorizer(
            analyzer='word',
            tokenizer=tokenizer,
            token_pattern=None,
            ngram_range=cfg.word_ngrams,
        )
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
    if name in {"cnb", "complementnb"}:
        return ComplementNB()
    if name in {"logreg", "logistic", "logistic_regression"}:
        return LogisticRegression(max_iter=1000, n_jobs=1, class_weight='balanced')
    if name in {"logreg_l1"}:
        return LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', class_weight='balanced')
    if name in {"linearsvm", "svm", "linsvm"}:
        return LinearSVC(class_weight='balanced')
    if name in {"sgd", "sgd_log"}:
        return SGDClassifier(loss='log_loss', max_iter=1000, class_weight='balanced')
    if name in {"sgd_hinge"}:
        return SGDClassifier(loss='hinge', max_iter=1000, class_weight='balanced')
    if name in {"pa", "passive_aggressive"}:
        return PassiveAggressiveClassifier(max_iter=1000, class_weight='balanced')
    if name in {"linearsvm_cal"}:  # used with calibrated probabilities
        return LinearSVC(class_weight='balanced')
    raise ValueError(
        "Unknown model '" + name + "'. Expected one of: nb, cnb, logreg, logreg_l1, linearsvm, sgd, sgd_hinge, pa, linearsvm_cal"
    )


def build_pipeline(
    model_name: str,
    vec_cfg: VectorizerConfig,
    pre_cfg: PreprocessorConfig | None = None,
    select_k: Optional[int] = None,
    calibrate_probabilities: bool = False,
) -> Pipeline:
    """Build full pipeline: vectorizer(s) + optional feature selection + classifier.

    If `calibrate_probabilities` is True, wraps the classifier with
    CalibratedClassifierCV for probability estimates (useful for LinearSVC/hinge).
    """
    vec = build_vectorizer(vec_cfg, pre_cfg)
    steps = [('vec', vec)]
    if select_k is not None and select_k > 0:
        steps.append(('select', SelectKBest(chi2, k=select_k)))
    base_clf = build_classifier(model_name)
    clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=3) if calibrate_probabilities else base_clf
    steps.append(('clf', clf))
    return Pipeline(steps)


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
    if model_name in {"cnb", "complementnb"}:
        return {
            'clf__alpha': [0.3, 0.7, 1.0, 1.5],
        }
    if model_name in {"sgd", "sgd_log", "sgd_hinge"}:
        return {
            'clf__alpha': [1e-5, 1e-4, 1e-3],
            'clf__penalty': ['l2'],
        }
    if model_name in {"pa", "passive_aggressive"}:
        return {
            'clf__C': [0.5, 1.0, 2.0],
        }
    if model_name in {"logreg_l1"}:
        return {
            'clf__C': [0.25, 1.0, 4.0],
            'clf__penalty': ['l1'],
            'clf__solver': ['liblinear'],
        }
    return {}
