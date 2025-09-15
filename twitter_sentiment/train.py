from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from .data import load_datasets, ensure_output_dirs, save_eda_plots
from .evaluate import compute_metrics, save_confusion_matrix, get_prediction_scores
from .model import build_pipeline, default_param_grid, VectorizerConfig


def _parse_tuple(s: str) -> tuple[int, int]:
    a, b = s.split(',')
    return int(a), int(b)


def _balance_data(X: np.ndarray, y: np.ndarray, method: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if method == 'none':
        return X, y
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return X, y
    maj_class = classes[np.argmax(counts)]
    min_class = classes[np.argmin(counts)]
    X_maj = X[y == maj_class]
    X_min = X[y == min_class]
    n_maj = X_maj.shape[0]
    n_min = X_min.shape[0]
    if method == 'downsample':
        idx = rng.choice(n_maj, size=n_min, replace=False)
        X_bal = np.concatenate([X_maj[idx], X_min], axis=0)
        y_bal = np.concatenate([np.full(n_min, maj_class), np.full(n_min, min_class)])
    elif method == 'upsample':
        idx = rng.choice(n_min, size=n_maj, replace=True)
        X_bal = np.concatenate([X_maj, X_min[idx]], axis=0)
        y_bal = np.concatenate([np.full(n_maj, maj_class), np.full(n_maj, min_class)])
    else:
        return X, y
    # Shuffle
    perm = rng.permutation(X_bal.shape[0])
    return X_bal[perm], y_bal[perm]


def _calibrate_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    # Choose threshold that maximizes macro-F1 on validation scores
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        m = compute_metrics(y_true, preds, scores=scores)
        f1 = float(m.get('f1_macro', 0.0))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train_and_evaluate(
    train_csv: str,
    test_csv: str,
    outputs_dir: str = "outputs",
    model: str = "nb",
    use_char_ngrams: bool = False,
    word_ngrams: str = "1,2",
    char_ngrams: str = "3,5",
    test_size: float = 0.2,
    random_state: int = 42,
    do_grid: bool = True,
    balance: str = 'none',  # none|downsample|upsample
    calibrate_threshold: bool = False,
) -> None:
    """End-to-end training with val split and final test evaluation."""
    # Prepare outputs
    base_out = ensure_output_dirs(outputs_dir)

    # EDA and data loading; prefer data/ fallback to provided paths
    if train_csv == 'train.csv' and Path('data/train.csv').exists():
        train_csv = 'data/train.csv'
    if test_csv == 'test.csv' and Path('data/test.csv').exists():
        test_csv = 'data/test.csv'
    train_df, test_df = load_datasets(train_csv, test_csv)
    save_eda_plots(train_df, base_out / 'figures')

    # Split into train/val
    X = train_df['tweet'].astype(str).values
    y = train_df['label'].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Optional balancing on training split only
    rng = np.random.default_rng(seed=random_state)
    X_train, y_train = _balance_data(X_train, y_train, balance, rng)

    # Vectorizer config
    vec_cfg = VectorizerConfig(
        word_ngrams=_parse_tuple(word_ngrams),
        use_char_ngrams=use_char_ngrams,
        char_ngrams=_parse_tuple(char_ngrams),
    )

    # Build pipeline
    pipeline = build_pipeline(model, vec_cfg)

    # Optional grid search on train split
    best_params: Optional[dict] = None
    if do_grid:
        param_grid = default_param_grid(model)
        if param_grid:
            gs = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=3,
                scoring='f1_macro',
                n_jobs=1,
                verbose=0,
            )
            gs.fit(X_train, y_train)
            pipeline = gs.best_estimator_
            best_params = gs.best_params_
        else:
            pipeline.fit(X_train, y_train)
    else:
        pipeline.fit(X_train, y_train)

    # Validation metrics, with optional threshold calibration
    val_scores = get_prediction_scores(pipeline, X_val)
    threshold = None
    if calibrate_threshold and val_scores is not None:
        threshold = _calibrate_threshold(y_val, val_scores)
        val_pred = (val_scores >= threshold).astype(int)
    else:
        val_pred = pipeline.predict(X_val)
    val_metrics = compute_metrics(y_val, val_pred, scores=val_scores)

    # Retrain on full train (optionally balanced on full set as well)
    X_full, y_full = X, y
    if balance != 'none':
        X_full, y_full = _balance_data(X_full, y_full, balance, rng)
    pipeline.fit(X_full, y_full)

    # Test evaluation (held out)
    X_test = test_df['tweet'].astype(str).values
    has_test_labels = 'label' in test_df.columns
    # Predict for all test rows
    test_scores_all = get_prediction_scores(pipeline, X_test)
    if threshold is not None and test_scores_all is not None:
        test_pred_all = (test_scores_all >= threshold).astype(int)
    else:
        test_pred_all = pipeline.predict(X_test)

    # Compute metrics only on rows that actually have labels (skip NaN)
    test_metrics = None
    y_test = None
    if has_test_labels:
        y_test_full = test_df['label'].values
        import pandas as _pd  # lazy import to avoid top-level dependency here
        mask = ~_pd.isna(y_test_full)
        if mask.any():
            y_test = y_test_full  # keep for CSV writing (may include NaN)
            scores_sub = test_scores_all[mask] if test_scores_all is not None else None
            preds_sub = test_pred_all[mask]
            labels_sub = y_test_full[mask]
            test_metrics = compute_metrics(labels_sub, preds_sub, scores=scores_sub)

    # Save artifacts
    artifacts_dir = base_out / 'artifacts'
    reports_dir = base_out / 'reports'
    figures_dir = base_out / 'figures'

    joblib.dump(pipeline, artifacts_dir / 'model_pipeline.joblib')

    # Metrics JSON
    out_metrics = {
        'model': model,
        'vectorizer': asdict(vec_cfg),
        'random_state': random_state,
        'test_size': test_size,
        'best_params': best_params,
        'balance': balance,
        'calibrated_threshold': threshold,
        'validation': val_metrics,
        'test': test_metrics,
    }
    (reports_dir / 'metrics.json').write_text(json.dumps(out_metrics, indent=2))

    # Confusion matrices
    if val_metrics and 'confusion_matrix' in val_metrics:
        cm = np.array(val_metrics['confusion_matrix'])
        save_confusion_matrix(cm, figures_dir / 'confusion_matrix_val.png', class_names=['0', '1'])
    if test_metrics and 'confusion_matrix' in test_metrics:
        cm = np.array(test_metrics['confusion_matrix'])
        save_confusion_matrix(cm, figures_dir / 'confusion_matrix_test.png', class_names=['0', '1'])

    # Prediction CSVs
    # Validation predictions
    val_rows = []
    for i, text in enumerate(X_val):
        row = {'text': text, 'y_pred': int(val_pred[i])}
        try:
            row_val = y_val[i]
            # allow non-numeric labels gracefully
            import pandas as _pd
            if _pd.notna(row_val):
                try:
                    row['y_true'] = int(row_val)
                except Exception:
                    row['y_true'] = str(row_val)
        except Exception:
            pass
        if val_scores is not None:
            try:
                row['score'] = float(val_scores[i])
            except Exception:
                pass
        val_rows.append(row)
    pd.DataFrame(val_rows).to_csv(reports_dir / 'val_predictions.csv', index=False)

    # Test predictions
    test_rows = []
    for i, text in enumerate(X_test):
        row = {'text': text, 'y_pred': int(test_pred_all[i])}
        if has_test_labels:
            try:
                val = test_df['label'].iloc[i]
                if _pd.notna(val):
                    row['y_true'] = int(val)
            except Exception:
                pass
        if test_scores_all is not None:
            try:
                row['score'] = float(test_scores_all[i])
            except Exception:
                pass
        test_rows.append(row)
    pd.DataFrame(test_rows).to_csv(reports_dir / 'test_predictions.csv', index=False)


def main():
    import argparse

    p = argparse.ArgumentParser(description='Train and evaluate tweet sentiment models.')
    p.add_argument('--train', default='train.csv', help='Path to training CSV (or data/train.csv if exists)')
    p.add_argument('--test', default='test.csv', help='Path to test CSV (or data/test.csv if exists)')
    p.add_argument('--outputs', default='outputs', help='Directory to save outputs')
    p.add_argument('--model', default='nb', choices=['nb', 'logreg', 'linearsvm'], help='Model type')
    p.add_argument('--use-char-ngrams', action='store_true', help='Include char-level n-grams')
    p.add_argument('--word-ngrams', default='1,2', help='Word n-gram range, e.g., 1,2')
    p.add_argument('--char-ngrams', default='3,5', help='Char n-gram range, e.g., 3,5')
    p.add_argument('--test-size', type=float, default=0.2, help='Validation size fraction')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--no-grid', action='store_true', help='Disable GridSearchCV')
    p.add_argument('--balance', default='none', choices=['none', 'downsample', 'upsample'], help='Class balancing strategy')
    p.add_argument('--calibrate-threshold', action='store_true', help='Calibrate decision threshold on validation set')

    args = p.parse_args()

    train_and_evaluate(
        train_csv=args.train,
        test_csv=args.test,
        outputs_dir=args.outputs,
        model=args.model,
        use_char_ngrams=args.use_char_ngrams,
        word_ngrams=args.word_ngrams,
        char_ngrams=args.char_ngrams,
        test_size=args.test_size,
        random_state=args.seed,
        do_grid=not args.no_grid,
        balance=args.balance,
        calibrate_threshold=args.calibrate_threshold,
    )


if __name__ == '__main__':
    main()
