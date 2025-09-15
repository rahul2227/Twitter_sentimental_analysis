from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, ParameterGrid
from sklearn.pipeline import Pipeline
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .data import load_datasets, ensure_output_dirs, save_eda_plots
from .evaluate import compute_metrics, save_confusion_matrix, get_prediction_scores, compute_top_features, compute_loss
from sklearn.base import clone
from .model import build_pipeline, default_param_grid, VectorizerConfig, build_vectorizer, build_classifier
from .preprocess import PreprocessorConfig


def _parse_tuple(s: str) -> tuple[int, int]:
    a, b = s.split(',')
    return int(a), int(b)


def _setup_logging(base_out: Path, level: str | int = "INFO", filename: str = "train.log") -> logging.Logger:
    if isinstance(level, str):
        lvl = getattr(logging, level.upper(), logging.INFO)
    else:
        lvl = int(level)
    logger = logging.getLogger("twitter_sentiment")
    logger.setLevel(lvl)
    # Avoid duplicate handlers on repeat runs
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    try:
        log_path = base_out / 'reports' / filename
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    return logger


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
    use_hashing: bool = False,
    hashing_features: int = 2 ** 20,
    test_size: float = 0.2,
    random_state: int = 42,
    do_grid: bool = True,
    balance: str = 'none',  # none|downsample|upsample
    calibrate_threshold: bool = False,
    select_k: int = 0,
    calibrate_probabilities: bool = False,
    resume: bool = False,
    # Logging / progress
    log_level: str = "INFO",
    progress: bool = True,
    # Preprocessing configuration
    lowercase: bool = True,
    max_repeat: int = 3,
    hashtag_policy: str = 'strip',  # strip|keep|segment
    replace_url: bool = True,
    replace_user: bool = True,
    number_policy: str = 'drop',  # keep|drop|token
    emoji_policy: str = 'ignore',  # ignore|drop|map
    use_emoticons: bool = True,
    contractions: str = 'none',  # none|default
    negation_scope: int = 0,
) -> None:
    """End-to-end training with val split and final test evaluation."""
    # Prepare outputs
    base_out = ensure_output_dirs(outputs_dir)
    logger = _setup_logging(base_out, log_level)
    logger.info("Starting offline training | model=%s | outputs=%s", model, outputs_dir)

    # EDA and data loading; prefer data/ fallback to provided paths
    if train_csv == 'train.csv' and Path('data/train.csv').exists():
        train_csv = 'data/train.csv'
    if test_csv == 'test.csv' and Path('data/test.csv').exists():
        test_csv = 'data/test.csv'
    logger.info("Loading datasets: train=%s, test=%s", train_csv, test_csv)
    train_df, test_df = load_datasets(train_csv, test_csv)
    logger.info("Loaded train=%d rows, test=%d rows", len(train_df), len(test_df))
    save_eda_plots(train_df, base_out / 'figures')

    # Split into train/val
    X = train_df['tweet'].astype(str).values
    y = train_df['label'].values
    logger.info("Splitting train/val: test_size=%.2f, stratify=True", test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info("Split: train=%d, val=%d", len(y_train), len(y_val))

    # Optional balancing on training split only
    rng = np.random.default_rng(seed=random_state)
    X_train, y_train = _balance_data(X_train, y_train, balance, rng)
    if balance != 'none':
        logger.info("Applied balancing=%s | train size now=%d", balance, len(y_train))

    # Vectorizer config
    vec_cfg = VectorizerConfig(
        word_ngrams=_parse_tuple(word_ngrams),
        use_char_ngrams=use_char_ngrams,
        char_ngrams=_parse_tuple(char_ngrams),
        use_hashing=use_hashing,
        hashing_features=int(hashing_features),
    )
    logger.info("Vectorizer config: %s", asdict(vec_cfg))

    # Preprocessor config
    contr_map = None
    if contractions == 'default':
        # signal to Tokenizer to enable defaults by passing a sentinel dict
        contr_map = {'__use__': 'default'}  # Tokenizer accepts 'default' string too; using dict sentinel to avoid mypy issues
        contr_map = 'default'  # actual signal expected in Tokenizer
    pre_cfg = PreprocessorConfig(
        lowercase=lowercase,
        max_repeat_char=max_repeat,
        hashtag_policy=hashtag_policy,
        replace_url=replace_url,
        replace_user=replace_user,
        number_policy=number_policy,
        emoji_policy=emoji_policy,
        emoticons=use_emoticons,
        contractions=contr_map,
        negation_scope=(None if negation_scope <= 0 else negation_scope),
    )
    logger.info("Preprocessor config: %s", asdict(pre_cfg))

    # Build or resume pipeline
    artifacts_dir = base_out / 'artifacts'
    reports_dir = base_out / 'reports'
    figures_dir = base_out / 'figures'

    resume_pipe = None
    if resume:
        ckpt = artifacts_dir / 'model_pipeline.joblib'
        if ckpt.exists():
            try:
                resume_pipe = joblib.load(ckpt)
            except Exception:
                resume_pipe = None

    if resume_pipe is not None:
        # Attempt to reuse loaded vec+clf when supported
        vec = getattr(resume_pipe, 'named_steps', {}).get('vec')
        clf = getattr(resume_pipe, 'named_steps', {}).get('clf')
        if vec is not None and clf is not None and hasattr(clf, 'partial_fit'):
            pipeline = Pipeline([('vec', vec), ('clf', clf)])
            use_resume = True
        else:
            # Fallback to fresh pipeline
            pipeline = build_pipeline(
                model, vec_cfg, pre_cfg,
                select_k=(select_k if select_k and select_k > 0 else None),
                calibrate_probabilities=calibrate_probabilities,
            )
            use_resume = False
    else:
        pipeline = build_pipeline(
            model, vec_cfg, pre_cfg,
            select_k=(select_k if select_k and select_k > 0 else None),
            calibrate_probabilities=calibrate_probabilities,
        )
        use_resume = False

    logger.info("Pipeline built%s", " (resumed components)" if resume_pipe is not None else "")

    # Optional grid search on train split
    best_params: Optional[dict] = None
    # If probability calibration is requested, or resuming, skip grid search.
    if do_grid and not calibrate_probabilities and not use_resume:
        param_grid = default_param_grid(model)
        if param_grid:
            n_cand = len(ParameterGrid(param_grid))
            n_folds = 3
            logger.info("Grid search: %d candidates x %d folds = %d fits", n_cand, n_folds, n_cand * n_folds)
            gs = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=3,
                scoring='f1_macro',
                n_jobs=1,
                verbose=(2 if progress else 0),
            )
            t0 = time.time()
            gs.fit(X_train, y_train)
            logger.info("Grid search done in %.2fs", time.time() - t0)
            pipeline = gs.best_estimator_
            best_params = gs.best_params_
            if best_params:
                logger.info("Best params: %s", best_params)
        else:
            logger.info("Fitting pipeline on train split (no grid)...")
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            logger.info("Fit complete in %.2fs", time.time() - t0)
    else:
        if use_resume and hasattr(pipeline.named_steps['clf'], 'partial_fit'):
            # Use existing fitted vectorizer; transform and partial_fit
            vec = pipeline.named_steps['vec']
            Xv_tr = vec.transform(X_train)
            clf = pipeline.named_steps['clf']
            classes = np.unique(y_train)
            try:
                logger.info("Resuming with partial_fit on train split...")
                if not hasattr(clf, 'classes_'):
                    clf.partial_fit(Xv_tr, y_train, classes=classes)
                else:
                    clf.partial_fit(Xv_tr, y_train)
            except Exception:
                # Fallback full fit
                logger.info("partial_fit failed; falling back to full fit...")
                pipeline.fit(X_train, y_train)
        else:
            logger.info("Fitting pipeline on train split...")
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            logger.info("Fit complete in %.2fs", time.time() - t0)

    # Validation metrics, with optional threshold calibration
    val_scores = get_prediction_scores(pipeline, X_val)
    threshold = None
    if calibrate_threshold and val_scores is not None:
        threshold = _calibrate_threshold(y_val, val_scores)
        val_pred = (val_scores >= threshold).astype(int)
    else:
        val_pred = pipeline.predict(X_val)
    val_metrics = compute_metrics(y_val, val_pred, scores=val_scores)
    try:
        f1m = float(val_metrics.get('f1_macro', 0.0)) if val_metrics else None
        acc = float(val_metrics.get('accuracy', 0.0)) if val_metrics else None
        thr_note = f", thr={threshold:.3f}" if threshold is not None else ""
        logging.getLogger("twitter_sentiment").info("Validation: f1_macro=%.4f, acc=%.4f%s", f1m, acc, thr_note)
    except Exception:
        pass

    # Retrain on full train (optionally balanced on full set as well)
    X_full, y_full = X, y
    if balance != 'none':
        X_full, y_full = _balance_data(X_full, y_full, balance, rng)
    if use_resume and hasattr(pipeline.named_steps['clf'], 'partial_fit'):
        vec = pipeline.named_steps['vec']
        Xv_full = vec.transform(X_full)
        clf = pipeline.named_steps['clf']
        try:
            clf.partial_fit(Xv_full, y_full)
        except Exception:
            pipeline.fit(X_full, y_full)
    else:
        logging.getLogger("twitter_sentiment").info("Fitting pipeline on full train set...")
        t0 = time.time()
        pipeline.fit(X_full, y_full)
        logging.getLogger("twitter_sentiment").info("Full fit complete in %.2fs", time.time() - t0)

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
    joblib.dump(pipeline, artifacts_dir / 'model_pipeline.joblib')
    logging.getLogger("twitter_sentiment").info("Saved model pipeline → %s", str(artifacts_dir / 'model_pipeline.joblib'))

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
    logging.getLogger("twitter_sentiment").info("Wrote metrics → %s", str(reports_dir / 'metrics.json'))

    # Confusion matrices
    if val_metrics and 'confusion_matrix' in val_metrics:
        cm = np.array(val_metrics['confusion_matrix'])
        save_confusion_matrix(cm, figures_dir / 'confusion_matrix_val.png', class_names=['0', '1'])
        logging.getLogger("twitter_sentiment").info("Saved validation confusion matrix → %s", str(figures_dir / 'confusion_matrix_val.png'))
    if test_metrics and 'confusion_matrix' in test_metrics:
        cm = np.array(test_metrics['confusion_matrix'])
        save_confusion_matrix(cm, figures_dir / 'confusion_matrix_test.png', class_names=['0', '1'])
        logging.getLogger("twitter_sentiment").info("Saved test confusion matrix → %s", str(figures_dir / 'confusion_matrix_test.png'))

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
    logging.getLogger("twitter_sentiment").info("Wrote validation predictions → %s", str(reports_dir / 'val_predictions.csv'))

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
    logging.getLogger("twitter_sentiment").info("Wrote test predictions → %s", str(reports_dir / 'test_predictions.csv'))

    # Top features (if available)
    try:
        top = compute_top_features(pipeline, top_k=30)
        if top is not None:
            (reports_dir / 'top_features.json').write_text(json.dumps(top, indent=2))
            logging.getLogger("twitter_sentiment").info("Saved top features → %s", str(reports_dir / 'top_features.json'))
    except Exception:
        pass

    # Offline training curve (loss vs. train fraction)
    try:
        fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
        val_losses = []
        train_losses = []
        n = len(X_full)
        rng_idx = np.random.default_rng(seed=random_state)
        for f in fracs:
            k = max(2, int(n * f))
            # Stratified subsample indices
            try:
                idx = rng_idx.permutation(n)[:k]
            except Exception:
                idx = np.arange(k)
            X_sub = X_full[idx]
            y_sub = y_full[idx]
            model_clone = clone(pipeline)
            model_clone.fit(X_sub, y_sub)
            # Train loss on subset
            try:
                s_tr = get_prediction_scores(model_clone, X_sub)
            except Exception:
                s_tr = None
            p_tr = model_clone.predict(X_sub)
            train_losses.append(compute_loss(y_sub, p_tr, s_tr))
            # Val loss on held-out val
            try:
                s_v = get_prediction_scores(model_clone, X_val)
            except Exception:
                s_v = None
            p_v = model_clone.predict(X_val)
            val_losses.append(compute_loss(y_val, p_v, s_v))

        curve = {
            'fractions': fracs,
            'train_loss': [float(x) for x in train_losses],
            'val_loss': [float(x) for x in val_losses],
        }
        (reports_dir / 'training_curve.json').write_text(json.dumps(curve, indent=2))
        logging.getLogger("twitter_sentiment").info("Saved offline training curve → %s", str(reports_dir / 'training_curve.json'))
    except Exception:
        pass


def online_train_and_evaluate(
    train_csv: str,
    outputs_dir: str = "outputs_online",
    model: str = "nb",
    use_char_ngrams: bool = False,
    word_ngrams: str = "1,2",
    char_ngrams: str = "3,5",
    use_hashing: bool = False,
    hashing_features: int = 2 ** 20,
    random_state: int = 42,
    batch_size: int = 4096,
    epochs: int = 1,
    eval_every: int = 1,
    checkpoint_every: int = 5,
    # Preprocessing configuration
    lowercase: bool = True,
    max_repeat: int = 3,
    hashtag_policy: str = 'strip',
    replace_url: bool = True,
    replace_user: bool = True,
    number_policy: str = 'drop',
    emoji_policy: str = 'ignore',
    use_emoticons: bool = True,
    contractions: str = 'none',
    negation_scope: int = 0,
    test_size: float = 0.2,
    stop_event=None,
    resume: bool = True,
    # Logging / progress
    log_level: str = "INFO",
    progress: bool = True,
) -> None:
    """Incremental training with partial_fit on streaming batches.

    Writes rolling validation metrics to reports/metrics_online.json and saves
    checkpoints to artifacts/model_pipeline.joblib.
    """
    rng = np.random.default_rng(seed=random_state)

    # Prepare outputs
    base_out = ensure_output_dirs(outputs_dir)
    artifacts_dir = base_out / 'artifacts'
    reports_dir = base_out / 'reports'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(base_out, log_level, filename='train_online.log')
    logger.info("Starting online training | model=%s | outputs=%s", model, outputs_dir)

    # Resolve default data path
    if train_csv == 'train.csv' and Path('data/train.csv').exists():
        train_csv = 'data/train.csv'

    # Build configs
    vec_cfg = VectorizerConfig(
        word_ngrams=_parse_tuple(word_ngrams),
        use_char_ngrams=use_char_ngrams,
        char_ngrams=_parse_tuple(char_ngrams),
        use_hashing=use_hashing,
        hashing_features=int(hashing_features),
    )
    contr_map = None
    if contractions == 'default':
        contr_map = 'default'
    pre_cfg = PreprocessorConfig(
        lowercase=lowercase,
        max_repeat_char=max_repeat,
        hashtag_policy=hashtag_policy,
        replace_url=replace_url,
        replace_user=replace_user,
        number_policy=number_policy,
        emoji_policy=emoji_policy,
        emoticons=use_emoticons,
        contractions=contr_map,
        negation_scope=(None if negation_scope <= 0 else negation_scope),
    )

    # Build or resume components
    vec = None
    clf = None
    fitted_vec = False
    pipe_path = artifacts_dir / 'model_pipeline.joblib'
    if resume and pipe_path.exists():
        try:
            pipe = joblib.load(pipe_path)
            vec = pipe.named_steps.get('vec')
            clf = pipe.named_steps.get('clf')
            fitted_vec = True
        except Exception:
            vec = None
            clf = None
    if vec is None or clf is None:
        vec = build_vectorizer(vec_cfg, pre_cfg)
        clf = build_classifier(model)
    if not hasattr(clf, 'partial_fit'):
        raise ValueError(f"Model '{model}' does not support online training (partial_fit)")

    # Scan classes and total rows (labels only)
    classes_set = set()
    total_rows = 0
    for ch in pd.read_csv(train_csv, usecols=['label'], chunksize=max(batch_size, 1024)):
        lbl = ch['label']
        lbl = lbl[~pd.isna(lbl)]
        total_rows += lbl.shape[0]
        try:
            classes_set.update(lbl.astype(int).unique().tolist())
        except Exception:
            # fall back to direct values if non-int labels
            classes_set.update(lbl.unique().tolist())
    classes = np.array(sorted(list(classes_set))) if classes_set else np.array([0, 1])
    logger.info("Discovered classes: %s | labeled rows=%d", list(classes), total_rows)

    # Training loop with a held-out val from the first chunk
    # fitted_vec may already be True if resuming
    first_val_taken = False
    X_val_text = None
    y_val = None
    batch_idx = 0
    samples_seen = 0
    last_eval_time = time.time()

    train_curve = []  # list of {'step': i, 'loss': float}

    def _write_metrics_json(val_m):
        payload = {
            'mode': 'online',
            'model': model,
            'vectorizer': asdict(vec_cfg),
            'preprocessing': asdict(pre_cfg),
            'random_state': random_state,
            'batch_size': batch_size,
            'epochs': epochs,
            'classes': list(map(int, classes)) if len(classes) and np.issubdtype(np.array(classes).dtype, np.number) else list(map(str, classes)),
            'samples_seen': samples_seen,
            'total_rows': total_rows,
            'epoch': epoch,
            'batch': batch_idx,
            'validation': val_m,
            'train_curve': train_curve[-500:],  # keep last 500 points to limit size
        }
        (reports_dir / 'metrics_online.json').write_text(json.dumps(payload, indent=2))
        try:
            logger.info(
                "step=%d | epoch=%d | seen=%d | val f1=%.4f acc=%.4f",
                samples_seen,
                epoch,
                samples_seen,
                float((val_m or {}).get('f1_macro', 0.0)),
                float((val_m or {}).get('accuracy', 0.0)),
            )
        except Exception:
            pass

    total_target = total_rows * epochs
    use_bar = bool(progress and tqdm is not None)
    pbar = tqdm(total=total_target, desc="online train", unit="samples", leave=False) if use_bar else None

    for epoch in range(epochs):
        logger.info("Epoch %d/%d", epoch + 1, epochs)
        for ch in pd.read_csv(train_csv, usecols=['tweet', 'label'], chunksize=batch_size):
            if stop_event is not None and getattr(stop_event, 'is_set', lambda: False)():
                break
            dfc = ch.dropna(subset=['label']).copy()
            if dfc.empty:
                continue
            Xc = dfc['tweet'].astype(str).values
            yc = dfc['label'].values
            # Establish validation split from the first batch only
            if not first_val_taken:
                try:
                    Xc, X_val_text, yc, y_val = train_test_split(
                        Xc, yc, test_size=test_size, stratify=yc, random_state=random_state
                    )
                    first_val_taken = True
                except Exception:
                    # fallback without stratify
                    n_val = max(1, int(len(Xc) * test_size))
                    X_val_text, y_val, Xc, yc = Xc[:n_val], yc[:n_val], Xc[n_val:], yc[n_val:]

            # Fit vectorizer on first training portion, then only transform
            if not fitted_vec:
                Xv = vec.fit_transform(Xc)
                clf.partial_fit(Xv, yc, classes=classes)
                fitted_vec = True
            else:
                Xv = vec.transform(Xc)
                clf.partial_fit(Xv, yc)

            batch_idx += 1
            samples_seen += len(yc)
            if pbar is not None:
                pbar.update(len(yc))

            # Compute training loss on current batch
            try:
                s_tr = None
                if hasattr(clf, 'predict_proba'):
                    s_tr = clf.predict_proba(Xv)[:, -1]
                elif hasattr(clf, 'decision_function'):
                    df = clf.decision_function(Xv)
                    s_tr = df if df.ndim == 1 else df[:, -1]
            except Exception:
                s_tr = None
            p_tr = clf.predict(Xv)
            try:
                batch_loss = compute_loss(yc, p_tr, s_tr)
                train_curve.append({'step': int(samples_seen), 'loss': float(batch_loss)})
            except Exception:
                pass

            # Periodic evaluation and checkpoint
            do_eval = (eval_every > 0 and (batch_idx % eval_every == 0))
            if do_eval and first_val_taken and X_val_text is not None:
                Xvv = vec.transform(X_val_text)
                # For linear models we may not have proba; use decision or None
                try:
                    if hasattr(clf, 'predict_proba'):
                        scores = clf.predict_proba(Xvv)[:, -1]
                    elif hasattr(clf, 'decision_function'):
                        df = clf.decision_function(Xvv)
                        scores = df if df.ndim == 1 else df[:, -1]
                    else:
                        scores = None
                except Exception:
                    scores = None
                preds = clf.predict(Xvv)
                val_m = compute_metrics(y_val, preds, scores=scores)
                _write_metrics_json(val_m)
                if pbar is not None:
                    try:
                        pbar.set_postfix({
                            'f1': f"{float(val_m.get('f1_macro', 0.0)):.4f}",
                            'acc': f"{float(val_m.get('accuracy', 0.0)):.4f}"
                        }, refresh=False)
                    except Exception:
                        pass

            if checkpoint_every > 0 and (batch_idx % checkpoint_every == 0):
                pipe = Pipeline([('vec', vec), ('clf', clf)])
                joblib.dump(pipe, artifacts_dir / 'model_pipeline.joblib')
                try:
                    logger.info("Saved checkpoint at batch %d → %s", batch_idx, str(artifacts_dir / 'model_pipeline.joblib'))
                except Exception:
                    pass

        if stop_event is not None and getattr(stop_event, 'is_set', lambda: False)():
            break

    # Final checkpoint and metrics write
    if first_val_taken and X_val_text is not None:
        Xvv = vec.transform(X_val_text)
        try:
            if hasattr(clf, 'predict_proba'):
                scores = clf.predict_proba(Xvv)[:, -1]
            elif hasattr(clf, 'decision_function'):
                df = clf.decision_function(Xvv)
                scores = df if df.ndim == 1 else df[:, -1]
            else:
                scores = None
        except Exception:
            scores = None
        preds = clf.predict(Xvv)
        val_m = compute_metrics(y_val, preds, scores=scores)
        _write_metrics_json(val_m)

    if pbar is not None:
        pbar.close()
    pipe = Pipeline([('vec', vec), ('clf', clf)])
    joblib.dump(pipe, artifacts_dir / 'model_pipeline.joblib')
    logger.info("Saved final model → %s", str(artifacts_dir / 'model_pipeline.joblib'))


def main():
    import argparse

    p = argparse.ArgumentParser(description='Train and evaluate tweet sentiment models.')
    p.add_argument('--train', default='train.csv', help='Path to training CSV (or data/train.csv if exists)')
    p.add_argument('--test', default='test.csv', help='Path to test CSV (or data/test.csv if exists)')
    p.add_argument('--outputs', default='outputs', help='Directory to save outputs')
    p.add_argument('--online', action='store_true', help='Use online/streaming training with partial_fit')
    p.add_argument('--log-level', default='INFO', choices=['CRITICAL','ERROR','WARNING','INFO','DEBUG'], help='Logging level')
    p.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bars and extra verbosity')
    p.add_argument('--model', default='nb', choices=['nb', 'cnb', 'logreg', 'logreg_l1', 'linearsvm', 'sgd', 'sgd_hinge', 'pa'], help='Model type')
    p.add_argument('--use-char-ngrams', action='store_true', help='Include char-level n-grams')
    p.add_argument('--use-hashing', action='store_true', help='Use HashingVectorizer instead of CountVectorizer')
    p.add_argument('--hashing-features', type=int, default=1048576, help='Number of hashing features (default 2^20)')
    p.add_argument('--word-ngrams', default='1,2', help='Word n-gram range, e.g., 1,2')
    p.add_argument('--char-ngrams', default='3,5', help='Char n-gram range, e.g., 3,5')
    p.add_argument('--test-size', type=float, default=0.2, help='Validation size fraction')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--no-grid', action='store_true', help='Disable GridSearchCV')
    p.add_argument('--balance', default='none', choices=['none', 'downsample', 'upsample'], help='Class balancing strategy')
    p.add_argument('--calibrate-threshold', action='store_true', help='Calibrate decision threshold on validation set')
    p.add_argument('--select-k', type=int, default=0, help='Optional chi^2 feature selection (SelectKBest)')
    p.add_argument('--calibrate-probabilities', action='store_true', help='Wrap classifier with CalibratedClassifierCV for probabilities')
    p.add_argument('--resume', action='store_true', help='Offline: resume from checkpoint if exists (partial_fit models)')
    # Online training flags
    p.add_argument('--batch-size', type=int, default=4096, help='Online mode: batch size for partial_fit')
    p.add_argument('--epochs', type=int, default=1, help='Online mode: number of passes over the data')
    p.add_argument('--eval-every', type=int, default=1, help='Online mode: evaluate validation metrics every N batches')
    p.add_argument('--checkpoint-every', type=int, default=5, help='Online mode: save checkpoint every N batches')
    # Preprocessing flags
    p.add_argument('--no-lowercase', action='store_true', help='Disable lowercasing')
    p.add_argument('--max-repeat', type=int, default=3, help='Max consecutive repeated chars to keep (0 disables)')
    p.add_argument('--hashtag-policy', default='strip', choices=['strip', 'keep', 'segment'], help='Hashtag handling strategy')
    p.add_argument('--no-replace-url', action='store_true', help='Do not replace URLs with <url>')
    p.add_argument('--no-replace-user', action='store_true', help='Do not replace @mentions with <user>')
    p.add_argument('--number-policy', default='drop', choices=['keep', 'drop', 'token'], help='Number handling policy')
    p.add_argument('--emoji-policy', default='ignore', choices=['ignore', 'drop', 'map'], help='Emoji handling policy')
    p.add_argument('--no-emoticons', action='store_true', help='Disable emoticon mapping (e.g., :-) -> smile)')
    p.add_argument('--contractions', default='none', choices=['none', 'default'], help='Contraction expansion policy')
    p.add_argument('--negation-scope', type=int, default=0, help='Apply _neg suffix to next N tokens after a negation cue (0 disables)')

    args = p.parse_args()

    if args.online:
        online_train_and_evaluate(
            train_csv=args.train,
            outputs_dir=args.outputs,
            model=args.model,
            use_char_ngrams=args.use_char_ngrams,
            word_ngrams=args.word_ngrams,
            char_ngrams=args.char_ngrams,
            use_hashing=args.use_hashing,
            hashing_features=args.hashing_features,
            random_state=args.seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            eval_every=args.eval_every,
            checkpoint_every=args.checkpoint_every,
            log_level=args.log_level,
            progress=(not args.no_progress),
            lowercase=(not args.no_lowercase),
            max_repeat=args.max_repeat,
            hashtag_policy=args.hashtag_policy,
            replace_url=(not args.no_replace_url),
            replace_user=(not args.no_replace_user),
            number_policy=args.number_policy,
            emoji_policy=args.emoji_policy,
            use_emoticons=(not args.no_emoticons),
            contractions=args.contractions,
            negation_scope=args.negation_scope,
            test_size=args.test_size,
        )
        return

    train_and_evaluate(
        train_csv=args.train,
        test_csv=args.test,
        outputs_dir=args.outputs,
        model=args.model,
        use_char_ngrams=args.use_char_ngrams,
        word_ngrams=args.word_ngrams,
        char_ngrams=args.char_ngrams,
        use_hashing=args.use_hashing,
        hashing_features=args.hashing_features,
        test_size=args.test_size,
        random_state=args.seed,
        do_grid=not args.no_grid,
        select_k=args.select_k,
        calibrate_probabilities=args.calibrate_probabilities,
        balance=args.balance,
        calibrate_threshold=args.calibrate_threshold,
        resume=args.resume,
        log_level=args.log_level,
        progress=(not args.no_progress),
        lowercase=(not args.no_lowercase),
        max_repeat=args.max_repeat,
        hashtag_policy=args.hashtag_policy,
        replace_url=(not args.no_replace_url),
        replace_user=(not args.no_replace_user),
        number_policy=args.number_policy,
        emoji_policy=args.emoji_policy,
        use_emoticons=(not args.no_emoticons),
        contractions=args.contractions,
        negation_scope=args.negation_scope,
    )


if __name__ == '__main__':
    main()
