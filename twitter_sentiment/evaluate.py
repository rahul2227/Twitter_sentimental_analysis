from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


def compute_metrics(y_true, y_pred, scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute standard metrics with correct (y_true, y_pred) ordering.

    If ``scores`` are provided (probabilities or decision function values),
    the binary ROC-AUC is computed when possible.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    out = {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'accuracy': acc,
        'f1_macro': f1_macro,
    }
    try:
        if scores is not None and len(np.unique(y_true)) == 2:
            out['roc_auc'] = float(roc_auc_score(y_true, scores))
    except Exception:
        pass
    return out


def get_prediction_scores(pipeline, X) -> Optional[np.ndarray]:
    """Return continuous scores for ROC-AUC from a fitted pipeline."""
    try:
        clf = getattr(pipeline, 'named_steps', {}).get('clf', None)
        if clf is None:
            return None
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba(X)
            classes = getattr(clf, 'classes_', None)
            idx = -1
            if classes is not None:
                try:
                    idx = int(np.where(classes == 1)[0][0]) if 1 in classes else -1
                except Exception:
                    idx = -1
            return proba[:, idx]
        if hasattr(pipeline, 'decision_function'):
            df = pipeline.decision_function(X)
            if df.ndim == 2:
                classes = getattr(clf, 'classes_', None)
                idx = 1 if classes is None else (int(np.where(classes == 1)[0][0]) if 1 in classes else -1)
                return df[:, idx]
            return df
    except Exception:
        return None
    return None


def save_confusion_matrix(cm: np.ndarray, out_path: Path, class_names=None) -> None:
    """Save confusion matrix heatmap to disk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _get_feature_names_from_vec(vec) -> Optional[list]:
    try:
        names = vec.get_feature_names_out()
        return list(map(str, names))
    except Exception:
        return None


def _get_linear_weights(clf) -> Optional[np.ndarray]:
    # Handles LinearSVC/SGD/LogReg shapes; returns 1D weights for binary problems
    try:
        if hasattr(clf, 'coef_'):
            w = clf.coef_
            return w.ravel()
        if hasattr(clf, 'feature_log_prob_'):
            flp = clf.feature_log_prob_
            if flp.shape[0] >= 2:
                return (flp[1] - flp[0]).ravel()
    except Exception:
        return None
    return None


def compute_top_features(pipeline, top_k: int = 20) -> Optional[Dict[str, Any]]:
    """Extract top positive/negative features for linear or NB-like models.

    Returns a dict with keys: 'top_positive', 'top_negative', 'feature_names', and
    optionally 'source' describing classifier type. If unavailable, returns None.
    """
    try:
        vec = pipeline.named_steps.get('vec')
        clf = pipeline.named_steps.get('clf')
        if vec is None or clf is None:
            return None

        # If calibrated, attempt to get underlying estimator
        base_clf = clf
        try:
            if clf.__class__.__name__ == 'CalibratedClassifierCV':
                base_clf = getattr(clf, 'estimator', clf)
                # estimator may be unfitted clone; try calibrated_classifiers_
                if hasattr(clf, 'calibrated_classifiers_') and clf.calibrated_classifiers_:
                    base_clf = clf.calibrated_classifiers_[0].estimator
        except Exception:
            pass

        names = _get_feature_names_from_vec(vec)
        if names is None:
            return None
        w = _get_linear_weights(base_clf)
        if w is None or len(w) != len(names):
            return None

        idx_pos = np.argsort(w)[-top_k:][::-1]
        idx_neg = np.argsort(w)[:top_k]
        top_pos = [{
            'feature': names[i],
            'weight': float(w[i])
        } for i in idx_pos]
        top_neg = [{
            'feature': names[i],
            'weight': float(w[i])
        } for i in idx_neg]
        return {
            'top_positive': top_pos,
            'top_negative': top_neg,
            'source': base_clf.__class__.__name__,
        }
    except Exception:
        return None


def compute_loss(y_true, y_pred, scores: Optional[np.ndarray] = None) -> float:
    """Compute a scalar loss value given true labels, predictions, and optional scores.

    Priority:
    - If scores are in [0,1], use log-loss on positive class probability.
    - Else if scores are decision function margins, use hinge loss.
    - Else, use (1 - macro-F1) as a generic loss proxy.
    """
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if scores is not None:
            scores = np.asarray(scores)
            smin, smax = float(np.nanmin(scores)), float(np.nanmax(scores))
            if smin >= 0.0 and smax <= 1.0 and scores.ndim == 1:
                probs = np.vstack([1.0 - scores, scores]).T
                return float(log_loss(y_true, probs, labels=[0, 1]))
            # hinge-like loss: y in {-1, +1}
            y_pm = np.where(y_true == 1, 1.0, -1.0)
            margins = scores if scores.ndim == 1 else scores[:, -1]
            loss = np.maximum(0.0, 1.0 - y_pm * margins)
            return float(np.mean(loss))
        # fallback
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return float(1.0 - f1)
    except Exception:
        return 1.0
