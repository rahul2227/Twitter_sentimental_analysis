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

