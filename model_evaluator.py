"""
model_evaluator.py
------------------
Comprehensive evaluation engine calculating Accuracy, Precision, Recall, F1-score,
Confusion Matrix, and ROC-AUC data for news classification algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
from utils import get_logger, ModelEvaluationError

logger = get_logger("ModelEvaluator")


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    pos_label: int = 1
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics comparing predictions against benchmark labels.
    pos_label=1 (Fake News).
    """
    try:
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        rec = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))

        cm = confusion_matrix(y_true, y_pred)
        # For binary classification (0: Real, 1: Fake)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        roc_data = {}
        auc_score = 0.0
        if y_proba is not None and len(np.unique(y_true)) == 2:
            # y_proba is probability of class 1 (Fake)
            fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=pos_label)
            auc_score = float(roc_auc_score(y_true, y_proba))
            roc_data = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": auc_score
            }

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "raw_matrix": cm.tolist()
            },
            "roc_auc": roc_data
        }

        logger.info(
            f"Evaluation Complete | Acc: {acc*100:.1f}% | Prec: {prec*100:.1f}% | "
            f"Recall: {rec*100:.1f}% | F1: {f1*100:.1f}% | AUC: {auc_score:.3f}"
        )
        return metrics

    except Exception as e:
        logger.error(f"Failed to compute model evaluation metrics: {str(e)}")
        raise ModelEvaluationError(f"Evaluation failed: {str(e)}") from e
