"""Evaluation helpers shared by training and reporting."""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)


def evaluate_binary_classifier(y_true, probabilities, threshold=0.5):
    y_pred = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=["Stayed", "Churned"],
                                   zero_division=0, output_dict=True)
    return metrics, cm, report


def metrics_table(results):
    return pd.DataFrame(results).T.rename_axis("model").reset_index()
