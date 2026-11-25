
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
    classification_report
)


CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]


def calculate_metrics(y_true, y_pred, y_prob):

    # Convert inputs to NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # Validate probability dimensions
    if y_prob.ndim != 2 or y_prob.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected probabilities with shape (N, 4), got {y_prob.shape}"
        )

    if len(y_true) != len(y_pred) or len(y_true) != len(y_prob):
        raise ValueError("y_true, y_pred and y_prob lengths do not match.")

    # Validate probability values
    if not np.isfinite(y_prob).all():
        raise ValueError("Probabilities contain NaN or infinite values.")

    if np.any(y_prob < 0):
        raise ValueError("Probabilities contain negative values.")

    # Normalize each probability row
    row_sums = y_prob.sum(axis=1)

    if np.any(row_sums <= 0):
        raise ValueError("Some probability rows have a zero sum.")

    y_prob = y_prob / row_sums[:, np.newaxis]

    # Calculate classification metrics
    result = {
        "accuracy": accuracy_score(y_true, y_pred),

        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),

        "macro_precision": precision_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),

        "mcc": matthews_corrcoef(y_true, y_pred),

        "cohen_kappa": cohen_kappa_score(
            y_true, y_pred
        ),

        "log_loss": log_loss(
            y_true,
            y_prob,
            labels=list(range(len(CLASS_NAMES)))
        )
    }

    # Multiclass ROC-AUC
    try:
        result["macro_roc_auc"] = roc_auc_score(
            y_true,
            y_prob,
            labels=list(range(len(CLASS_NAMES))),
            average="macro",
            multi_class="ovr"
        )

    except ValueError:
        result["macro_roc_auc"] = float("nan")

    return result


def make_classification_report(y_true, y_pred):

    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
