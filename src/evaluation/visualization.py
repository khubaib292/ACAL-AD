
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.evaluation.metrics import CLASS_NAMES


def save_roc_curve(y_true, y_prob, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_binary = label_binarize(
        y_true,
        classes=list(range(len(CLASS_NAMES)))
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    for index, class_name in enumerate(CLASS_NAMES):
        # A class needs both positive and negative samples.
        if len(set(y_binary[:, index])) < 2:
            continue

        fpr, tpr, _ = roc_curve(
            y_binary[:, index],
            y_prob[:, index]
        )

        class_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC = {class_auc:.3f})"
        )

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Multi-class ROC Curve"
    )

    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
