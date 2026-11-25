
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from src.evaluation.metrics import CLASS_NAMES


def save_confusion_matrix(y_true, y_pred, output_path):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES)))
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the numerical matrix.
    pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    ).to_csv(output_path.with_suffix(".csv"))

    # Save its visual representation.
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix)

    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Confusion Matrix"
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor"
    )

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    plt.close(fig)

    return matrix
