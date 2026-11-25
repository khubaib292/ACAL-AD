
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloaders
from src.models.predatornet import PredatorNet
from src.checkpoint_utils import get_device, load_checkpoint

from src.evaluation.metrics import (
    CLASS_NAMES,
    calculate_metrics,
    make_classification_report
)
from src.evaluation.confusion_matrix import save_confusion_matrix
from src.evaluation.visualization import save_roc_curve


# Separate output directory to protect previous evaluation results.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation_reconstructed_80_10_10"
)

# Existing trained checkpoints.
EVALUATION_CHECKPOINTS = {
    "BGRA": PROJECT_ROOT / "models" / "PredatorNet_BGRA_Seed123.pt",
    "ACAL": PROJECT_ROOT / "models" / "PredatorNet_ACAL_Seed123.pt"
}


def print_heading(text):
    print("\n" + "=" * 72)
    print(text.center(72))
    print("=" * 72)


@torch.no_grad()
def predict(model, test_loader, device):
    model.eval()

    targets = []
    predictions = []
    probabilities = []

    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        logits = outputs["logits"]

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        if not torch.isfinite(probs).all():
            raise ValueError("Non-finite probability values detected.")

        row_sums = probs.sum(dim=1)

        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-5
        ):
            raise ValueError(
                "Predicted probabilities do not sum to 1."
            )

        targets.extend(labels.numpy().tolist())
        predictions.extend(preds.cpu().numpy().tolist())
        probabilities.extend(probs.cpu().numpy().tolist())

    return (
        np.asarray(targets),
        np.asarray(predictions),
        np.asarray(probabilities)
    )


def evaluate_one_checkpoint(name, checkpoint_path, test_loader, device):
    display_name = (
        "BGRA (Baseline)" if name == "BGRA"
        else "ACAL (Proposed)"
    )

    print_heading(f"Evaluating {display_name}")

    model = PredatorNet(
        num_classes=len(CLASS_NAMES),
        pretrained=False
    )

    model = load_checkpoint(
        model,
        checkpoint_path,
        device
    )

    y_true, y_pred, y_prob = predict(
        model,
        test_loader,
        device
    )

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    report = make_classification_report(y_true, y_pred)

    checkpoint_dir = OUTPUT_DIR / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation metrics.
    with open(
        checkpoint_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {key: float(value) for key, value in metrics.items()},
            file,
            indent=4
        )

    # Save class-wise classification report.
    with open(
        checkpoint_dir / "classification_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    # Save confusion matrix.
    save_confusion_matrix(
        y_true,
        y_pred,
        checkpoint_dir / "confusion_matrix"
    )

    # Save ROC curve.
    save_roc_curve(
        y_true,
        y_prob,
        checkpoint_dir / "roc_curve.png"
    )

    print("\nTEST RESULTS")
    print("-" * 50)

    for metric_name, value in metrics.items():
        print(f"{metric_name:22s}: {value:.4f}")

    print("\nPER-CLASS PERFORMANCE")
    print("-" * 50)

    for class_name in CLASS_NAMES:
        values = report[class_name]

        print(
            f"{class_name:20s} "
            f"Precision={values['precision']:.3f}  "
            f"Recall={values['recall']:.3f}  "
            f"F1={values['f1-score']:.3f}"
        )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def print_comparison(results):
    print_heading("BGRA VS ACAL COMPARISON")

    rows = []

    for name, metrics in results.items():
        display_name = (
            "ACAL (Proposed)" if name == "ACAL"
            else "BGRA (Baseline)"
        )

        row = {"method": display_name}
        row.update(metrics)
        rows.append(row)

    frame = pd.DataFrame(rows)

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "mcc",
        "cohen_kappa",
        "macro_roc_auc",
        "log_loss"
    ]

    print("\nINDIVIDUAL MODEL RESULTS")
    print("-" * 72)

    print(
        frame.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    comparison = frame.set_index("method")[metric_names]

    print("\nDIRECT COMPARISON")
    print("-" * 72)

    print(
        comparison.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # Save comparison results.
    frame.to_csv(
        OUTPUT_DIR / "checkpoint_metrics.csv",
        index=False
    )

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv"
    )

    print("\nINTERPRETATION")
    print("-" * 72)

    print(
        "BGRA is the baseline loss experiment. "
        "ACAL is the proposed adaptive confusion-aware loss experiment."
    )

    print(
        "Both models use the same PredatorNet architecture and reconstructed "
        "test subset. Independence from the original training data "
        "has not been verified."
    )

    print(
        "Higher accuracy, macro F1, balanced accuracy and ROC-AUC "
        "are generally better. Lower log loss is better."
    )

    difference = (
        results["ACAL"]["macro_f1"]
        - results["BGRA"]["macro_f1"]
    )

    print(f"\nMacro-F1 difference (ACAL - BGRA): {difference:+.4f}")

    if difference > 0:
        print("ACAL has a higher macro-F1 in this comparison.")
    elif difference < 0:
        print("BGRA has a higher macro-F1 in this comparison.")
    else:
        print("Both methods have the same macro-F1.")


def main():
    print_heading("ACAL-AD RECONSTRUCTED-SPLIT EVALUATION")

    print(
        "WARNING: Exploratory evaluation using a reconstructed 80:10:10 "
        "split. The original checkpoint training split is unverified. "
        "Independent test-set performance cannot be guaranteed."
    )

    device = get_device()

    print("\nDevice:", device)
    print("Project:", PROJECT_ROOT)
    print("Models: BGRA Seed123 and ACAL Seed123")
    print("Training: Disabled")
    print("Fine-tuning: Disabled")

    manifest = (
        PROJECT_ROOT
        / "outputs"
        / "dataset_split"
        / "original_only_manifest_80_10_10_reconstructed.csv"
    )

    if not manifest.is_file():
        raise FileNotFoundError(
            f"Reconstructed manifest not found: {manifest}"
        )

    for name, checkpoint_path in EVALUATION_CHECKPOINTS.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"{name} checkpoint not found: {checkpoint_path}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading reconstructed test dataset...")

    _, _, test_loader = create_dataloaders(
        manifest_path=manifest,
        batch_size=16,
        num_workers=0,
        normalization="kaggle"
    )

    print("Test images:", len(test_loader.dataset))

    results = {}

    for name, checkpoint_path in EVALUATION_CHECKPOINTS.items():
        results[name] = evaluate_one_checkpoint(
            name,
            checkpoint_path,
            test_loader,
            device
        )

    print_comparison(results)

    print_heading("EVALUATION COMPLETED")

    print("Results saved in:")
    print(OUTPUT_DIR)

    print("\nNo model weights were modified.")
    print("No training or fine-tuning was performed.")


if __name__ == "__main__":
    main()
