
"""
ACAL-AD: Dataset Splitting

Creates reproducible, stratified train/validation/test splits.
"""

from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split


def stratified_split(
    samples,
    labels,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
):
    """
    Split samples while preserving class proportions.

    Args:
        samples: List of image paths or sample identifiers.
        labels: Corresponding integer class labels.
        train_ratio: Training proportion.
        validation_ratio: Validation proportion.
        test_ratio: Testing proportion.
        random_seed: Reproducibility seed.

    Returns:
        Dictionary containing train, validation and test samples.
    """

    if len(samples) != len(labels):
        raise ValueError(
            "samples and labels must have equal lengths."
        )

    if len(samples) == 0:
        raise ValueError("Cannot split an empty dataset.")

    ratios = [train_ratio, validation_ratio, test_ratio]

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be positive.")

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    samples = np.asarray(samples)
    labels = np.asarray(labels)

    train_samples, remaining_samples, train_labels, remaining_labels = (
        train_test_split(
            samples,
            labels,
            test_size=(validation_ratio + test_ratio),
            stratify=labels,
            random_state=random_seed,
        )
    )

    relative_test_ratio = (
        test_ratio / (validation_ratio + test_ratio)
    )

    validation_samples, test_samples, validation_labels, test_labels = (
        train_test_split(
            remaining_samples,
            remaining_labels,
            test_size=relative_test_ratio,
            stratify=remaining_labels,
            random_state=random_seed,
        )
    )

    return {
        "train": {
            "samples": train_samples.tolist(),
            "labels": train_labels.tolist(),
        },
        "validation": {
            "samples": validation_samples.tolist(),
            "labels": validation_labels.tolist(),
        },
        "test": {
            "samples": test_samples.tolist(),
            "labels": test_labels.tolist(),
        },
    }


def save_splits(split_data, output_dir):
    """Save split assignments as CSV files."""

    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split in split_data.items():
        output_file = output_dir / f"{split_name}.csv"

        with output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["image_path", "label"])

            writer.writerows(
                zip(split["samples"], split["labels"])
            )

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    print("ACAL-AD splitting module loaded successfully.")
    print("Use stratified_split() after dataset auditing.")
