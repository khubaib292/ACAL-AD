
"""
ACAL-AD: Reproducible Dataset Split Preparation

Creates stratified train, validation and test manifests
using the original-only Alzheimer MRI dataset.

Rules:
1. Original images are never modified.
2. Exact duplicate images remain in the same split.
3. Images are stratified by their four classes.
4. A fixed random seed ensures reproducibility.
5. Near-duplicate candidates are not automatically removed.
"""

import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INSPECTION_DIR = PROJECT_ROOT / "outputs" / "dataset_inspection"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_split"

INVENTORY_FILE = INSPECTION_DIR / "image_inventory.csv"

SEED = 42

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

CLASSES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]


def load_inventory():
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Dataset inventory not found: {INVENTORY_FILE}"
        )

    with open(INVENTORY_FILE, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def create_groups(records):
    groups = defaultdict(list)

    for record in records:
        if record["valid"].lower() != "true":
            continue

        class_name = record["class"]

        if class_name not in CLASSES:
            continue

        group_id = record["sha256"] or record["path"]
        groups[(class_name, group_id)].append(record)

    return groups


def calculate_targets(total):
    train = round(total * 0.70)
    val = round(total * 0.15)
    test = total - train - val

    return {
        "train": train,
        "val": val,
        "test": test
    }


def allocate_groups(class_groups):
    rng = random.Random(SEED)
    assignments = []

    for class_name in CLASSES:
        groups = [
            records
            for (label, _), records in class_groups.items()
            if label == class_name
        ]

        rng.shuffle(groups)

        total_images = sum(len(group) for group in groups)
        targets = calculate_targets(total_images)

        counts = Counter()

        # Assign the largest duplicate groups first.
        groups.sort(key=len, reverse=True)

        for group in groups:
            # Select the split with the greatest remaining capacity.
            split = max(
                SPLIT_RATIOS,
                key=lambda name: targets[name] - counts[name]
            )

            for record in group:
                assignments.append({
                    "path": record["path"],
                    "class": class_name,
                    "split": split,
                    "sha256": record["sha256"],
                    "original_split": record["split"],
                    "duplicate_group_size": len(group)
                })

            counts[split] += len(group)

        print(f"\n{class_name}")
        print("Total:", total_images)

        for split in SPLIT_RATIOS:
            print(f"{split}: {counts[split]} (target: {targets[split]})")

    return assignments


def save_manifest(assignments):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_file = OUTPUT_DIR / "dataset_manifest.csv"

    fieldnames = [
        "path",
        "class",
        "split",
        "sha256",
        "original_split",
        "duplicate_group_size"
    ]

    with open(manifest_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assignments)

    # Also create the filename expected by evaluation scripts.
    original_manifest = (
        OUTPUT_DIR / "original_only_manifest_80_10_10.csv"
    )

    # This file is a compatibility copy of the 70:15:15 manifest.
    # The filename is retained only because existing scripts expect it.
    with open(original_manifest, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assignments)

    return manifest_file


def validate_manifest(assignments):
    print("\nFINAL SPLIT SUMMARY")
    print("=" * 55)

    counts = Counter(
        (row["split"], row["class"])
        for row in assignments
    )

    for split in SPLIT_RATIOS:
        print(f"\n{split.upper()}")

        total = 0

        for class_name in CLASSES:
            count = counts[(split, class_name)]
            print(f"{class_name}: {count}")
            total += count

        print("Total:", total)

    hash_splits = defaultdict(set)

    for row in assignments:
        if row["sha256"]:
            hash_splits[row["sha256"]].add(row["split"])

    leakage_groups = [
        key for key, splits in hash_splits.items()
        if len(splits) > 1
    ]

    print(
        "\nEXACT DUPLICATE GROUPS ACROSS SPLITS:",
        len(leakage_groups)
    )

    if leakage_groups:
        raise RuntimeError("Exact duplicate leakage detected.")

    total = len(assignments)

    print("\nTotal valid images assigned:", total)

    if total != 6400:
        raise RuntimeError(
            f"Expected 6400 original images, but assigned {total}."
        )

    print("Dataset split validation passed.")


def main():
    print("ACAL-AD DATASET SPLIT PREPARATION")
    print("=" * 55)

    records = load_inventory()

    print("Inventory records:", len(records))

    class_groups = create_groups(records)

    print("Duplicate-aware groups:", len(class_groups))

    assignments = allocate_groups(class_groups)

    validate_manifest(assignments)

    manifest_file = save_manifest(assignments)

    print("\nManifest saved to:")
    print(manifest_file)

    print("\nOriginal dataset remains unchanged.")


if __name__ == "__main__":
    main()
