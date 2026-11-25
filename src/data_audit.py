
"""
ACAL-AD: Alzheimer's MRI Dataset Audit

Checks:
- Dataset directory and expected class folders
- Supported image files
- Corrupted or unreadable images
- Exact duplicate images using SHA-256
- Image dimensions and formats

This script does not modify or delete dataset files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


EXPECTED_CLASSES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def identify_class(
    file_path: Path,
    expected_classes: list[str],
) -> str | None:
    """Identify a class from the image's parent directory structure."""

    expected_lookup = {
        class_name.lower(): class_name
        for class_name in expected_classes
    }

    matched_classes = [
        expected_lookup[part.lower()]
        for part in file_path.parts[:-1]
        if part.lower() in expected_lookup
    ]

    if not matched_classes:
        return None

    # Use the nearest matching class directory.
    return matched_classes[-1]


def inspect_image(file_path: Path) -> dict[str, Any]:
    """Check whether an image is readable and collect its properties."""

    result = {
        "status": "corrupted",
        "width": None,
        "height": None,
        "format": None,
        "error": None,
    }

    try:
        with Image.open(file_path) as image:
            image.verify()

        # Reopen after verify to read dimensions.
        with Image.open(file_path) as image:
            result["width"] = image.width
            result["height"] = image.height
            result["format"] = image.format
            result["status"] = "valid"

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
    ) as error:
        result["error"] = str(error)

    return result


def audit_dataset(
    dataset_root: str | Path,
    expected_classes: list[str] | None = None,
    output_dir: str | Path = "results/data_audit",
) -> dict[str, Any]:
    """
    Audit an image classification dataset.

    Expected structure example:

        dataset/
            MildDemented/
            ModerateDemented/
            NonDemented/
            VeryMildDemented/

    Nested structures are also supported when class folder names
    appear somewhere in the image's parent directory path.
    """

    root = Path(dataset_root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    classes = expected_classes or EXPECTED_CLASSES

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory: {root}"
        )

    print("\n" + "=" * 65)
    print("ACAL-AD DATASET AUDIT")
    print("=" * 65)
    print(f"Dataset location: {root}")
    print("Scanning image files...\n")

    inventory: list[dict[str, Any]] = []
    class_counts = Counter()
    dimension_counts = Counter()
    format_counts = Counter()
    hash_to_files: dict[str, list[str]] = defaultdict(list)

    total_supported = 0
    valid_images = 0
    corrupted_images = 0
    unassigned_images = 0

    image_files = sorted(
        file_path
        for file_path in root.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for index, file_path in enumerate(image_files, start=1):
        total_supported += 1

        class_name = identify_class(file_path, classes)
        image_info = inspect_image(file_path)

        relative_path = str(file_path.relative_to(root))

        try:
            file_hash = calculate_sha256(file_path)
        except OSError:
            file_hash = None

        if class_name is None:
            unassigned_images += 1
        else:
            class_counts[class_name] += 1

        if image_info["status"] == "valid":
            valid_images += 1

            dimension_counts[
                f'{image_info["width"]}x{image_info["height"]}'
            ] += 1

            if image_info["format"]:
                format_counts[image_info["format"]] += 1

        else:
            corrupted_images += 1

        if file_hash:
            hash_to_files[file_hash].append(relative_path)

        inventory.append({
            "file_name": file_path.name,
            "relative_path": relative_path,
            "class_name": class_name or "UNASSIGNED",
            "status": image_info["status"],
            "width": image_info["width"],
            "height": image_info["height"],
            "format": image_info["format"],
            "sha256": file_hash,
            "error": image_info["error"],
        })

        if index % 500 == 0:
            print(f"Processed {index}/{len(image_files)} images")

    duplicate_groups = [
        {
            "sha256": file_hash,
            "files": files,
            "count": len(files),
        }
        for file_hash, files in hash_to_files.items()
        if len(files) > 1
    ]

    duplicate_files = sum(
        group["count"] - 1
        for group in duplicate_groups
    )

    missing_classes = [
        class_name
        for class_name in classes
        if class_counts[class_name] == 0
    ]

    summary = {
        "project": "ACAL-AD",
        "dataset_root": str(root),
        "total_supported_images": total_supported,
        "valid_images": valid_images,
        "corrupted_images": corrupted_images,
        "unassigned_images": unassigned_images,
        "exact_duplicate_groups": len(duplicate_groups),
        "additional_duplicate_files": duplicate_files,
        "class_counts": {
            class_name: class_counts[class_name]
            for class_name in classes
        },
        "missing_classes": missing_classes,
        "image_dimensions": dict(dimension_counts),
        "image_formats": dict(format_counts),
        "duplicate_groups": duplicate_groups,
    }

    # Save audit results.
    output_path.mkdir(parents=True, exist_ok=True)

    summary_file = output_path / "audit_summary.json"
    inventory_file = output_path / "image_inventory.csv"

    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    if inventory:
        with inventory_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=inventory[0].keys(),
            )
            writer.writeheader()
            writer.writerows(inventory)

    else:
        with inventory_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow([
                "file_name",
                "relative_path",
                "class_name",
                "status",
                "width",
                "height",
                "format",
                "sha256",
                "error",
            ])

    # Print results.
    print("\n" + "=" * 65)
    print("DATASET AUDIT SUMMARY")
    print("=" * 65)

    print(f"Total supported images : {total_supported}")
    print(f"Valid images           : {valid_images}")
    print(f"Corrupted images       : {corrupted_images}")
    print(f"Unassigned images      : {unassigned_images}")

    print("\nCLASS DISTRIBUTION")
    for class_name in classes:
        print(f"{class_name:22s}: {class_counts[class_name]}")

    print("\nDUPLICATE ANALYSIS")
    print(f"Exact duplicate groups : {len(duplicate_groups)}")
    print(f"Additional duplicate files: {duplicate_files}")

    print("\nIMAGE FORMATS")
    for image_format, count in format_counts.items():
        print(f"{image_format}: {count}")

    print("\nIMAGE DIMENSIONS")
    for dimensions, count in dimension_counts.most_common():
        print(f"{dimensions}: {count}")

    print("\nMISSING CLASSES")
    print(missing_classes if missing_classes else "None")

    print("\nSAVED REPORTS")
    print(f"Summary   : {summary_file}")
    print(f"Inventory : {inventory_file}")

    print("\nDataset audit completed.")
    print("=" * 65)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the ACAL-AD Alzheimer's MRI dataset."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the dataset directory.",
    )

    parser.add_argument(
        "--output",
        default="results/data_audit",
        help="Directory for audit reports.",
    )

    args = parser.parse_args()

    audit_dataset(
        dataset_root=args.dataset,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
