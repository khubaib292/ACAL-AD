
"""
ACAL-AD Dataset Inspection

Checks image integrity, dimensions, image modes,
class distribution and exact duplicate images.

Supports the original, unsplit dataset structure.
Original dataset files are never modified.
"""

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CLASSES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]

SOURCE_SPLIT = "original"

# --------------------------------------------------
# 2. SHA-256 HASH
# --------------------------------------------------

def calculate_hash(image_path):

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:

        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# 3. IMAGE VALIDATION
# --------------------------------------------------

def inspect_image(image_path):

    try:

        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:

            width, height = image.size
            mode = image.mode

        return {
            "valid": True,
            "width": width,
            "height": height,
            "mode": mode,
            "error": ""
        }

    except (OSError, ValueError, UnidentifiedImageError) as error:

        return {
            "valid": False,
            "width": "",
            "height": "",
            "mode": "",
            "error": str(error)
        }


# --------------------------------------------------
# 4. DATASET DISCOVERY
# --------------------------------------------------

def discover_images():

    image_paths = []

    print("\nDATASET CLASS DISTRIBUTION")
    print("=" * 55)

    for class_name in CLASSES:

        folder = DATA_ROOT / class_name

        if not folder.exists():

            print(f"WARNING: Missing class folder: {folder}")
            continue

        files = sorted(
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        )

        print(f"{class_name:18s} | {len(files):6d} images")

        image_paths.extend(files)

    return image_paths


# --------------------------------------------------
# 5. SAVE CSV REPORT
# --------------------------------------------------

def save_csv(file_path, fieldnames, records):

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)


# --------------------------------------------------
# 6. MAIN INSPECTION
# --------------------------------------------------

def main():

    print("\nACAL-AD DATASET INSPECTION")
    print("=" * 55)

    print("Dataset location:", DATA_ROOT)

    if not DATA_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_ROOT}"
        )

    image_paths = discover_images()

    print("\nTotal images discovered:", len(image_paths))

    if not image_paths:

        raise RuntimeError(
            "No images found. Check the dataset directory and class folders."
        )

    # --------------------------------------------------
    # 7. INSPECT IMAGES
    # --------------------------------------------------

    inventory = []

    class_counts = Counter()
    dimension_counts = Counter()
    mode_counts = Counter()

    hash_groups = defaultdict(list)

    invalid_images = []

    print("\nChecking images and calculating hashes...")

    for image_path in tqdm(image_paths):

        relative_path = image_path.relative_to(DATA_ROOT)

        class_name = relative_path.parts[0]

        result = inspect_image(image_path)

        file_hash = ""

        if result["valid"]:

            file_hash = calculate_hash(image_path)

            dimension_counts[
                f'{result["width"]}x{result["height"]}'
            ] += 1

            mode_counts[result["mode"]] += 1

            hash_groups[file_hash].append({
                "path": str(relative_path),
                "split": SOURCE_SPLIT,
                "class": class_name
            })

        else:

            invalid_images.append({
                "path": str(relative_path),
                "error": result["error"]
            })

        class_counts[class_name] += 1

        inventory.append({
            "path": str(relative_path),
            "split": SOURCE_SPLIT,
            "class": class_name,
            "valid": result["valid"],
            "width": result["width"],
            "height": result["height"],
            "mode": result["mode"],
            "size_bytes": image_path.stat().st_size,
            "sha256": file_hash,
            "error": result["error"]
        })

    # --------------------------------------------------
    # 8. SAVE IMAGE INVENTORY
    # --------------------------------------------------

    inventory_file = OUTPUT_DIR / "image_inventory.csv"

    inventory_fields = [
        "path",
        "split",
        "class",
        "valid",
        "width",
        "height",
        "mode",
        "size_bytes",
        "sha256",
        "error"
    ]

    save_csv(
        inventory_file,
        inventory_fields,
        inventory
    )

    # --------------------------------------------------
    # 9. EXACT DUPLICATE ANALYSIS
    # --------------------------------------------------

    duplicate_groups = []

    for file_hash, records in hash_groups.items():

        if len(records) > 1:

            duplicate_groups.append({
                "sha256": file_hash,
                "number_of_copies": len(records),
                "paths": records
            })

    duplicate_records = []

    for group in duplicate_groups:

        for record in group["paths"]:

            duplicate_records.append({
                "sha256": group["sha256"],
                "number_of_copies": group["number_of_copies"],
                "split": record["split"],
                "class": record["class"],
                "path": record["path"]
            })

    duplicate_file = OUTPUT_DIR / "exact_duplicates.csv"

    save_csv(
        duplicate_file,
        [
            "sha256",
            "number_of_copies",
            "split",
            "class",
            "path"
        ],
        duplicate_records
    )

    # --------------------------------------------------
    # 10. INVALID IMAGE REPORT
    # --------------------------------------------------

    invalid_file = OUTPUT_DIR / "invalid_images.csv"

    save_csv(
        invalid_file,
        ["path", "error"],
        invalid_images
    )

    # --------------------------------------------------
    # 11. SUMMARY REPORT
    # --------------------------------------------------

    summary_file = OUTPUT_DIR / "inspection_summary.txt"

    valid_count = len(inventory) - len(invalid_images)

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("ACAL-AD DATASET INSPECTION SUMMARY\n")
        file.write("=" * 45 + "\n\n")

        file.write(f"Dataset: {DATA_ROOT}\n")
        file.write(f"Total images: {len(image_paths)}\n")
        file.write(f"Valid images: {valid_count}\n")
        file.write(f"Invalid images: {len(invalid_images)}\n\n")

        file.write("CLASS COUNTS\n")

        for class_name in CLASSES:
            file.write(
                f"{class_name}: {class_counts[class_name]}\n"
            )

        file.write("\nIMAGE DIMENSIONS\n")

        for dimension, count in dimension_counts.most_common():
            file.write(f"{dimension}: {count}\n")

        file.write("\nIMAGE MODES\n")

        for mode, count in mode_counts.items():
            file.write(f"{mode}: {count}\n")

        file.write("\nEXACT DUPLICATE ANALYSIS\n")

        file.write(
            f"Duplicate groups: {len(duplicate_groups)}\n"
        )

        file.write("\nINVALID IMAGES\n")

        for record in invalid_images:
            file.write(
                f'{record["path"]}: {record["error"]}\n'
            )

    # --------------------------------------------------
    # 12. DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("INSPECTION COMPLETED")
    print("=" * 55)

    print("Total images:", len(image_paths))
    print("Valid images:", valid_count)
    print("Invalid images:", len(invalid_images))
    print("Exact duplicate groups:", len(duplicate_groups))

    print("\nClass distribution:")

    for class_name in CLASSES:
        print(f"{class_name}: {class_counts[class_name]}")

    print("\nImage dimensions:")

    for dimension, count in dimension_counts.most_common():
        print(f"{dimension}: {count}")

    print("\nImage modes:")

    for mode, count in mode_counts.items():
        print(f"{mode}: {count}")

    print("\nReports saved in:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("1. image_inventory.csv")
    print("2. exact_duplicates.csv")
    print("3. invalid_images.csv")
    print("4. inspection_summary.txt")


if __name__ == "__main__":
    main()
