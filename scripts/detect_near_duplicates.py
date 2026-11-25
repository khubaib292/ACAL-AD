
"""
ACAL-AD Near-Duplicate Detection

Uses perceptual hashing to identify visually
similar MRI images.

This is an initial candidate-detection stage.
It does not delete or modify original images.
"""

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val"]

CLASSES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]

# Lower threshold = stricter similarity.
# We will review the results before choosing a final threshold.
PHASH_THRESHOLD = 4


def get_images():

    images = []

    for split in SPLITS:

        for class_name in CLASSES:

            folder = DATA_ROOT / split / class_name

            if not folder.exists():
                continue

            for path in folder.glob("*.jpg"):

                images.append({
                    "path": path,
                    "split": split,
                    "class": class_name
                })

    return images


def main():

    images = get_images()

    print("Total images:", len(images))
    print("Calculating perceptual hashes...")

    hash_groups = defaultdict(list)

    for item in tqdm(images):

        try:

            with Image.open(item["path"]) as image:

                phash = imagehash.phash(image.convert("RGB"))

            item["hash"] = phash
            hash_groups[item["class"]].append(item)

        except Exception as error:

            print("Could not process:", item["path"], error)

    print("\nComparing images within each class...")

    candidate_pairs = []

    total_comparisons = 0

    for class_name, records in hash_groups.items():

        for first, second in tqdm(
            combinations(records, 2),
            desc=class_name
        ):

            total_comparisons += 1

            distance = first["hash"] - second["hash"]

            if distance <= PHASH_THRESHOLD:

                candidate_pairs.append({
                    "class": class_name,
                    "split_1": first["split"],
                    "split_2": second["split"],
                    "path_1": str(
                        first["path"].relative_to(DATA_ROOT)
                    ),
                    "path_2": str(
                        second["path"].relative_to(DATA_ROOT)
                    ),
                    "phash_distance": distance
                })

    output_file = OUTPUT_DIR / "near_duplicate_candidates.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class",
                "split_1",
                "split_2",
                "path_1",
                "path_2",
                "phash_distance"
            ]
        )

        writer.writeheader()
        writer.writerows(candidate_pairs)

    cross_split = [
        pair for pair in candidate_pairs
        if pair["split_1"] != pair["split_2"]
    ]

    print("\n" + "=" * 55)
    print("NEAR-DUPLICATE DETECTION COMPLETED")
    print("=" * 55)

    print("Images processed:", len(images))
    print("Pair comparisons:", total_comparisons)
    print("Candidate pairs:", len(candidate_pairs))
    print("Cross-split candidate pairs:", len(cross_split))
    print("Threshold:", PHASH_THRESHOLD)

    print("\nReport saved at:")
    print(output_file)


if __name__ == "__main__":
    main()
