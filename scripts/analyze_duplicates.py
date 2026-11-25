
"""
ACAL-AD Exact Duplicate Analysis

Analyzes duplicate groups found during dataset inspection.
Does not delete or modify any images.
"""

import csv
from collections import defaultdict, Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DUPLICATE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_inspection"
    / "exact_duplicates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_inspection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    if not DUPLICATE_FILE.exists():
        raise FileNotFoundError(DUPLICATE_FILE)

    groups = defaultdict(list)

    with open(
        DUPLICATE_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            groups[row["sha256"]].append(row)

    print("\nEXACT DUPLICATE ANALYSIS")
    print("=" * 55)

    print("Duplicate groups:", len(groups))

    total_copies = sum(len(records) for records in groups.values())

    extra_copies = sum(
        len(records) - 1
        for records in groups.values()
    )

    print("Images involved in duplicate groups:", total_copies)
    print("Extra copies beyond one per group:", extra_copies)

    class_group_counts = Counter()
    split_group_counts = Counter()

    cross_class_groups = []
    within_class_groups = []
    cross_split_groups = []

    detailed_rows = []

    for group_id, records in enumerate(groups.values(), start=1):

        classes = {row["class"] for row in records}
        splits = {row["split"] for row in records}

        if len(classes) > 1:
            cross_class_groups.append(records)
        else:
            within_class_groups.append(records)

        if len(splits) > 1:
            cross_split_groups.append(records)

        for class_name in classes:
            class_group_counts[class_name] += 1

        for split in splits:
            split_group_counts[split] += 1

        for row in records:
            detailed_rows.append({
                "group_id": group_id,
                "class": row["class"],
                "split": row["split"],
                "path": row["path"],
                "number_of_copies": len(records),
                "cross_class": len(classes) > 1,
                "cross_split": len(splits) > 1
            })

    print("\nWithin-class duplicate groups:", len(within_class_groups))
    print("Cross-class duplicate groups:", len(cross_class_groups))
    print("Cross-split duplicate groups:", len(cross_split_groups))

    print("\nDuplicate group counts by class:")

    for class_name, count in sorted(class_group_counts.items()):
        print(f"{class_name}: {count}")

    print("\nDuplicate group counts by split:")

    for split, count in sorted(split_group_counts.items()):
        print(f"{split}: {count}")

    report_path = OUTPUT_DIR / "duplicate_analysis.csv"

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "group_id",
                "class",
                "split",
                "path",
                "number_of_copies",
                "cross_class",
                "cross_split"
            ]
        )

        writer.writeheader()
        writer.writerows(detailed_rows)

    print("\nDetailed report saved to:")
    print(report_path)


if __name__ == "__main__":
    main()
