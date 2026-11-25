
"""
ACAL-AD: Visual review of near-duplicate candidates.

Selects a reproducible sample of candidate pairs and
creates contact sheets for manual inspection.

Does not modify or delete original images.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

INPUT_FILE = (
    PROJECT_ROOT / "outputs" / "dataset_inspection"
    / "near_duplicate_candidates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "dataset_inspection"
    / "visual_review"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLES_PER_GROUP = 3
PAIRS_PER_PAGE = 12

random.seed(SEED)


def load_candidates():

    with open(INPUT_FILE, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def create_contact_sheet(records, page_number):

    tile_width = 440
    image_height = 230
    label_height = 65
    pair_height = image_height + label_height
    page_height = PAIRS_PER_PAGE * pair_height

    sheet = Image.new(
        "RGB",
        (tile_width * 2, page_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    start = page_number * PAIRS_PER_PAGE
    page_records = records[start:start + PAIRS_PER_PAGE]

    for index, pair in enumerate(page_records):

        y = index * pair_height

        for side, key in enumerate(["path_1", "path_2"]):

            image_path = DATA_ROOT / pair[key]

            with Image.open(image_path) as image:

                image = image.convert("RGB")
                image.thumbnail((tile_width - 10, image_height - 10))

                x = side * tile_width + (
                    tile_width - image.width
                ) // 2

                sheet.paste(image, (x, y + 5))

        label = (
            f'{pair["class"]} | '
            f'distance={pair["phash_distance"]} | '
            f'{pair["split_1"]} vs {pair["split_2"]}'
        )

        draw.text(
            (5, y + image_height + 5),
            label,
            fill="black"
        )

        draw.line(
            (0, y + pair_height - 1,
             tile_width * 2, y + pair_height - 1),
            fill="gray",
            width=1
        )

    output_file = OUTPUT_DIR / f"review_page_{page_number + 1}.jpg"

    sheet.save(output_file, quality=90)

    return output_file


def main():

    candidates = load_candidates()

    groups = defaultdict(list)

    for pair in candidates:

        key = (
            pair["class"],
            pair["phash_distance"],
            pair["split_1"],
            pair["split_2"]
        )

        groups[key].append(pair)

    selected = []

    for key in sorted(groups):

        records = groups[key]

        sample_count = min(
            SAMPLES_PER_GROUP,
            len(records)
        )

        selected.extend(
            random.sample(records, sample_count)
        )

    print("Total candidate pairs:", len(candidates))
    print("Candidate categories:", len(groups))
    print("Selected review pairs:", len(selected))

    with open(
        OUTPUT_DIR / "selected_review_pairs.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=selected[0].keys()
        )

        writer.writeheader()
        writer.writerows(selected)

    page_count = (
        len(selected) + PAIRS_PER_PAGE - 1
    ) // PAIRS_PER_PAGE

    print("\nCreating contact sheets...")

    for page in tqdm(range(page_count)):

        output = create_contact_sheet(selected, page)

        print("Created:", output)

    print("\nVisual review files saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
