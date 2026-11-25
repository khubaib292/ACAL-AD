
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import preprocess_image


CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

CLASS_TO_INDEX = {
    name: index for index, name in enumerate(CLASS_NAMES)
}


class AlzheimerDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        split,
        normalization="kaggle"
    ):
        self.manifest_path = Path(manifest_path).resolve()

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        # Manifest location: project_root/outputs/dataset_split/
        self.project_root = self.manifest_path.parents[2]
        self.data_root = self.project_root / "data" / "raw"

        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_root}"
            )

        self.data = pd.read_csv(self.manifest_path)

        required_columns = {"path", "class", "split"}
        missing = required_columns - set(self.data.columns)

        if missing:
            raise ValueError(
                f"Manifest missing columns: {sorted(missing)}"
            )

        requested_split = "validation" if split == "val" else split
        self.data["split"] = (
            self.data["split"].astype(str).str.lower()
        )

        self.data = self.data[
            self.data["split"] == requested_split
        ].reset_index(drop=True)

        if self.data.empty:
            raise ValueError(
                f"No records found for split: {requested_split}"
            )

        unknown = set(self.data["class"]) - set(CLASS_NAMES)

        if unknown:
            raise ValueError(
                f"Unknown classes in manifest: {unknown}"
            )

        self.normalization = normalization

        print(
            f"{requested_split.upper()} dataset loaded: "
            f"{len(self.data)} images"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        relative_path = Path(
            str(row["path"]).replace("\\", "/")
        )

        # The manifest paths are relative to data/raw.
        image_path = (self.data_root / relative_path).resolve()

        # Prevent paths from escaping the dataset directory.
        if not image_path.is_relative_to(self.data_root.resolve()):
            raise ValueError(
                f"Invalid image path in manifest: {row['path']}"
            )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        expected_class = str(row["class"])

        if image_path.parent.name != expected_class:
            raise ValueError(
                f"Class mismatch for image: {image_path}"
            )

        image = preprocess_image(
            image_path,
            normalization=self.normalization
        )

        label = CLASS_TO_INDEX[expected_class]

        return image, torch.tensor(
            label,
            dtype=torch.long
        )
