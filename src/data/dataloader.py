
# import torch

# from torch.utils.data import DataLoader

# from src.data.dataset import AlzheimerDataset


# DEFAULT_MANIFEST = (
#     "outputs/dataset_split/"
#     "original_only_manifest_80_10_10.csv"
# )


# def create_dataloaders(
#     manifest_path=DEFAULT_MANIFEST,
#     batch_size=16,
#     num_workers=0,
#     normalization="kaggle"
# ):

#     train_dataset = AlzheimerDataset(
#         manifest_path,
#         split="train",
#         normalization=normalization
#     )

#     val_dataset = AlzheimerDataset(
#         manifest_path,
#         split="val",
#         normalization=normalization
#     )

#     test_dataset = AlzheimerDataset(
#         manifest_path,
#         split="test",
#         normalization=normalization
#     )

#     pin_memory = torch.cuda.is_available()

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=pin_memory
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=pin_memory
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=pin_memory
#     )

#     return train_loader, val_loader, test_loader


# if __name__ == "__main__":

#     train_loader, val_loader, test_loader = (
#         create_dataloaders()
#     )

#     print("\nACAL-AD DATALOADER VALIDATION")
#     print("=============================")

#     images, labels = next(iter(train_loader))

#     print("Training images:", len(train_loader.dataset))
#     print("Validation images:", len(val_loader.dataset))
#     print("Testing images:", len(test_loader.dataset))

#     print("Training batches:", len(train_loader))
#     print("Validation batches:", len(val_loader))
#     print("Testing batches:", len(test_loader))

#     print("Image batch shape:", images.shape)
#     print("Label batch shape:", labels.shape)

#     print("Image dtype:", images.dtype)
#     print("Label dtype:", labels.dtype)

#     print("First batch labels:", labels.tolist())

#     print("\nDataLoader verification completed.")



from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import AlzheimerDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_split"
    / "original_only_manifest_80_10_10.csv"
)


def create_dataloaders(
    manifest_path=DEFAULT_MANIFEST,
    batch_size=16,
    num_workers=0,
    normalization="kaggle"
):
    train_dataset = AlzheimerDataset(
        manifest_path,
        split="train",
        normalization=normalization
    )

    val_dataset = AlzheimerDataset(
        manifest_path,
        split="val",
        normalization=normalization
    )

    test_dataset = AlzheimerDataset(
        manifest_path,
        split="test",
        normalization=normalization
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders()

    print("\nACAL-AD DATALOADER CHECK")
    print("-----------------------")
    print("Train images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("Labels:", labels.tolist())
