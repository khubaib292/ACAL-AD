
# """
# ACAL-AD Image Preprocessing

# Supports both ImageNet and Kaggle PredatorNet normalization.
# Original image files are never modified.
# """

# from pathlib import Path

# import numpy as np
# import torch

# from PIL import Image
# from torchvision import transforms


# IMAGE_SIZE = 224

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)

# KAGGLE_MEAN = (0.5, 0.5, 0.5)
# KAGGLE_STD = (0.5, 0.5, 0.5)


# def load_image(image_path):
#     """Load an MRI image and convert it to RGB."""

#     image_path = Path(image_path)

#     if not image_path.exists():
#         raise FileNotFoundError(image_path)

#     with Image.open(image_path) as image:
#         return image.convert("RGB")


# def build_preprocessing_transform(normalization="imagenet"):
#     """Build the deterministic preprocessing pipeline."""

#     if normalization == "imagenet":
#         mean = IMAGENET_MEAN
#         std = IMAGENET_STD

#     elif normalization == "kaggle":
#         mean = KAGGLE_MEAN
#         std = KAGGLE_STD

#     else:
#         raise ValueError(
#             f"Unsupported normalization: {normalization}"
#         )

#     return transforms.Compose([
#         transforms.Resize(
#             (IMAGE_SIZE, IMAGE_SIZE),
#             interpolation=transforms.InterpolationMode.BILINEAR
#         ),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)
#     ])


# def preprocess_image(image_path, normalization="imagenet"):
#     """Convert one MRI image into a normalized tensor."""

#     image = load_image(image_path)
#     transform = build_preprocessing_transform(normalization)

#     return transform(image)


# def validate_preprocessed_tensor(tensor):
#     """Validate output shape, data type and finite values."""

#     if not isinstance(tensor, torch.Tensor):
#         raise TypeError("Expected a PyTorch tensor.")

#     if tensor.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
#         raise ValueError(
#             f"Unexpected image shape: {tensor.shape}"
#         )

#     if not torch.isfinite(tensor).all():
#         raise ValueError("Tensor contains NaN or infinite values.")

#     return True


# def preprocess_numpy(image_path, normalization="imagenet"):
#     """Return a normalized NumPy array for inspection."""

#     tensor = preprocess_image(image_path, normalization)

#     validate_preprocessed_tensor(tensor)

#     return tensor.permute(1, 2, 0).numpy()


# if __name__ == "__main__":

#     print("ACAL-AD PREPROCESSING MODULE")
#     print("Image size:", IMAGE_SIZE)
#     print("Color mode: RGB")
#     print("Available normalization: ImageNet, Kaggle")


from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

KAGGLE_MEAN = (0.5, 0.5, 0.5)
KAGGLE_STD = (0.5, 0.5, 0.5)


def load_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    with Image.open(image_path) as image:
        return image.convert("RGB")


def build_preprocessing_transform(normalization="kaggle"):
    if normalization == "imagenet":
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

    elif normalization == "kaggle":
        mean = KAGGLE_MEAN
        std = KAGGLE_STD

    else:
        raise ValueError(
            f"Unsupported normalization: {normalization}"
        )

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def preprocess_image(image_path, normalization="kaggle"):
    image = load_image(image_path)
    transform = build_preprocessing_transform(normalization)
    tensor = transform(image)

    validate_preprocessed_tensor(tensor)
    return tensor


def validate_preprocessed_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected a PyTorch tensor.")

    if tensor.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    if not torch.isfinite(tensor).all():
        raise ValueError("Tensor contains NaN or infinite values.")

    return True
