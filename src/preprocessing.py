
"""
ACAL-AD: MRI Image Preprocessing

Provides resizing, optional CLAHE enhancement,
training augmentation, and ImageNet normalization.
"""

import cv2
import numpy as np

from PIL import Image
from torchvision import transforms


class ApplyCLAHE:
    """Apply CLAHE to the luminance channel of an MRI image."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        image_array = np.array(image.convert("RGB"))

        lab_image = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2LAB,
        )

        luminance, channel_a, channel_b = cv2.split(lab_image)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

        enhanced_luminance = clahe.apply(luminance)

        enhanced_lab = cv2.merge(
            [enhanced_luminance, channel_a, channel_b]
        )

        enhanced_rgb = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2RGB,
        )

        return Image.fromarray(enhanced_rgb)


def get_preprocessing_pipeline(
    image_size=224,
    training=False,
    use_clahe=False,
):
    """
    Build the MRI preprocessing pipeline.

    Args:
        image_size: Target image dimensions.
        training: Whether training augmentation is enabled.
        use_clahe: Whether CLAHE enhancement is enabled.

    Returns:
        torchvision.transforms.Compose
    """

    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    operations = [
        transforms.Resize((image_size, image_size)),
    ]

    if use_clahe:
        operations.append(ApplyCLAHE())

    if training:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(operations)


def preprocess_image(
    image_path,
    image_size=224,
    use_clahe=False,
):
    """Preprocess one MRI image for inference."""

    image = Image.open(image_path).convert("RGB")

    pipeline = get_preprocessing_pipeline(
        image_size=image_size,
        training=False,
        use_clahe=use_clahe,
    )

    return pipeline(image)
