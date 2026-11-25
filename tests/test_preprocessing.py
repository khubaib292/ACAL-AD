
import numpy as np
import pytest

from PIL import Image

from src.preprocessing import (
    ApplyCLAHE,
    get_preprocessing_pipeline,
    preprocess_image,
)


def test_preprocessing_output_shape():
    image = Image.new("RGB", (300, 300), color="white")

    pipeline = get_preprocessing_pipeline(image_size=224)
    result = pipeline(image)

    assert tuple(result.shape) == (3, 224, 224)


def test_preprocessing_output_type():
    image = Image.new("RGB", (224, 224), color="white")

    pipeline = get_preprocessing_pipeline()
    result = pipeline(image)

    assert result.dtype.is_floating_point


def test_training_augmentation_pipeline():
    image = Image.new("RGB", (224, 224), color="white")

    pipeline = get_preprocessing_pipeline(training=True)
    result = pipeline(image)

    assert tuple(result.shape) == (3, 224, 224)


def test_clahe_output():
    image = Image.new("RGB", (224, 224), color="gray")

    clahe = ApplyCLAHE()
    result = clahe(image)

    assert isinstance(result, Image.Image)
    assert result.size == (224, 224)


def test_clahe_pipeline():
    image = Image.new("RGB", (224, 224), color="gray")

    pipeline = get_preprocessing_pipeline(use_clahe=True)
    result = pipeline(image)

    assert tuple(result.shape) == (3, 224, 224)


def test_invalid_image_size():
    with pytest.raises(ValueError):
        get_preprocessing_pipeline(image_size=0)


def test_preprocess_image_from_file(tmp_path):
    image_path = tmp_path / "sample.png"

    image = Image.new("RGB", (300, 300), color="white")
    image.save(image_path)

    result = preprocess_image(image_path)

    assert tuple(result.shape) == (3, 224, 224)
