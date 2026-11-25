
from pathlib import Path

from PIL import Image

from src.data_audit import (
    EXPECTED_CLASSES,
    audit_dataset,
    calculate_sha256,
    identify_class,
    inspect_image,
)


def create_test_dataset(root: Path):
    """Create a small temporary MRI-style dataset."""

    for class_name in EXPECTED_CLASSES:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)

        image = Image.new("RGB", (224, 224), color="white")
        image.save(class_dir / "sample.png")

    return root


def test_sha256_calculation(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("ACAL-AD")

    first_hash = calculate_sha256(file_path)
    second_hash = calculate_sha256(file_path)

    assert first_hash == second_hash


def test_class_identification(tmp_path):
    image_path = (
        tmp_path
        / "MildDemented"
        / "sample.png"
    )

    assert identify_class(
        image_path,
        EXPECTED_CLASSES,
    ) == "MildDemented"


def test_valid_image_inspection(tmp_path):
    image_path = tmp_path / "sample.png"

    image = Image.new("RGB", (224, 224), color="white")
    image.save(image_path)

    result = inspect_image(image_path)

    assert result["status"] == "valid"
    assert result["width"] == 224
    assert result["height"] == 224


def test_corrupted_image_detection(tmp_path):
    image_path = tmp_path / "corrupted.jpg"
    image_path.write_text("This is not an image")

    result = inspect_image(image_path)

    assert result["status"] == "corrupted"


def test_dataset_audit(tmp_path):
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "audit_results"

    create_test_dataset(dataset_dir)

    summary = audit_dataset(
        dataset_root=dataset_dir,
        output_dir=output_dir,
    )

    assert summary["total_supported_images"] == 4
    assert summary["valid_images"] == 4
    assert summary["corrupted_images"] == 0
    assert summary["unassigned_images"] == 0

    assert len(summary["missing_classes"]) == 0

    assert (output_dir / "audit_summary.json").exists()
    assert (output_dir / "image_inventory.csv").exists()


def test_missing_dataset_rejected(tmp_path):
    missing_dir = tmp_path / "missing_dataset"

    try:
        audit_dataset(missing_dir)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "Expected FileNotFoundError for missing dataset"
        )
