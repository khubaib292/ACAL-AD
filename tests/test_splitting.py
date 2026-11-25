
import pytest

from src.splitting import stratified_split


def create_sample_data():
    samples = [
        f"image_{i}.jpg"
        for i in range(200)
    ]

    labels = (
        [0] * 50
        + [1] * 50
        + [2] * 50
        + [3] * 50
    )

    return samples, labels


def test_split_sizes():
    samples, labels = create_sample_data()

    result = stratified_split(samples, labels)

    assert len(result["train"]["samples"]) == 140
    assert len(result["validation"]["samples"]) == 30
    assert len(result["test"]["samples"]) == 30


def test_no_overlap_between_splits():
    samples, labels = create_sample_data()

    result = stratified_split(samples, labels)

    train = set(result["train"]["samples"])
    validation = set(result["validation"]["samples"])
    test = set(result["test"]["samples"])

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_class_distribution():
    samples, labels = create_sample_data()

    result = stratified_split(samples, labels)

    for split_name in ["train", "validation", "test"]:
        split_labels = result[split_name]["labels"]

        assert set(split_labels) == {0, 1, 2, 3}


def test_reproducibility():
    samples, labels = create_sample_data()

    first = stratified_split(samples, labels)
    second = stratified_split(samples, labels)

    assert first == second


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        stratified_split(
            ["image1.jpg", "image2.jpg"],
            [0],
        )


def test_empty_dataset():
    with pytest.raises(ValueError):
        stratified_split([], [])


def test_invalid_ratios():
    samples, labels = create_sample_data()

    with pytest.raises(ValueError):
        stratified_split(
            samples,
            labels,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.2,
        )
