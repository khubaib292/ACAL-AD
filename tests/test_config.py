
"""
Automated tests for ACAL-AD configuration.
"""

from copy import deepcopy

import pytest

from src.config import (
    ConfigurationError,
    load_config,
    validate_config,
)


@pytest.fixture
def config():
    """Load the project configuration."""
    return load_config()


def test_config_loads_successfully(config):
    """Verify that the configuration loads."""
    assert config["project"]["name"] == "ACAL-AD"


def test_dataset_has_four_classes(config):
    """Verify that exactly four classes are configured."""
    assert len(config["dataset"]["classes"]) == 4


def test_model_and_dataset_classes_match(config):
    """Verify model and dataset class counts."""
    assert (
        config["model"]["num_classes"]
        == config["dataset"]["num_classes"]
    )


def test_paths_are_absolute(config):
    """Verify that all project paths are absolute."""
    for path in config["paths"].values():
        assert path.is_absolute()


def test_invalid_split_ratios_are_rejected(config):
    """Verify that invalid split ratios raise an error."""
    invalid_config = deepcopy(config)

    invalid_config["data_split"]["train"] = 0.80
    invalid_config["data_split"]["validation"] = 0.15
    invalid_config["data_split"]["test"] = 0.15

    with pytest.raises(ConfigurationError):
        validate_config(invalid_config)


def test_duplicate_class_names_are_rejected(config):
    """Verify that duplicate class names raise an error."""
    invalid_config = deepcopy(config)

    invalid_config["dataset"]["classes"].append(
        invalid_config["dataset"]["classes"][0]
    )

    with pytest.raises(ConfigurationError):
        validate_config(invalid_config)


def test_test_data_is_not_used_for_training(config):
    """Verify that test data is protected."""
    assert config["research"]["use_test_data_for_training"] is False


def test_validation_does_not_update_confusion_statistics(config):
    """Verify that validation data cannot update confusion statistics."""
    assert (
        config["research"]["update_confusion_statistics_from_validation"]
        is False
    )
