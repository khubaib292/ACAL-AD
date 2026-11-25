
# """
# ACAL-AD configuration loader and validator.

# Loads configs/config.yaml, validates the configuration,
# and resolves project paths.
# """

# from pathlib import Path
# from typing import Any, Dict, Union

# import yaml


# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# class ConfigurationError(ValueError):
#     """Raised when the project configuration is invalid."""


# def require_section(
#     config: Dict[str, Any],
#     section: str
# ) -> Dict[str, Any]:
#     """Return a required dictionary section."""

#     value = config.get(section)

#     if not isinstance(value, dict):
#         raise ConfigurationError(
#             f"Missing or invalid configuration section: '{section}'"
#         )

#     return value


# def validate_config(config: Dict[str, Any]) -> None:
#     """Validate the ACAL-AD configuration."""

#     if not isinstance(config, dict):
#         raise ConfigurationError(
#             "Configuration must be a dictionary."
#         )

#     required_sections = [
#         "project",
#         "paths",
#         "dataset",
#         "data_split",
#         "preprocessing",
#         "model",
#         "loss",
#         "training",
#         "experiments",
#         "evaluation",
#         "logging",
#         "checkpoints",
#         "reproducibility",
#         "application",
#         "research",
#     ]

#     for section in required_sections:
#         require_section(config, section)

#     # Project validation
#     project = config["project"]

#     if not isinstance(project.get("name"), str):
#         raise ConfigurationError(
#             "project.name must be a string."
#         )

#     # Dataset validation
#     dataset = config["dataset"]
#     classes = dataset.get("classes")

#     if not isinstance(classes, list) or len(classes) != 4:
#         raise ConfigurationError(
#             "ACAL-AD requires exactly four dataset classes."
#         )

#     if not all(
#         isinstance(name, str) and name.strip()
#         for name in classes
#     ):
#         raise ConfigurationError(
#             "Every class name must be a non-empty string."
#         )

#     if len(classes) != len(set(classes)):
#         raise ConfigurationError(
#             "Duplicate dataset class names are not allowed."
#         )

#     if dataset.get("num_classes") != len(classes):
#         raise ConfigurationError(
#             "dataset.num_classes must match the class count."
#         )

#     # Data split validation
#     data_split = config["data_split"]
#     split_names = ["train", "validation", "test"]
#     split_values = []

#     for name in split_names:
#         value = data_split.get(name)

#         if (
#             isinstance(value, bool)
#             or not isinstance(value, (int, float))
#         ):
#             raise ConfigurationError(
#                 f"data_split.{name} must be numeric."
#             )

#         if not 0 < value < 1:
#             raise ConfigurationError(
#                 f"data_split.{name} must be between 0 and 1."
#             )

#         split_values.append(value)

#     if abs(sum(split_values) - 1.0) > 1e-8:
#         raise ConfigurationError(
#             "Data split ratios must sum to 1.0."
#         )

#     # Preprocessing validation
#     preprocessing = config["preprocessing"]

#     image_size = preprocessing.get("image_size")

#     if (
#         isinstance(image_size, bool)
#         or not isinstance(image_size, int)
#         or image_size <= 0
#     ):
#         raise ConfigurationError(
#             "preprocessing.image_size must be positive."
#         )

#     if preprocessing.get("channels") != 3:
#         raise ConfigurationError(
#             "Three-channel RGB images are required."
#         )

#     normalization = preprocessing.get("normalization", {})
#     mean = normalization.get("mean")
#     std = normalization.get("std")

#     if (
#         not isinstance(mean, list)
#         or not isinstance(std, list)
#         or len(mean) != 3
#         or len(std) != 3
#     ):
#         raise ConfigurationError(
#             "Normalization mean and std must contain three values."
#         )

#     if any(
#         isinstance(value, bool)
#         or not isinstance(value, (int, float))
#         for value in mean + std
#     ):
#         raise ConfigurationError(
#             "Normalization values must be numeric."
#         )

#     if any(value <= 0 for value in std):
#         raise ConfigurationError(
#             "Normalization standard deviations must be positive."
#         )

#     # Model validation
#     model = config["model"]

#     if model.get("architecture") != "PredatorNet":
#         raise ConfigurationError(
#             "The architecture must be PredatorNet."
#         )

#     if model.get("num_classes") != len(classes):
#         raise ConfigurationError(
#             "Model and dataset class counts must match."
#         )

#     backbone = model.get("backbone")

#     if (
#         not isinstance(backbone, dict)
#         or backbone.get("name") != "ResNet50"
#     ):
#         raise ConfigurationError(
#             "PredatorNet must use the configured ResNet50 backbone."
#         )

#     # Loss validation
#     loss = config["loss"]

#     baseline = loss.get("baseline")
#     proposed = loss.get("proposed")

#     if not isinstance(baseline, dict):
#         raise ConfigurationError(
#             "Baseline loss configuration is missing."
#         )

#     if not isinstance(proposed, dict):
#         raise ConfigurationError(
#             "Proposed loss configuration is missing."
#         )

#     if baseline.get("name") != "BGRA":
#         raise ConfigurationError(
#             "Baseline loss must be BGRA."
#         )

#     if proposed.get("name") != "AdaptiveConfusionAwareLoss":
#         raise ConfigurationError(
#             "Proposed loss must be AdaptiveConfusionAwareLoss."
#         )

#     # Training validation
#     training = config["training"]

#     for field in ["epochs", "batch_size"]:
#         value = training.get(field)

#         if (
#             isinstance(value, bool)
#             or not isinstance(value, int)
#             or value <= 0
#         ):
#             raise ConfigurationError(
#                 f"training.{field} must be a positive integer."
#             )

#     optimizer = training.get("optimizer")

#     if not isinstance(optimizer, dict):
#         raise ConfigurationError(
#             "training.optimizer must be a dictionary."
#         )

#     learning_rate = optimizer.get("learning_rate")
#     weight_decay = optimizer.get("weight_decay")

#     if (
#         isinstance(learning_rate, bool)
#         or not isinstance(learning_rate, (int, float))
#         or learning_rate <= 0
#     ):
#         raise ConfigurationError(
#             "Optimizer learning rate must be positive."
#         )

#     if (
#         isinstance(weight_decay, bool)
#         or not isinstance(weight_decay, (int, float))
#         or weight_decay < 0
#     ):
#         raise ConfigurationError(
#             "Optimizer weight decay must be non-negative."
#         )

#     # Experiment validation
#     experiments = config["experiments"]

#     for name in ["baseline", "proposed"]:
#         experiment = experiments.get(name)

#         if not isinstance(experiment, dict):
#             raise ConfigurationError(
#                 f"Missing experiment configuration: {name}"
#             )

#         if experiment.get("architecture") != "PredatorNet":
#             raise ConfigurationError(
#                 f"{name} experiment must use PredatorNet."
#             )

#     # Research integrity validation
#     research = config["research"]

#     protected_settings = [
#         "use_test_data_for_training",
#         "use_test_data_for_tuning",
#         "update_confusion_statistics_from_validation",
#         "update_confusion_statistics_from_test",
#     ]

#     for setting in protected_settings:
#         if research.get(setting, False) is not False:
#             raise ConfigurationError(
#                 f"Research integrity violation: {setting} must be false."
#             )


# def resolve_project_paths(
#     config: Dict[str, Any]
# ) -> Dict[str, Any]:
#     """Convert configured paths into absolute paths."""

#     resolved = dict(config)
#     resolved["paths"] = dict(config["paths"])

#     for name, value in resolved["paths"].items():

#         if not isinstance(value, (str, Path)):
#             raise ConfigurationError(
#                 f"paths.{name} must be a string or Path."
#             )

#         path = Path(value)

#         if not path.is_absolute():
#             path = PROJECT_ROOT / path

#         resolved["paths"][name] = path.resolve()

#     return resolved


# def load_config(
#     config_path: Union[str, Path] = DEFAULT_CONFIG_PATH
# ) -> Dict[str, Any]:
#     """Load, validate and resolve the configuration."""

#     config_path = Path(config_path)

#     if not config_path.is_absolute():
#         config_path = PROJECT_ROOT / config_path

#     if not config_path.exists():
#         raise FileNotFoundError(
#             f"Configuration file not found: {config_path}"
#         )

#     try:
#         with config_path.open("r", encoding="utf-8") as file:
#             config = yaml.safe_load(file)

#     except yaml.YAMLError as error:
#         raise ConfigurationError(
#             f"Invalid YAML configuration: {error}"
#         ) from error

#     if config is None:
#         raise ConfigurationError(
#             "Configuration file is empty."
#         )

#     validate_config(config)

#     return resolve_project_paths(config)


# def print_config_summary(
#     config: Dict[str, Any]
# ) -> None:
#     """Print a summary of the loaded configuration."""

#     print("\n" + "=" * 55)
#     print("ACAL-AD CONFIGURATION SUMMARY")
#     print("=" * 55)

#     print(f"Project       : {config['project']['name']}")
#     print(f"Architecture  : {config['model']['architecture']}")
#     print(f"Backbone      : {config['model']['backbone']['name']}")
#     print(f"Classes       : {config['dataset']['num_classes']}")
#     print(f"Image size    : {config['preprocessing']['image_size']}")

#     print(
#         f"Data split    : "
#         f"{config['data_split']['train']}/"
#         f"{config['data_split']['validation']}/"
#         f"{config['data_split']['test']}"
#     )

#     print(f"Epochs        : {config['training']['epochs']}")
#     print(f"Batch size    : {config['training']['batch_size']}")

#     print(f"Baseline loss : {config['loss']['baseline']['name']}")
#     print(f"Proposed loss : {config['loss']['proposed']['name']}")

#     print("\nProject paths:")

#     for name, path in config["paths"].items():
#         print(f"  {name}: {path}")

#     print("=" * 55)
#     print("Configuration loaded and validated successfully!")
#     print("=" * 55)


from pathlib import Path
from typing import Any, Dict, Union
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ConfigurationError(ValueError):
    pass


def require_section(config: Dict[str, Any], section: str):
    value = config.get(section)
    if not isinstance(value, dict):
        raise ConfigurationError(
            f"Missing or invalid configuration section: '{section}'"
        )
    return value


def validate_config(config: Dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a dictionary.")

    required = [
        "project", "paths", "dataset", "data_split",
        "preprocessing", "model", "loss", "training",
        "experiments", "evaluation", "logging",
        "checkpoints", "reproducibility",
        "application", "research"
    ]

    for section in required:
        require_section(config, section)

    classes = config["dataset"].get("classes")

    expected_classes = [
        "MildDemented",
        "ModerateDemented",
        "NonDemented",
        "VeryMildDemented"
    ]

    if classes != expected_classes:
        raise ConfigurationError(
            f"Class order must be {expected_classes}; got {classes}"
        )

    if config["dataset"].get("num_classes") != 4:
        raise ConfigurationError("dataset.num_classes must be 4.")

    if config["model"].get("architecture") != "PredatorNet":
        raise ConfigurationError("Model architecture must be PredatorNet.")

    if config["model"].get("num_classes") != 4:
        raise ConfigurationError("model.num_classes must be 4.")

    split = config["data_split"]
    values = []

    for key in ("train", "validation", "test"):
        value = split.get(key)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError(f"data_split.{key} must be numeric.")

        if not 0 < value < 1:
            raise ConfigurationError(f"data_split.{key} must be between 0 and 1.")

        values.append(value)

    if abs(sum(values) - 1.0) > 1e-8:
        raise ConfigurationError("Data split ratios must sum to 1.0.")

    preprocessing = config["preprocessing"]

    if preprocessing.get("image_size") != 224:
        raise ConfigurationError("This PredatorNet pipeline expects 224x224 images.")

    if preprocessing.get("channels") != 3:
        raise ConfigurationError("Three-channel RGB images are required.")

    normalization = preprocessing.get("normalization", {})
    mean = normalization.get("mean")
    std = normalization.get("std")

    if not isinstance(mean, list) or not isinstance(std, list):
        raise ConfigurationError("Normalization mean/std must be lists.")

    if len(mean) != 3 or len(std) != 3:
        raise ConfigurationError("Normalization mean/std must have 3 values.")

    if any(value <= 0 for value in std):
        raise ConfigurationError("Normalization standard deviations must be positive.")

    if config["loss"].get("baseline", {}).get("name") != "BGRA":
        raise ConfigurationError("Baseline loss must be BGRA.")

    if config["loss"].get("proposed", {}).get("name") != "AdaptiveConfusionAwareLoss":
        raise ConfigurationError("Proposed loss must be AdaptiveConfusionAwareLoss.")

    for name in ("baseline", "proposed"):
        experiment = config["experiments"].get(name)
        if not isinstance(experiment, dict):
            raise ConfigurationError(f"Missing experiment: {name}")
        if experiment.get("architecture") != "PredatorNet":
            raise ConfigurationError(f"{name} must use PredatorNet.")

    research = config["research"]

    protected = [
        "use_test_data_for_training",
        "use_test_data_for_tuning",
        "update_confusion_statistics_from_validation",
        "update_confusion_statistics_from_test"
    ]

    for key in protected:
        if research.get(key) is not False:
            raise ConfigurationError(
                f"Research integrity setting must be false: {key}"
            )


def resolve_project_paths(config: Dict[str, Any]):
    resolved = dict(config)
    resolved["paths"] = {}

    for name, value in config["paths"].items():
        path = Path(value)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        resolved["paths"][name] = path.resolve()

    return resolved


def load_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH):
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ConfigurationError("Configuration file is empty.")

    validate_config(config)
    return resolve_project_paths(config)


def print_config_summary(config):
    print("\nACAL-AD CONFIGURATION")
    print("---------------------")
    print("Project:", config["project"]["name"])
    print("Architecture:", config["model"]["architecture"])
    print("Classes:", config["dataset"]["classes"])
    print("Image size:", config["preprocessing"]["image_size"])
    print("Split:", config["data_split"])
    print("Paths:")

    for name, path in config["paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    cfg = load_config()
    print_config_summary(cfg)
