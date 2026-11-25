
# from pathlib import Path

# import torch

# from src.models.predatornet import PredatorNet


# CLASS_NAMES = [
#     "MildDemented",
#     "ModerateDemented",
#     "NonDemented",
#     "VeryMildDemented",
# ]

# CHECKPOINT_NAMES = [
#     "PredatorNet_ACAL_Seed123.pt",
#     "PredatorNet_BGRA_Seed123.pt",
#     "PredatorNet_ACAL_Seed42.pt",
#     "PredatorNet_BGRA_Seed42.pt",
# ]


# def get_device():
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def extract_state_dict(checkpoint):
#     if not isinstance(checkpoint, dict):
#         raise TypeError("Checkpoint must contain a state dictionary.")

#     for key in ("model_state_dict", "state_dict", "model"):
#         value = checkpoint.get(key)
#         if isinstance(value, dict):
#             checkpoint = value
#             break

#     if not checkpoint or not all(
#         isinstance(key, str) for key in checkpoint
#     ):
#         raise ValueError("Could not identify model state dictionary.")

#     # Handle checkpoints saved using DataParallel.
#     if all(key.startswith("module.") for key in checkpoint):
#         checkpoint = {
#             key[len("module."):]: value
#             for key, value in checkpoint.items()
#         }

#     return checkpoint


# def load_predatornet(checkpoint_path, device=None):
#     checkpoint_path = Path(checkpoint_path)

#     if not checkpoint_path.is_file():
#         raise FileNotFoundError(
#             f"Checkpoint not found: {checkpoint_path}"
#         )

#     device = device or get_device()

#     try:
#         checkpoint = torch.load(
#             checkpoint_path,
#             map_location="cpu",
#             weights_only=True,
#         )
#     except TypeError:
#         checkpoint = torch.load(
#             checkpoint_path,
#             map_location="cpu",
#         )

#     state_dict = extract_state_dict(checkpoint)

#     model = PredatorNet(
#         num_classes=4,
#         pretrained=False,
#     )

#     model.load_state_dict(state_dict, strict=True)
#     model.to(device)
#     model.eval()

#     return model


from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = {
    "BGRA_Seed123": PROJECT_ROOT / "models" / "PredatorNet_BGRA_Seed123.pt",
    "ACAL_Seed123": PROJECT_ROOT / "models" / "PredatorNet_ACAL_Seed123.pt",
    "BGRA_Seed42": PROJECT_ROOT / "models" / "PredatorNet_BGRA_Seed42.pt",
    "ACAL_Seed42": PROJECT_ROOT / "models" / "PredatorNet_ACAL_Seed42.pt",
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("No model state dictionary found in checkpoint.")

    cleaned = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    return cleaned


def load_checkpoint(model, checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = get_device()

    # Only load checkpoints that you trust and created yourself.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

    state_dict = get_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    return model
