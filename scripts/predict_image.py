
from pathlib import Path
import argparse
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import preprocess_image
from src.models.predatornet import PredatorNet
from src.checkpoint_utils import get_device, load_checkpoint


CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

# Seed 123 only
CHECKPOINTS = {
    "BGRA": PROJECT_ROOT / "models" / "PredatorNet_BGRA_Seed123.pt",
    "ACAL": PROJECT_ROOT / "models" / "PredatorNet_ACAL_Seed123.pt"
}


def load_model(checkpoint_path, device):
    model = PredatorNet(num_classes=4, pretrained=False)
    model = load_checkpoint(model, checkpoint_path, device)
    return model


def predict(model, image_tensor, device):
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        logits = outputs["logits"]
        probabilities = F.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    return predicted_index, probabilities.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Compare ACAL and BGRA on a new MRI image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the MRI image to predict."
    )

    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    for name, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"{name} checkpoint not found: {checkpoint_path}"
            )

    print("\n" + "=" * 65)
    print("          NEW MRI IMAGE PREDICTION DEMO")
    print("=" * 65)

    print(f"Input image : {image_path}")
    print("Comparison  : BGRA vs ACAL ")

    device = get_device()
    print(f"Device      : {device}")

    print("\nPreprocessing image...")
    image_tensor = preprocess_image(
        image_path,
        normalization="kaggle"
    )

    print(f"Image tensor shape: {tuple(image_tensor.shape)}")

    results = {}

    for model_name, checkpoint_path in CHECKPOINTS.items():

        print("\n" + "-" * 65)
        print(f"Loading {model_name}")

        model = load_model(checkpoint_path, device)

        predicted_index, probabilities = predict(
            model,
            image_tensor,
            device
        )

        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

        results[model_name] = {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        }

        print(f"\n{model_name} PREDICTION")
        print("-" * 40)
        print(f"Predicted class : {predicted_class}")
        print(f"Confidence      : {confidence:.2f}%")

        print("\nClass probabilities:")

        for class_name, probability in zip(
            CLASS_NAMES, probabilities
        ):
            print(
                f"{class_name:<22}: {probability * 100:7.2f}%"
            )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 65)
    print("                 FINAL COMPARISON")
    print("=" * 65)

    print(
        f"{'Model':<12} {'Predicted Class':<22} {'Confidence':>12}"
    )
    print("-" * 65)

    for model_name in ["BGRA", "ACAL"]:
        result = results[model_name]

        print(
            f"{model_name:<12} "
            f"{result['class']:<22} "
            f"{result['confidence']:>10.2f}%"
        )

    if results["BGRA"]["class"] == results["ACAL"]["class"]:
        print("\nBoth models predicted the same class.")
    else:
        print("\nThe models predicted different classes.")

    print("\nPrediction demo completed.")

if __name__ == "__main__":
    main()
