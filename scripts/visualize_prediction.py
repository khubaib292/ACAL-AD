
from pathlib import Path
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image

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

CHECKPOINTS = {
    "BGRA": PROJECT_ROOT / "models" / "PredatorNet_BGRA_Seed123.pt",
    "ACAL": PROJECT_ROOT / "models" / "PredatorNet_ACAL_Seed123.pt"
}

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations"


def load_model(checkpoint_path, device):
    model = PredatorNet(
        num_classes=len(CLASS_NAMES),
        pretrained=False
    )

    model = load_checkpoint(
        model,
        checkpoint_path,
        device
    )

    model.eval()
    return model


def normalize_heatmap(heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)

    minimum = heatmap.min()
    maximum = heatmap.max()

    if maximum - minimum < 1e-8:
        return np.zeros_like(heatmap)

    return (heatmap - minimum) / (maximum - minimum)


@torch.inference_mode()
def generate_visualization(model, image_tensor, device):
    image_batch = image_tensor.unsqueeze(0).to(device)

    outputs = model(image_batch)

    probabilities = F.softmax(
        outputs["logits"],
        dim=1
    )[0]

    predicted_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_index].item()) * 100

    # Region importance map: [1, 1, 7, 7]
    region_map = outputs["region_map"][0, 0]

    region_map = F.interpolate(
        region_map.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    region_map = normalize_heatmap(
        region_map.cpu().numpy()
    )

    # Attention map: [1, 49, 49]
    attention = outputs["attention_map"][0]

    if attention.ndim != 2:
        raise ValueError(
            f"Unexpected attention shape: {attention.shape}"
        )

    side = int(attention.shape[0] ** 0.5)

    if side * side != attention.shape[0]:
        raise ValueError(
            f"Cannot reshape attention map: {attention.shape}"
        )

    # Average attention received by each spatial position.
    attention_map = attention.mean(dim=0).reshape(side, side)

    attention_map = F.interpolate(
        attention_map.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    attention_map = normalize_heatmap(
        attention_map.cpu().numpy()
    )

    gamma = float(
        model.eagle_attention.gamma.detach().cpu().item()
    )

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "probabilities": probabilities.cpu().numpy(),
        "region_map": region_map,
        "attention_map": attention_map,
        "gamma": gamma
    }


def save_comparison(original_image, results, image_path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    model_names = ["BGRA", "ACAL"]

    for row, model_name in enumerate(model_names):
        result = results[model_name]

        axes[row, 0].imshow(original_image)
        axes[row, 0].set_title(
            f"{model_name} - Original MRI"
        )

        axes[row, 1].imshow(original_image)
        axes[row, 1].imshow(
            result["region_map"],
            cmap="jet",
            alpha=0.45
        )
        axes[row, 1].set_title(
            f"Region Importance\n"
            f"Prediction: {result['predicted_class']}"
        )

        axes[row, 2].imshow(original_image)
        axes[row, 2].imshow(
            result["attention_map"],
            cmap="jet",
            alpha=0.45
        )
        axes[row, 2].set_title(
            f"EagleAttention\n"
            f"Confidence: {result['confidence']:.2f}%"
        )

        for column in range(3):
            axes[row, column].axis("off")

    fig.suptitle(
        "ACAL-AD: BGRA vs ACAL Visualization",
        fontsize=16
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_visualization.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ACAL-AD model predictions and attention maps."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the MRI image."
    )

    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    for name, checkpoint in CHECKPOINTS.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"{name} checkpoint not found: {checkpoint}"
            )

    print("\nACAL-AD VISUALIZATION")
    print("=" * 60)

    device = get_device()
    print("Device:", device)

    with Image.open(image_path) as image:
        original_image = image.convert("RGB").resize((224, 224))

    image_tensor = preprocess_image(
        image_path,
        normalization="kaggle"
    )

    results = {}

    for model_name, checkpoint_path in CHECKPOINTS.items():
        print(f"\nEvaluating {model_name}...")

        model = load_model(
            checkpoint_path,
            device
        )

        results[model_name] = generate_visualization(
            model,
            image_tensor,
            device
        )

        result = results[model_name]

        print("Predicted class:", result["predicted_class"])
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Learned attention gamma: {result['gamma']:.6f}")

        if abs(result["gamma"]) < 1e-8:
            print(
                "Note: Gamma is approximately zero; "
                "the attention residual has negligible influence "
                "on the model's features."
            )

        print("\nClass probabilities:")

        for class_name, probability in zip(
            CLASS_NAMES,
            result["probabilities"]
        ):
            print(
                f"{class_name:<22}: {probability * 100:.2f}%"
            )

        del model

    output_path = save_comparison(
        original_image,
        results,
        image_path
    )

    print("\nVisualization completed.")
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
