
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predatornet import PredatorNet
from src.checkpoint_utils import (
    CHECKPOINTS,
    get_device,
    load_checkpoint
)


def main():
    device = get_device()

    print("\nACAL-AD CHECKPOINT VERIFICATION")
    print("=" * 60)
    print("Device:", device)

    failed = []

    for name, path in CHECKPOINTS.items():
        print(f"\nChecking {name}...")

        try:
            model = PredatorNet(
                num_classes=4,
                pretrained=False
            )

            load_checkpoint(model, path, device)
            print("[SUCCESS] Model loaded.")

        except Exception as error:
            failed.append(name)
            print("[FAILED]", error)

    print("\n" + "=" * 60)

    if failed:
        print("Some checkpoints could not be loaded:")
        for name in failed:
            print("-", name)
        raise SystemExit(1)

    print("All four checkpoints loaded successfully.")
    print("Ready for evaluation.")


if __name__ == "__main__":
    main()
