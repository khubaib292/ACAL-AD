
# # from pathlib import Path
# # import sys
# # import json

# # import numpy as np
# # import pandas as pd
# # import torch

# # from sklearn.metrics import (
# #     accuracy_score,
# #     classification_report,
# #     confusion_matrix,
# #     precision_recall_fscore_support
# # )

# # PROJECT_ROOT = Path(__file__).resolve().parents[1]
# # sys.path.insert(0, str(PROJECT_ROOT))

# # from src.data.dataloader import create_dataloaders
# # from src.models.predatornet import PredatorNet
# # from src.utils.checkpoint_utils import (
# #     CHECKPOINTS,
# #     get_device,
# #     load_model_checkpoint
# # )

# # CLASS_NAMES = [
# #     "MildDemented",
# #     "ModerateDemented",
# #     "NonDemented",
# #     "VeryMildDemented"
# # ]


# # @torch.no_grad()
# # def evaluate_model(model, loader, device):
# #     model.eval()

# #     all_targets = []
# #     all_predictions = []
# #     all_probabilities = []

# #     for images, labels in loader:
# #         images = images.to(device)
# #         labels = labels.to(device)

# #         outputs = model(images)
# #         logits = outputs["logits"]

# #         probabilities = torch.softmax(logits, dim=1)
# #         predictions = logits.argmax(dim=1)

# #         all_targets.extend(labels.cpu().numpy())
# #         all_predictions.extend(predictions.cpu().numpy())
# #         all_probabilities.extend(probabilities.cpu().numpy())

# #     return (
# #         np.asarray(all_targets),
# #         np.asarray(all_predictions),
# #         np.asarray(all_probabilities)
# #     )


# # def main():
# #     device = get_device()

# #     _, _, test_loader = create_dataloaders(
# #         batch_size=16,
# #         num_workers=0,
# #         normalization="kaggle"
# #     )

# #     output_dir = PROJECT_ROOT / "outputs" / "evaluation"
# #     output_dir.mkdir(parents=True, exist_ok=True)

# #     summary = []

# #     for model_name, relative_path in CHECKPOINTS.items():
# #         print("\nEvaluating:", model_name)

# #         model = PredatorNet(
# #             num_classes=4,
# #             pretrained=False
# #         )

# #         checkpoint_path = PROJECT_ROOT / relative_path

# #         load_model_checkpoint(
# #             model,
# #             checkpoint_path,
# #             device
# #         )

# #         y_true, y_pred, y_prob = evaluate_model(
# #             model,
# #             test_loader,
# #             device
# #         )

# #         accuracy = accuracy_score(y_true, y_pred)

# #         precision, recall, f1, _ = (
# #             precision_recall_fscore_support(
# #                 y_true,
# #                 y_pred,
# #                 average="macro",
# #                 zero_division=0
# #             )
# #         )

# #         report = classification_report(
# #             y_true,
# #             y_pred,
# #             labels=list(range(4)),
# #             target_names=CLASS_NAMES,
# #             output_dict=True,
# #             zero_division=0
# #         )

# #         matrix = confusion_matrix(
# #             y_true,
# #             y_pred,
# #             labels=list(range(4))
# #         )

# #         safe_name = model_name.lower().replace(" ", "_")

# #         pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
# #             output_dir / f"{safe_name}_confusion_matrix.csv"
# #         )

# #         with open(
# #             output_dir / f"{safe_name}_classification_report.json",
# #             "w",
# #             encoding="utf-8"
# #         ) as file:
# #             json.dump(report, file, indent=4)

# #         summary.append({
# #             "model": model_name,
# #             "accuracy": float(accuracy),
# #             "macro_precision": float(precision),
# #             "macro_recall": float(recall),
# #             "macro_f1": float(f1)
# #         })

# #         print("Accuracy:", round(accuracy, 4))
# #         print("Macro precision:", round(precision, 4))
# #         print("Macro recall:", round(recall, 4))
# #         print("Macro F1:", round(f1, 4))

# #         del model

# #     pd.DataFrame(summary).to_csv(
# #         output_dir / "checkpoint_comparison.csv",
# #         index=False
# #     )

# #     print("\nSaved evaluation results to:", output_dir)


# # if __name__ == "__main__":
# #     main()


# from pathlib import Path
# import subprocess
# import sys


# PROJECT_ROOT = Path(__file__).resolve().parents[1]


# def main():
#     script = PROJECT_ROOT / "scripts" / "compare_experiments.py"

#     result = subprocess.run(
#         [sys.executable, str(script)],
#         cwd=str(PROJECT_ROOT)
#     )

#     raise SystemExit(result.returncode)


# if __name__ == "__main__":
#     main()


from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    script = PROJECT_ROOT / "scripts" / "compare_experiments.py"

    print("\nStarting model evaluation...")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT)
    )

    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
