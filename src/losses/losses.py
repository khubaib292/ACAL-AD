import torch
import torch.nn as nn
import torch.nn.functional as F


def region_total_variation(region_map):
    """Mean absolute horizontal + vertical neighboring-pixel differences."""
    horizontal = torch.abs(region_map[:, :, :, 1:] - region_map[:, :, :, :-1])
    vertical = torch.abs(region_map[:, :, 1:, :] - region_map[:, :, :-1, :])
    return horizontal.mean() + vertical.mean()


class BaselineBGRALoss(nn.Module):
    """
    Implemented baseline: CE + lambda_smooth * region total variation.
    Confirm the exact smoothness definition against the paper before claiming
    an exact reproduction of the paper's BGRA objective.
    """
    def __init__(self, lambda_smooth: float = 0.01):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        ce = self.ce_loss(outputs["logits"], targets)
        smooth = region_total_variation(outputs["region_map"])
        total = ce + self.lambda_smooth * smooth
        return {
            "total_loss": total,
            "classification_loss": ce,
            "smoothness_loss": smooth,
        }


class AdaptiveConfusionAwareLoss(nn.Module):
    """
    Proposed ACAL implementation:
      CE + lambda_smooth * TV(region_map)
         + lambda_confusion * confusion_penalty.

    IMPORTANT:
    - Call update_training_statistics(logits, targets) ONLY on training batches.
    - Call finish_training_epoch() once after each complete training epoch.
    - Never update these statistics using validation or test data.
    - Hyperparameters are provisional until selected and frozen by protocol.
    """
    def __init__(
        self,
        lambda_smooth: float = 0.01,
        lambda_confusion: float = 0.10,
        num_classes: int = 4,
        ema_momentum: float = 0.90,
        smoothing: float = 1.0,
    ):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_confusion = lambda_confusion
        self.num_classes = num_classes
        self.ema_momentum = ema_momentum
        self.smoothing = smoothing
        self.ce_loss = nn.CrossEntropyLoss()

        weights = torch.ones(num_classes, num_classes) - torch.eye(num_classes)
        weights = weights / (num_classes - 1)
        self.register_buffer("confusion_weights", weights)
        self.register_buffer(
            "epoch_confusion_counts", torch.zeros(num_classes, num_classes)
        )

    @torch.no_grad()
    def update_training_statistics(self, logits, targets):
        """Accumulate argmax confusion counts from a training batch only."""
        predictions = logits.detach().argmax(dim=1)
        targets = targets.detach().long()
        flat_indices = targets * self.num_classes + predictions
        counts = torch.bincount(
            flat_indices, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        self.epoch_confusion_counts.add_(
            counts.to(
                device=self.epoch_confusion_counts.device,
                dtype=self.epoch_confusion_counts.dtype,
            )
        )

    @torch.no_grad()
    def finish_training_epoch(self):
        """Update EMA confusion weights and reset epoch counts."""
        counts = self.epoch_confusion_counts.clone()
        counts.fill_diagonal_(0)

        off_diagonal = 1 - torch.eye(
            self.num_classes, device=counts.device, dtype=counts.dtype
        )
        counts = counts + self.smoothing * off_diagonal
        row_sums = counts.sum(dim=1, keepdim=True)
        new_weights = counts / row_sums.clamp_min(1e-12)

        updated = (
            self.ema_momentum * self.confusion_weights
            + (1 - self.ema_momentum) * new_weights
        )
        updated.fill_diagonal_(0)
        updated = updated / updated.sum(dim=1, keepdim=True).clamp_min(1e-12)
        self.confusion_weights.copy_(updated)
        self.epoch_confusion_counts.zero_()

    def forward(self, outputs, targets):
        logits = outputs["logits"]
        region_map = outputs["region_map"]

        ce = self.ce_loss(logits, targets)
        smooth = region_total_variation(region_map)

        probabilities = F.softmax(logits, dim=1)
        sample_weights = self.confusion_weights[targets]
        confusion = (probabilities * sample_weights).sum(dim=1).mean()

        total = (
            ce
            + self.lambda_smooth * smooth
            + self.lambda_confusion * confusion
        )
        return {
            "total_loss": total,
            "classification_loss": ce,
            "smoothness_loss": smooth,
            "confusion_loss": confusion,
        }
