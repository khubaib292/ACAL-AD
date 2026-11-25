import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

NUM_BRANCHES = 3


class EagleAttention(nn.Module):
    """Spatial self-attention over the ResNet feature-map positions."""
    def __init__(self, channels: int = 2048):
        super().__init__()
        reduced_channels = channels // 8
        self.query = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.shape
        q = self.query(x).flatten(2).transpose(1, 2)
        k = self.key(x).flatten(2)
        v = self.value(x).flatten(2).transpose(1, 2)
        attention = torch.bmm(q, k) / (q.shape[-1] ** 0.5)
        attention = F.softmax(attention, dim=-1)
        attended = torch.bmm(attention, v)
        attended = attended.transpose(1, 2).reshape(
            batch, channels, height, width
        )
        return x + self.gamma * attended, attention


class RegionImportance(nn.Module):
    """Produces a spatial importance map and weights feature activations."""
    def __init__(self, channels: int = 2048):
        super().__init__()
        self.region_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        region_map = torch.sigmoid(self.region_conv(x))
        return x * region_map, region_map


class WolfPackFusion(nn.Module):
    """Three classification branches; final logits are their mean."""
    def __init__(self, channels: int = 2048, num_classes: int = 4):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Linear(channels, num_classes) for _ in range(NUM_BRANCHES)
        ])

    def forward(self, x):
        features = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        branch_logits = [branch(features) for branch in self.branches]
        return torch.stack(branch_logits, dim=0).mean(dim=0)


class PredatorNet(nn.Module):
    """
    ACAL-AD PredatorNet implementation.

    pretrained=False avoids an automatic weight download. Set True only when
    pretrained ImageNet weights are available and the experiment protocol
    explicitly specifies their use.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.eagle_attention = EagleAttention(2048)
        self.region_importance = RegionImportance(2048)
        self.wolfpack_fusion = WolfPackFusion(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        attended_features, attention_map = self.eagle_attention(features)
        weighted_features, region_map = self.region_importance(attended_features)
        logits = self.wolfpack_fusion(weighted_features)
        return {
            "logits": logits,
            "region_map": region_map,
            "attention_map": attention_map,
        }
