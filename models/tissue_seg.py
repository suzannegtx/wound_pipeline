from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import models


class TissueSegmentationModel:
    def __init__(
        self,
        weights_path: Path | None,
        device: torch.device,
        num_classes: int,
    ) -> None:
        self.device = device
        self.num_classes = num_classes

        # DeepLabv3-ResNet50 paramétré pour le nombre de classes demandé
        self.model = models.segmentation.deeplabv3_resnet50(
            weights=None, num_classes=num_classes
        )
        if weights_path and weights_path.exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: float tensor [B, 3, H, W], déjà normalisé pour le backbone.
        Returns:
            probs: float tensor [B, num_classes, H, W] contenant les probabilités par classe.
        """
        logits = self.model(image)["out"]          # [B, C, H, W]
        probs = F.softmax(logits, dim=1)           # [B, C, H, W]
        return probs
