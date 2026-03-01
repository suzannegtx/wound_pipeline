from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import models

class WoundSegmentationModel:
    def __init__(self, weights_path: Path | None, device: torch.device, threshold: float = 0.5) -> None:
        self.device = device
        self.threshold = threshold
        self.model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=2)
        if weights_path and weights_path.exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(image)["out"]
        prob = F.softmax(out, dim=1)[:, 1:2]
        mask = (prob > self.threshold).float()
        return prob, mask
