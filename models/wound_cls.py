from __future__ import annotations
from pathlib import Path
import torch
from torchvision import models

class WoundClassifier:
    def __init__(self, weights_path: Path | None, device: torch.device, num_classes: int) -> None:
        self.device = device
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, num_classes)
        if weights_path and weights_path.exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(image), dim=1)
