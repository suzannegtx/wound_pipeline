from __future__ import annotations
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice loss pour sortie logit binaire (forme [B,1,H,W]).
    Applique sigmoïde en interne.
    """
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1, 2, 3))
        den = (probs + targets).sum(dim=(1, 2, 3)) + self.eps
        dice = num / den
        return 1 - dice.mean()
