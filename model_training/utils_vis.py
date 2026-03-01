from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import cv2
from torchvision.utils import make_grid

def _to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    """img_tensor: [3,H,W], denormalized outside."""
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return (img * 255).clip(0, 255).astype(np.uint8)

def overlay_contours(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """Dessine le contour du mask sur image."""
    mask_uint = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness=2)
    return overlay

@torch.no_grad()
def save_val_visuals(
    model,
    val_loader,
    device,
    out_dir: Path,
    epoch: int,
    max_images: int = 3,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)["out"]
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        for i in range(images.size(0)):
            if saved >= max_images:
                return
            # dé-normaliser (ImageNet)
            img = images[i].detach().cpu()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img_np = _to_numpy(img)
            gt = masks[i].squeeze().cpu().numpy()
            pr = preds[i].squeeze().cpu().numpy()

            gt_contour = overlay_contours(img_np, (gt > 0.5).astype(np.uint8), color=(0, 255, 0))
            pr_contour = overlay_contours(img_np, (pr > 0.5).astype(np.uint8), color=(255, 0, 0))
            mix = cv2.addWeighted(gt_contour, 0.5, pr_contour, 0.5, 0)

            base = out_dir / f"epoch{epoch:03d}_idx{saved}"
            cv2.imwrite(str(base.with_suffix(".overlay.png")), cv2.cvtColor(mix, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(base.with_suffix(".gt.png")), (gt * 255).astype(np.uint8))
            cv2.imwrite(str(base.with_suffix(".pred.png")), (pr * 255).astype(np.uint8))
            saved += 1
        if saved >= max_images:
            return
