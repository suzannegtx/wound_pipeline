from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PALETTE = {
    0: (0, 0, 0),
    1: (0, 200, 0),      # granulation
    2: (255, 220, 0),    # slough
    3: (150, 70, 50),    # eschar
}

def colorize(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in PALETTE.items():
        rgb[mask == idx] = color
    return rgb

def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 255), alpha=0.4) -> np.ndarray:
    overlay = image.copy()
    overlay[mask.astype(bool)] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def save_overlay(image: np.ndarray, mask: np.ndarray, path: Path, color=(0, 255, 255), alpha=0.4) -> None:
    out = overlay_mask(image, mask, color, alpha)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

def save_tissue_overlay(image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    colored = colorize(mask)
    out = cv2.addWeighted(colored, 0.45, image, 0.55, 0)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

def show_figures(figures: list[tuple[str, np.ndarray]]) -> None:
    cols = min(3, len(figures))
    rows = int(np.ceil(len(figures) / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (title, img) in enumerate(figures, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
