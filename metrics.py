from __future__ import annotations
import cv2
import numpy as np

def wound_metrics(mask: np.ndarray, image_shape: tuple[int, int]) -> dict:
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    return {
        "area_px": float(area),
        "perimeter_px": float(perimeter),
        "bbox": [int(x), int(y), int(w), int(h)],
        "solidity": float(area / hull_area) if hull_area else 0.0,
        "circularity": float(4 * np.pi * area / (perimeter**2 + 1e-8)),
        "aspect_ratio": float(w / (h + 1e-8)),
        "coverage_pct": float(area / (image_shape[0] * image_shape[1] + 1e-8)),
        "contour": cnt.squeeze(1).tolist(),
    }

def tissue_stats(tissue_mask: np.ndarray, tissue_classes: list[str]) -> dict:
    out: dict[str, float] = {}
    total = (tissue_mask > 0).sum() + 1e-8
    for idx, name in enumerate(tissue_classes, start=1):
        out[name] = float((tissue_mask == idx).sum() / total)
    return out
