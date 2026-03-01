from __future__ import annotations
import cv2
import numpy as np
from .preprocessing import resize_with_pad, undo_resize

def heuristic_roi(img: np.ndarray, size: int, padding: float) -> tuple[np.ndarray, np.ndarray]:
    small, scale, pad = resize_with_pad(img, size)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 30, 50])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num, labels = cv2.connectedComponents(mask)
    if num < 2:
        bbox = (0, 0, small.shape[1], small.shape[0])
    else:
        areas = [(labels == i).sum() for i in range(1, num)]
        idx = int(np.argmax(areas) + 1)
        ys, xs = np.where(labels == idx)
        bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    pad_x, pad_y = int(w * padding), int(h * padding)
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(small.shape[1], x1 + pad_x), min(small.shape[0], y1 + pad_y)
    roi_small = small[y0:y1, x0:x1]
    mask_crop = np.zeros_like(mask)
    mask_crop[y0:y1, x0:x1] = 255
    roi_mask_full = undo_resize(mask_crop, img.shape[:2], scale, pad)
    roi_full = img[y0:int(y1 / scale), x0:int(x1 / scale)] if False else img  # fallback: full image
    return roi_full, roi_mask_full
