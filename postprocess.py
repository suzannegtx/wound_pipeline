from __future__ import annotations
import cv2
import numpy as np

def clean_mask(mask: np.ndarray, min_size: int, kernel: int) -> np.ndarray:
    kernel_mat = np.ones((kernel, kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_mat)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_mat)
    num, labels = cv2.connectedComponents(mask.astype("uint8"))
    if num <= 1:
        return mask
    sizes = np.array([(labels == i).sum() for i in range(1, num)])
    keep = sizes >= min_size
    out = np.zeros_like(mask)
    for i, k in enumerate(keep, start=1):
        if k:
            out[labels == i] = 1
    return out
