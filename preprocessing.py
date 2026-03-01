from __future__ import annotations
import cv2
import numpy as np

def resize_with_pad(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded, scale, (left, top)

def undo_resize(mask: np.ndarray, orig_shape: tuple[int, int], scale: float, pad: tuple[int, int]) -> np.ndarray:
    left, top = pad
    size = mask.shape[0]
    cropped = mask[top:size - top, left:size - left]
    out = cv2.resize(cropped, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    return out

def gray_world(img: np.ndarray) -> np.ndarray:
    avg = img.mean(axis=(0, 1))
    scale = avg.mean() / (avg + 1e-8)
    return np.clip(img * scale, 0, 255).astype(np.uint8)

def clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe_op.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
