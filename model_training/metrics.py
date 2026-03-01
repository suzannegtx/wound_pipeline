from __future__ import annotations

def precision_score(tp: float, fp: float, eps: float = 1e-7) -> float:
    return tp / (tp + fp + eps)

def dice_score(tp: float, fp: float, fn: float, eps: float = 1e-7) -> float:
    return 2 * tp / (2 * tp + fp + fn + eps)

def iou_score(tp: float, fp: float, fn: float, eps: float = 1e-7) -> float:
    return tp / (tp + fp + fn + eps)
