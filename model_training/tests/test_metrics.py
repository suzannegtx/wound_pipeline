import torch
from metrics import precision_score, dice_score, iou_score

def test_metrics_perfect():
    tp, fp, fn = 10, 0, 0
    assert precision_score(tp, fp) == 1.0
    assert dice_score(tp, fp, fn) == 1.0
    assert iou_score(tp, fp, fn) == 1.0

def test_metrics_half_precision():
    tp, fp, fn = 5, 5, 0
    assert abs(precision_score(tp, fp) - 0.5) < 1e-6
    assert abs(dice_score(tp, fp, fn) - (2*5)/(2*5+5+0)) < 1e-6
    assert abs(iou_score(tp, fp, fn) - (5/(5+5+0))) < 1e-6

def test_metrics_fn_effect():
    tp, fp, fn = 5, 0, 5
    assert abs(dice_score(tp, fp, fn) - 0.5) < 1e-6
    assert abs(iou_score(tp, fp, fn) - (5/10)) < 1e-6
