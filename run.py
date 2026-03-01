from __future__ import annotations
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
import cv2

from .config import load_config
from .io import read_image, write_image
from .preprocessing import resize_with_pad, undo_resize, gray_world, clahe
from .roi import heuristic_roi
from .models.wound_seg import WoundSegmentationModel
from .models.tissue_seg import TissueSegmentationModel
from .models.wound_cls import WoundClassifier
from .postprocess import clean_mask
from .metrics import wound_metrics, tissue_stats
from .visualize import colorize, save_tissue_overlay, show_figures
from .utils import set_seed, save_json

LOGGER = logging.getLogger("wound_pipeline")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--roi-mode", choices=["heuristic", "model"], default="heuristic")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config(args.config)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else cfg.device)
    set_seed()

    img = read_image(args.image)
    if cfg.color_correction == "gray-world":
        img = gray_world(img)
    elif cfg.color_correction == "clahe":
        img = clahe(img)

    resized, scale, pad = resize_with_pad(img, cfg.image_size)
    roi_img, roi_mask_preview = heuristic_roi(img, cfg.image_size, cfg.roi_padding)
    write_image(Path(args.out) / Path(args.image).stem / "roi.jpg", roi_img)
    write_image(Path(args.out) / Path(args.image).stem / "roi_mask_preview.png", roi_mask_preview.astype(np.uint8))

    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    # Segmentation binaire plaie / non-plaie
    wound_model = WoundSegmentationModel(Path(cfg.weights_dir) / "wound_deeplabv3_r50.pth", device, cfg.binary_threshold)
    wound_prob, wound_mask_t = wound_model.predict(tensor)
    wound_mask_small = wound_mask_t.squeeze().cpu().numpy().astype(np.uint8)
    wound_mask_full = undo_resize(wound_mask_small, img.shape[:2], scale, pad)
    wound_mask_full = clean_mask(wound_mask_full, cfg.min_obj_size, cfg.morph_kernel)
    write_image(Path(args.out) / Path(args.image).stem / "wound_mask.png", wound_mask_full * 255)
    write_image(
        Path(args.out) / Path(args.image).stem / "overlay_wound.png",
        cv2.cvtColor(
            cv2.addWeighted(img, 0.55, cv2.cvtColor(wound_mask_full * 255, cv2.COLOR_GRAY2RGB), 0.45, 0),
            cv2.COLOR_RGB2BGR,
        ),
    )

    # Contours visibles des plaies détectées
    outline = cv2.morphologyEx(wound_mask_full.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    contour_img = img.copy()
    contour_img[outline.astype(bool)] = (255, 0, 0)  # contour en rouge
    write_image(Path(args.out) / Path(args.image).stem / "wound_contours.png", contour_img)

    # Segmentation des tissus
    tissue_model = TissueSegmentationModel(Path(cfg.weights_dir) / "tissue_seg.pth", device, cfg.tissue_num_classes + 1)
    tissue_prob = tissue_model.predict(tensor).cpu().numpy()[0]  # (C,H,W)
    tissue_mask_small = tissue_prob.argmax(axis=0).astype(np.uint8)
    tissue_mask_full = undo_resize(tissue_mask_small, img.shape[:2], scale, pad)
    tissue_mask_full[wound_mask_full == 0] = 0  # on ne garde la classe tissu que dans la plaie
    write_image(Path(args.out) / Path(args.image).stem / "tissue_mask.png", tissue_mask_full)
    colorized = colorize(tissue_mask_full)
    write_image(Path(args.out) / Path(args.image).stem / "tissue_mask_color.png", colorized)
    save_tissue_overlay(img, tissue_mask_full, Path(args.out) / Path(args.image).stem / "overlay_tissues.png")

    # Classification globale
    cls_model = WoundClassifier(Path(cfg.weights_dir) / "wound_classifier.pth", device, len(cfg.wound_classes))
    cls_input = torch.from_numpy(cv2.resize(img, (224, 224)).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    cls_probs = cls_model.predict(cls_input.to(device)).cpu().numpy()[0]
    class_probs = {c: float(p) for c, p in zip(cfg.wound_classes, cls_probs)}
    class_pred = cfg.wound_classes[int(cls_probs.argmax())]

    metrics = wound_metrics(wound_mask_full, img.shape[:2])
    summary = {
        "image": args.image,
        "device": str(device),
        "wound_metrics": metrics,
        "tissue_proportions": tissue_stats(tissue_mask_full, cfg.tissue_classes),
        "classification": {"probs": class_probs, "pred": class_pred},
        "outputs": {
            "roi": "roi.jpg",
            "wound_mask": "wound_mask.png",
            "tissue_mask": "tissue_mask.png",
            "tissue_mask_color": "tissue_mask_color.png",
            "overlay_wound": "overlay_wound.png",
            "overlay_tissues": "overlay_tissues.png",
            "wound_contours": "wound_contours.png",
        },
    }
    save_json(Path(args.out) / Path(args.image).stem / "summary.json", summary)

    if args.show:
        show_figures([
            ("ROI", roi_img),
            ("Wound overlay", cv2.cvtColor(cv2.imread(str(Path(args.out) / Path(args.image).stem / "overlay_wound.png")), cv2.COLOR_BGR2RGB)),
            ("Wound contours", contour_img),
            ("Tissue overlay", cv2.cvtColor(cv2.imread(str(Path(args.out) / Path(args.image).stem / "overlay_tissues.png")), cv2.COLOR_BGR2RGB)),
        ])


if __name__ == "__main__":
    main()
