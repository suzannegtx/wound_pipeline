#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.auto import tqdm  # barre de progression

from wound_data import FootUlcerDataset, collate_fn
from losses import DiceLoss
from metrics import precision_score, dice_score, iou_score
from utils_vis import save_val_visuals


# ----------------------- utilitaires généraux ----------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="Mixed precision si GPU dispo")
    ap.add_argument("--save_every", type=int, default=5, help="sauvegarde last tous les N epochs")
    ap.add_argument("--vis_every", type=int, default=1, help="sauvegarde visuels val tous les N epochs")
    return ap.parse_args()


# ----------------------- boucle train/val --------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler | None,
    bce_loss: nn.Module,
    dice_loss: DiceLoss,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        with amp.autocast(enabled=scaler is not None):
            logits = model(images)["out"]  # [B,1,H,W]
            bce = bce_loss(logits, masks)
            dice = dice_loss(logits, masks)
            loss = bce + dice
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    bce_loss: nn.Module,
    dice_loss: DiceLoss,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    tp = fp = fn = 0.0
    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)["out"]  # [B,1,H,W]
        bce = bce_loss(logits, masks)
        dice_l = dice_loss(logits, masks)
        loss = bce + dice_l
        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        tp += (preds * masks).sum().item()
        fp += (preds * (1 - masks)).sum().item()
        fn += ((1 - preds) * masks).sum().item()

    eps = 1e-7
    precision = precision_score(tp, fp, eps)
    dice = dice_score(tp, fp, fn, eps)
    iou = iou_score(tp, fp, fn, eps)
    val_loss = total_loss / len(loader.dataset)
    return val_loss, precision, dice, iou


# ----------------------- sauvegardes & logs ------------------------- #
def maybe_save_ckpt(
    model: nn.Module,
    epoch: int,
    is_best: bool,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "deeplabv3_fuseg_last.pth")
    if is_best:
        torch.save(model.state_dict(), out_dir / "deeplabv3_fuseg_best.pth")


def append_csv(log_path: Path, row: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(row.keys())
    if not log_path.exists():
        log_path.write_text(header + "\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(",".join(f"{v}" for v in row.values()) + "\n")


# ----------------------- main --------------------------------------- #
def main() -> None:
    args = parse_args()
    set_seed()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    print(f"[INFO] Device: {device}")

    # Datasets & loaders
    train_ds = FootUlcerDataset(args.data_dir, split="train", img_size=args.img_size, augment=True)
    val_ds = FootUlcerDataset(args.data_dir, split="validation", img_size=args.img_size, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Modèle
    model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=1)
    model.to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = amp.GradScaler() if args.amp and device.type == "cuda" else None

    outputs_dir = Path("outputs")
    vis_dir = outputs_dir / "vis"
    log_csv = outputs_dir / "training_log.csv"
    metrics_json = outputs_dir / "metrics_summary.json"
    ckpt_dir = Path("download_dummy_weights")
    ckpt_dir.mkdir(exist_ok=True)

    best_dice = -1.0
    best_precision = -1.0
    best_summary: Dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, scaler, bce_loss, dice_loss, device)
        val_loss, precision, dice, iou = validate(model, val_dl, bce_loss, dice_loss, device)

        scheduler.step()

        # tie-break: dice puis precision
        is_best = (dice > best_dice) or (abs(dice - best_dice) < 1e-6 and precision > best_precision)
        if is_best:
            best_dice, best_precision = dice, precision
            best_summary = {
                "epoch": epoch,
                "precision": precision,
                "dice": dice,
                "iou": iou,
                "val_loss": val_loss,
            }

        if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
            maybe_save_ckpt(model, epoch, is_best, ckpt_dir)

        if epoch % args.vis_every == 0 or is_best:
            save_val_visuals(model, val_dl, device, vis_dir, epoch, max_images=3)

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "precision": f"{precision:.6f}",
            "dice": f"{dice:.6f}",
            "iou": f"{iou:.6f}",
            "best_dice": f"{best_dice:.6f}",
            "best_precision": f"{best_precision:.6f}",
        }
        append_csv(log_csv, row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"precision={precision:.4f} dice={dice:.4f} iou={iou:.4f} "
            f"best_dice={best_dice:.4f}"
        )

    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(best_summary, indent=2), encoding="utf-8")

    print("\n=== Résumé final ===")
    print(f"Best epoch: {best_summary.get('epoch', 'NA')}")
    print(f"Best precision: {best_summary.get('precision', 'NA'):.4f}")
    print(f"Best dice: {best_summary.get('dice', 'NA'):.4f}")
    print(f"Best IoU: {best_summary.get('iou', 'NA'):.4f}")
    print(f"Checkpoint (meilleur): {ckpt_dir / 'deeplabv3_fuseg_best.pth'}")
    print(f"Checkpoint (dernier): {ckpt_dir / 'deeplabv3_fuseg_last.pth'}")
    print(f"Logs CSV: {log_csv}")
    print(f"Visuels: {vis_dir}")


if __name__ == "__main__":
    main()
