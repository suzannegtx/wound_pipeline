from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, ColorJitter

class FootUlcerDataset(Dataset):
    """
    Charge les paires (image, masque) à partir de la structure FUSeg.
    Masques sont binaires 0/1. Les images sont normalisées ImageNet.
    """
    def __init__(self, root: Path, split: str, img_size: int = 512, augment: bool = False) -> None:
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.img_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "labels"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Images folder missing: {self.img_dir}")
        if split != "test" and not self.mask_dir.exists():
            raise FileNotFoundError(f"Labels folder missing: {self.mask_dir}")

        self.images: List[Path] = sorted(self.img_dir.glob("*"))
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def _load_pair(self, idx: int) -> Tuple[Image.Image, Image.Image | None]:
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        if self.split == "test":
            return img, None
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path.name}")
        mask = Image.open(mask_path).convert("L")
        return img, mask

    def _augment(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Flip horizontal
        if random.random() < 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)
        # Petite rotation (-10 à 10 degrés)
        angle = random.uniform(-10, 10)
        img = F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
        # Color jitter léger
        jitter = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)
        img = jitter(img)
        return img, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self._load_pair(idx)
        if self.split != "test" and self.augment:
            img, mask = self._augment(img, mask)

        # Resize
        img = F.resize(img, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
        if mask is not None:
            mask = F.resize(mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)

        # To tensor & normalisation ImageNet
        img_t = F.to_tensor(img)
        img_t = F.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if mask is None:
            mask_t = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        else:
            mask_np = np.array(mask, dtype=np.uint8)
            mask_bin = (mask_np > 127).astype(np.float32)  # 0/255 -> 0/1
            mask_t = torch.from_numpy(mask_bin).unsqueeze(0)
        return img_t, mask_t

def collate_fn(batch):
    # batch = list of tuples (img, mask)
    imgs, masks = zip(*batch)
    return torch.stack(imgs), torch.stack(masks)
