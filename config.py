from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    image_size: int
    roi_padding: float
    binary_threshold: float
    tissue_num_classes: int
    wound_classes: list[str]
    tissue_classes: list[str]
    device: str
    weights_dir: Path
    color_correction: str | None
    morph_kernel: int
    min_obj_size: int
    debug: bool

def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(
        image_size=data["image_size"],
        roi_padding=data["roi_padding"],
        binary_threshold=data["binary_threshold"],
        tissue_num_classes=len(data["tissue_classes"]),
        wound_classes=data["wound_classes"],
        tissue_classes=data["tissue_classes"],
        device=data["device"],
        weights_dir=Path(data["weights_dir"]),
        color_correction=data.get("color_correction"),
        morph_kernel=data["morph_kernel"],
        min_obj_size=data["min_obj_size"],
        debug=data.get("debug", False),
    )
