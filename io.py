from __future__ import annotations
from pathlib import Path
import logging
from PIL import Image, ExifTags
import numpy as np
import cv2

LOGGER = logging.getLogger(__name__)

def read_image(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                exif = img._getexif()
                if exif and orientation in exif:
                    if exif[orientation] == 3:
                        img = img.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        img = img.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        img = img.rotate(90, expand=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("EXIF orientation failed: %s", exc)
    return np.array(img)

def write_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
