"""
Stage 2: Preprocessing.

Responsibility: turn a raw image + a bounding box into a normalised,
fixed-size, grayscale face crop ready for representation. Deterministic and
side-effect free so it can be unit-tested in isolation.
"""
import cv2
import numpy as np

from .config import PreprocessConfig
from .detection import BoundingBox


def crop_and_normalize(
    image_bgr: np.ndarray, box: BoundingBox, config: PreprocessConfig = None
) -> np.ndarray:
    config = config or PreprocessConfig()
    h_img, w_img = image_bgr.shape[:2]

    margin_w = int(box.w * config.face_margin)
    margin_h = int(box.h * config.face_margin)

    x1 = max(box.x - margin_w, 0)
    y1 = max(box.y - margin_h, 0)
    x2 = min(box.x + box.w + margin_w, w_img)
    y2 = min(box.y + box.h + margin_h, h_img)

    crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, config.target_size, interpolation=cv2.INTER_LINEAR)

    if config.equalize_histogram:
        resized = cv2.equalizeHist(resized)

    return resized  # uint8, shape == target_size
