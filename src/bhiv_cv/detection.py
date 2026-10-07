"""
Stage 1: Detection.

Responsibility: find face bounding boxes in a raw image.
Explicitly NOT this module's job: preprocessing, embedding, matching, or
deciding what a low-confidence detection means for access.
"""
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .config import DetectionConfig
from .exceptions import ResourceLimitError


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int
    detector_score: float  # pseudo-confidence from the cascade's reject levels


class FaceDetector:
    """Thin, swappable wrapper around a detector backend.

    Backend today: OpenCV Haar Cascade (bundled with opencv-python, Intel/
    OpenCV licensed, zero external download, zero training-dataset
    provenance question -- see docs/LICENSE_AND_PROVENANCE.md).

    To swap in a DNN or PyTorch-based detector later, implement a class with
    the same `.detect(image) -> List[BoundingBox]` signature and change
    `RepresentationConfig`/wiring in pipeline.py. Nothing else needs to
    change.
    """

    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        import os
        # project root is 3 levels up from this file (src/bhiv_cv/detection.py)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cascade_path = os.path.join(base_dir, "data", "models", self.config.cascade_name)
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade at {cascade_path}")

    def detect(self, image_bgr: np.ndarray) -> List[BoundingBox]:
        h, w = image_bgr.shape[:2]
        megapixels = (h * w) / 1_000_000
        if megapixels > self.config.max_image_megapixels:
            raise ResourceLimitError(
                f"Image is {megapixels:.1f} MP, exceeds configured ceiling "
                f"of {self.config.max_image_megapixels} MP."
            )

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        boxes, reject_levels, level_weights = self._cascade.detectMultiScale3(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=self.config.min_size,
            outputRejectLevels=True,
        )

        results = []
        for (x, y, bw, bh), weight in zip(boxes, level_weights):
            # level_weights are unbounded cascade-stage scores; squash to (0,1)
            score = float(1 / (1 + np.exp(-(weight - 3))))
            results.append(BoundingBox(int(x), int(y), int(bw), int(bh), score))

        results.sort(key=lambda b: b.detector_score, reverse=True)
        return results[: self.config.max_faces_returned]
