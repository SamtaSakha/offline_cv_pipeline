"""
Top-level orchestrator:

    Input Image -> Detection -> Preprocessing -> Representation
                -> (enroll into Gallery) OR (Matching against Gallery)
                -> IdentityResult (the ONLY thing that leaves this module)

This is the single place that turns internal exceptions into the bounded
IdentityResult.failure(...) contract, so callers never see a raw traceback
from a partner module.
"""
import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .config import PipelineConfig, DEFAULT_CONFIG
from .detection import FaceDetector
from .preprocessing import crop_and_normalize
from .representation import LBPGridEmbedder
from .matching import match as run_match
from .gallery import Gallery
from .contracts import IdentityResult
from .exceptions import (
    CVPipelineError,
    UnsupportedInputError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
)

logger = logging.getLogger("bhiv_cv.pipeline")


def load_image(source: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """Decode a path / raw bytes / ndarray into a BGR uint8 image.

    Raises UnsupportedInputError for anything that cannot be decoded --
    this is the pipeline's single entry point for "garbage in".
    """
    if isinstance(source, np.ndarray):
        img = source
    elif isinstance(source, (bytes, bytearray)):
        buf = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise UnsupportedInputError(f"No such file: {source}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    else:
        raise UnsupportedInputError(f"Unsupported input type: {type(source)}")

    if img is None or img.size == 0:
        raise UnsupportedInputError("Input could not be decoded as an image.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise UnsupportedInputError(f"Expected a 3-channel color image, got shape {img.shape}")

    return img


class CVPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.detector = FaceDetector(self.config.detection)
        self.embedder = LBPGridEmbedder(self.config.representation)
        self.gallery = Gallery()

    # ---- enrollment -----------------------------------------------------
    def enroll(self, source, identity: str, allow_update: bool = False) -> IdentityResult:
        try:
            image = load_image(source)
            boxes = self.detector.detect(image)
            if len(boxes) == 0:
                raise NoFaceDetectedError("No face found during enrollment.")
            if len(boxes) > 1:
                raise MultipleFacesDetectedError(len(boxes), boxes)

            face = crop_and_normalize(image, boxes[0], self.config.preprocess)
            embedding = self.embedder.embed(face)
            self.gallery.enroll(identity, embedding, allow_update=allow_update)

            return IdentityResult.success(
                identity=identity,
                match=True,
                similarity=1.0,
                confidence=1.0,
                model_version=self.embedder.model_version,
            )
        except CVPipelineError as e:
            logger.warning("enroll() failed: %s: %s", type(e).__name__, e)
            return IdentityResult.failure(type(e).__name__, self.embedder.model_version)

    # ---- identification ---------------------------------------------------
    def identify(self, source, require_single_face: bool = True) -> IdentityResult:
        try:
            image = load_image(source)
            boxes = self.detector.detect(image)

            if len(boxes) == 0:
                raise NoFaceDetectedError("No face found in probe image.")
            if require_single_face and len(boxes) > 1:
                raise MultipleFacesDetectedError(len(boxes), boxes)

            # policy: if multiple faces allowed, identify the largest (closest) face
            box = max(boxes, key=lambda b: b.w * b.h) if not require_single_face else boxes[0]

            face = crop_and_normalize(image, box, self.config.preprocess)
            embedding = self.embedder.embed(face)

            result = run_match(embedding, self.gallery.as_dict(), self.config.matching)

            return IdentityResult.success(
                identity=result.identity,
                match=result.match,
                similarity=result.similarity,
                confidence=result.confidence,
                model_version=self.embedder.model_version,
            )
        except CVPipelineError as e:
            logger.warning("identify() failed: %s: %s", type(e).__name__, e)
            return IdentityResult.failure(type(e).__name__, self.embedder.model_version)
