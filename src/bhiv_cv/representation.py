"""
Stage 3: Representation (embedding).

Responsibility: turn a normalised face crop into a fixed-length numeric
vector ("embedding") that can be compared against other vectors.

Backend selected for this baseline: grid-based Local Binary Patterns (LBP).

Why LBP instead of a pretrained deep embedding model (FaceNet/ArcFace/dlib)?
See docs/CV_MODEL_EVALUATION.md and docs/LICENSE_AND_PROVENANCE.md for the
full investigation. Short version:
  - LBP is a fixed mathematical transform, not a model trained on a dataset.
    There is no training-data provenance/licence question at all.
  - It runs on CPU with no external weight download -- fully reproducible
    in an offline/air-gapped dev environment.
  - It is fully interpretable: every number in the vector is a texture
    histogram bin, not an opaque learned feature.
  - It is a legitimate, published, widely used baseline for constrained
    face-recognition systems (Ahonen et al., 2006).

Trade-off (documented, not hidden): LBP is less accurate than a modern deep
embedding on hard cases (large pose/illumination variation, large galleries).
`TorchEmbedder` below is a stub showing exactly where a PyTorch-based
embedding model would plug in without touching any other pipeline stage.
"""
from abc import ABC, abstractmethod

import numpy as np
from skimage.feature import local_binary_pattern

from .config import RepresentationConfig
from .exceptions import ModelNotLoadedError


class Embedder(ABC):
    """Interface every representation backend must implement."""

    model_version: str

    @abstractmethod
    def embed(self, face_gray: np.ndarray) -> np.ndarray:
        """face_gray: uint8 2D array. Returns a 1D float embedding vector."""


class LBPGridEmbedder(Embedder):
    """Reference implementation used by this baseline."""

    def __init__(self, config: RepresentationConfig = None):
        self.config = config or RepresentationConfig()
        self.model_version = self.config.model_version
        # uniform LBP with P points has P+2 distinct pattern bins
        self._n_bins = self.config.lbp_points + 2

    def embed(self, face_gray: np.ndarray) -> np.ndarray:
        if face_gray is None or face_gray.size == 0:
            raise ValueError("Empty face crop passed to embedder.")

        lbp = local_binary_pattern(
            face_gray,
            P=self.config.lbp_points,
            R=self.config.lbp_radius,
            method=self.config.lbp_method,
        )

        gh, gw = self.config.grid_cells
        h, w = lbp.shape
        cell_h, cell_w = h // gh, w // gw

        histograms = []
        for i in range(gh):
            for j in range(gw):
                cell = lbp[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                hist, _ = np.histogram(
                    cell.ravel(), bins=self._n_bins, range=(0, self._n_bins)
                )
                hist = hist.astype(np.float64)
                hist /= hist.sum() + 1e-8  # normalise per-cell (illumination robustness)
                histograms.append(hist)

        return np.concatenate(histograms)  # shape: (gh*gw*n_bins,)

    @property
    def embedding_dim(self) -> int:
        gh, gw = self.config.grid_cells
        return gh * gw * self._n_bins


class TorchEmbedder(Embedder):
    """
    Placeholder for a future PyTorch-based deep embedding backend
    (e.g. facenet-pytorch InceptionResnetV1, or ArcFace via insightface).

    Not wired to real weights in this environment (no network access to
    fetch pretrained weights during this sprint -- see
    docs/LICENSE_AND_PROVENANCE.md, "Environment constraints"). Implement
    `embed()` to satisfy the same interface as LBPGridEmbedder and nothing
    else in the pipeline needs to change.
    """

    model_version = "torch-backend-not-configured"

    def __init__(self, weights_path: str = None):
        self.weights_path = weights_path

    def embed(self, face_gray: np.ndarray) -> np.ndarray:
        raise ModelNotLoadedError(
            "TorchEmbedder has no weights configured. This is a documented "
            "extension point, not an active backend, in this sprint."
        )
