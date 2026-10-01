"""
Stage 4: Matching.

Responsibility: compare one embedding against a gallery of enrolled
embeddings and produce a ranked result with similarity + confidence.

This module DOES NOT decide "known" vs "unknown" for access purposes -- it
applies a configured statistical threshold and reports the numbers. What an
integrator does with those numbers (see docs/INTEGRATION.md) is outside this
module's authority.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import MatchingConfig
from .exceptions import EmptyGalleryError


@dataclass(frozen=True)
class MatchCandidate:
    identity: str
    distance: float
    similarity: float


@dataclass(frozen=True)
class MatchResult:
    identity: Optional[str]      # None if below threshold ("unknown")
    match: bool                  # True only if identity is not None
    similarity: float            # top-1 similarity, 0..1
    confidence: float            # margin-based distinctiveness, 0..1
    top_k: List[MatchCandidate]  # full ranked list, for audit/debugging


def _chi_square_distance(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1e-10
    return float(0.5 * np.sum(((a - b) ** 2) / (a + b + eps)))


def match(
    embedding: np.ndarray,
    gallery: dict,  # identity -> List[np.ndarray] (one or more templates)
    config: MatchingConfig = None,
) -> MatchResult:
    config = config or MatchingConfig()

    if not gallery:
        raise EmptyGalleryError("Matching requested against an empty gallery.")

    candidates: List[MatchCandidate] = []
    for identity, templates in gallery.items():
        best_distance = min(_chi_square_distance(embedding, t) for t in templates)
        similarity = float(np.exp(-best_distance / config.distance_scale))
        candidates.append(MatchCandidate(identity, best_distance, similarity))

    candidates.sort(key=lambda c: c.distance)
    top1 = candidates[0]

    if len(candidates) > 1:
        top2 = candidates[1]
        # margin-based confidence: how much better is top1 than the runner-up,
        # relative to top2's distance. Large gap => high confidence.
        confidence = float(
            max(0.0, min(1.0, (top2.distance - top1.distance) / (top2.distance + 1e-8)))
        )
    else:
        # Single-identity gallery: fall back to absolute similarity as confidence.
        confidence = top1.similarity

    is_match = (
        top1.similarity >= config.similarity_threshold
        and confidence >= config.min_confidence
    )

    return MatchResult(
        identity=top1.identity if is_match else None,
        match=is_match,
        similarity=top1.similarity,
        confidence=confidence,
        top_k=candidates[:5],
    )
