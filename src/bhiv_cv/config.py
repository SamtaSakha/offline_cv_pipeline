"""
Central, explicit configuration for the pipeline.

Nothing about any real identity, hostel, or organisation lives in code.
Every tunable in this file is one that a real deployment would sweep during
benchmarking (see docs/BENCHMARK.md).
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectionConfig:
    cascade_name: str = "haarcascade_frontalface_default.xml"
    scale_factor: float = 1.08
    min_neighbors: int = 5
    min_size: tuple = (40, 40)
    max_faces_returned: int = 10          # safety ceiling, not a business rule
    max_image_megapixels: float = 25.0    # ResourceLimitError guard


@dataclass(frozen=True)
class PreprocessConfig:
    target_size: tuple = (200, 200)
    face_margin: float = 0.25             # extra context around detected box
    equalize_histogram: bool = True       # basic illumination normalisation


@dataclass(frozen=True)
class RepresentationConfig:
    backend: str = "lbp_grid"             # "lbp_grid" | "torch" (future)
    lbp_radius: int = 1
    lbp_points: int = 8
    lbp_method: str = "uniform"
    grid_cells: tuple = (8, 8)            # spatial grid over the 200x200 face
    model_version: str = "cv-baseline-lbp-v1"


@dataclass(frozen=True)
class MatchingConfig:
    distance_metric: str = "chi_square"   # standard for LBP histograms
    similarity_threshold: float = 0.55    # below -> reported as "unknown"
    min_confidence: float = 0.35          # margin-based distinctiveness floor
    distance_scale: float = 0.35          # tunes distance->similarity decay


@dataclass(frozen=True)
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)


DEFAULT_CONFIG = PipelineConfig()
