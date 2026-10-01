"""
Exception hierarchy for the BHIV CV pipeline.

Design rule (see docs/ARCHITECTURE.md):
    The CV module NEVER decides access. It only ever produces one of:
      - a successful IdentityResult (possibly "unknown" / low confidence)
      - a raised, typed error describing why it could not produce a result.
    It is the caller's (HIAS's) job to decide what "no result" or
    "low confidence" means for access control.
"""


class CVPipelineError(Exception):
    """Base class for all pipeline errors. Never raised directly."""


class UnsupportedInputError(CVPipelineError):
    """Input is not a decodable image (wrong type, corrupt bytes, unsupported format)."""


class NoFaceDetectedError(CVPipelineError):
    """Detector ran successfully but found zero faces in the frame."""


class MultipleFacesDetectedError(CVPipelineError):
    """More than one face found where the caller required exactly one (e.g. enrollment)."""

    def __init__(self, count: int, boxes):
        super().__init__(f"Expected exactly 1 face, found {count}.")
        self.count = count
        self.boxes = boxes


class EmptyGalleryError(CVPipelineError):
    """Matching was requested but zero identities are enrolled."""


class ModelNotLoadedError(CVPipelineError):
    """The representation model/backend has not been initialised."""


class DuplicateIdentityError(CVPipelineError):
    """Attempted to enroll an identity name that already exists in the gallery."""

    def __init__(self, identity: str):
        super().__init__(f"Identity '{identity}' is already enrolled.")
        self.identity = identity


class ResourceLimitError(CVPipelineError):
    """Input exceeded a configured safety ceiling (image size, memory guard, etc)."""
