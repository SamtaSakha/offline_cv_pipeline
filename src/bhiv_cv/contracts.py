"""
The integration contract: the ONLY thing that crosses the boundary from the
CV module into HIAS. See docs/INTEGRATION.md for the full boundary
discussion.

Hard rule enforced by this schema: there is no "access", "authorized",
"door", "unlock" or similar field anywhere in this object. The CV module
structurally cannot express an access decision, because the contract has no
field to put one in.
"""
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class IdentityResult:
    identity: Optional[str]     # enrolled identity string, or None ("unknown")
    match: bool                 # True only if similarity+confidence cleared threshold
    similarity: float           # 0..1
    confidence: float           # 0..1, margin-based distinctiveness
    model_version: str
    trace_id: str
    timestamp: float
    error: Optional[str] = None  # populated instead of the above on failure

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def new_trace_id() -> str:
        return str(uuid.uuid4())

    @classmethod
    def success(cls, identity, match, similarity, confidence, model_version):
        return cls(
            identity=identity,
            match=match,
            similarity=round(similarity, 4),
            confidence=round(confidence, 4),
            model_version=model_version,
            trace_id=cls.new_trace_id(),
            timestamp=time.time(),
        )

    @classmethod
    def failure(cls, error_type: str, model_version: str):
        return cls(
            identity=None,
            match=False,
            similarity=0.0,
            confidence=0.0,
            model_version=model_version,
            trace_id=cls.new_trace_id(),
            timestamp=time.time(),
            error=error_type,
        )
