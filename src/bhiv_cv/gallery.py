"""
Enrollment gallery: in-memory + JSON-file persisted store of
identity -> [embedding templates].

No real identities are hard-coded anywhere in this repo. The gallery is
populated only by calling `enroll()` at runtime, normally from
scripts/enroll.py against caller-supplied images.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .exceptions import DuplicateIdentityError


class Gallery:
    def __init__(self):
        self._store: Dict[str, List[np.ndarray]] = {}

    def enroll(self, identity: str, embedding: np.ndarray, allow_update: bool = False):
        if identity in self._store and not allow_update:
            raise DuplicateIdentityError(identity)
        self._store.setdefault(identity, []).append(embedding)

    def remove(self, identity: str):
        self._store.pop(identity, None)

    def is_empty(self) -> bool:
        return len(self._store) == 0

    def identities(self) -> List[str]:
        return list(self._store.keys())

    def as_dict(self) -> Dict[str, List[np.ndarray]]:
        return self._store

    def save(self, path: str):
        serializable = {
            identity: [emb.tolist() for emb in embeddings]
            for identity, embeddings in self._store.items()
        }
        Path(path).write_text(json.dumps(serializable))

    @classmethod
    def load(cls, path: str) -> "Gallery":
        g = cls()
        raw = json.loads(Path(path).read_text())
        for identity, embeddings in raw.items():
            g._store[identity] = [np.array(e, dtype=np.float64) for e in embeddings]
        return g
