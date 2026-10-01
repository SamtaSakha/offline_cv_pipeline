"""
Shared test scaffolding using the Python standard-library `unittest`
framework (chosen deliberately: this environment has no network access to
install pytest, and stdlib unittest gives identical guarantees -- isolated
test functions, fixtures via setUp, exception assertions -- with zero
external dependency. See docs/REVIEW_PACKET.md, "environment constraints").
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bhiv_cv import CVPipeline  # noqa: E402

_DATA_PREPARED = False


def ensure_sample_data():
    global _DATA_PREPARED
    if not _DATA_PREPARED:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "prepare_sample_data.py")],
            check=True,
            cwd=ROOT,
            capture_output=True,
        )
        _DATA_PREPARED = True


class BHIVTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_sample_data()
        cls.data_dir = ROOT / "data"
        cls.gallery_dir = cls.data_dir / "gallery"
        cls.probes_dir = cls.data_dir / "probes"

    def make_enrolled_pipeline(self) -> CVPipeline:
        p = CVPipeline()
        p.enroll(str(self.gallery_dir / "candidate_001" / "sample_01_clean.png"), "candidate_001")
        p.enroll(str(self.gallery_dir / "candidate_002" / "sample_01_clean.png"), "candidate_002")
        return p
