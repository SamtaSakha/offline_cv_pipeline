import numpy as np

from ._base import BHIVTestCase
from bhiv_cv.detection import FaceDetector, BoundingBox
from bhiv_cv.preprocessing import crop_and_normalize
from bhiv_cv.config import PreprocessConfig
from bhiv_cv.pipeline import load_image


class TestPreprocessing(BHIVTestCase):
    def test_output_shape_matches_config(self):
        detector = FaceDetector()
        img = load_image(str(self.probes_dir / "probe_known_candidate_001_clean.png"))
        box = detector.detect(img)[0]

        config = PreprocessConfig(target_size=(128, 128))
        face = crop_and_normalize(img, box, config)

        self.assertEqual(face.shape, (128, 128))
        self.assertEqual(face.dtype.name, "uint8")

    def test_default_target_size(self):
        detector = FaceDetector()
        img = load_image(str(self.probes_dir / "probe_known_candidate_001_clean.png"))
        box = detector.detect(img)[0]
        face = crop_and_normalize(img, box)
        self.assertEqual(face.shape, (200, 200))

    def test_crop_stays_within_image_bounds_near_edges(self):
        """A box whose margin would extend outside the frame must be
        clipped, not throw. Classic off-by-one edge-crop failure mode."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        edge_box = BoundingBox(x=0, y=0, w=40, h=40, detector_score=0.9)
        face = crop_and_normalize(img, edge_box)
        self.assertEqual(face.shape, (200, 200))
