import numpy as np

from ._base import BHIVTestCase
from bhiv_cv.detection import FaceDetector
from bhiv_cv.config import DetectionConfig
from bhiv_cv.pipeline import load_image
from bhiv_cv.exceptions import ResourceLimitError


class TestDetection(BHIVTestCase):
    def test_detects_face_in_clean_probe(self):
        detector = FaceDetector()
        img = load_image(str(self.probes_dir / "probe_known_candidate_001_clean.png"))
        boxes = detector.detect(img)
        self.assertGreaterEqual(len(boxes), 1)
        top = boxes[0]
        self.assertGreater(top.w, 30)
        self.assertGreater(top.h, 30)

    def test_no_face_on_blank_frame(self):
        detector = FaceDetector()
        img = load_image(str(self.probes_dir / "probe_no_face.png"))
        boxes = detector.detect(img)
        self.assertEqual(len(boxes), 0)

    def test_detects_two_faces_in_multi_face_frame(self):
        detector = FaceDetector()
        img = load_image(str(self.probes_dir / "probe_multiple_faces.png"))
        boxes = detector.detect(img)
        self.assertGreaterEqual(len(boxes), 2)

    def test_resource_limit_error_on_oversized_image(self):
        detector = FaceDetector(DetectionConfig(max_image_megapixels=0.01))
        huge = np.zeros((2000, 2000, 3), dtype=np.uint8)
        with self.assertRaises(ResourceLimitError):
            detector.detect(huge)

    def test_detector_scores_are_bounded_0_1(self):
        detector = FaceDetector()
        img = load_image(str(self.probes_dir / "probe_known_candidate_001_clean.png"))
        boxes = detector.detect(img)
        for b in boxes:
            self.assertGreaterEqual(b.detector_score, 0.0)
            self.assertLessEqual(b.detector_score, 1.0)
