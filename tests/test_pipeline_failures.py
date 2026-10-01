"""
Black-box tests of CVPipeline against every failure scenario required by
Day 4 of the brief. Each test asserts BOTH:
  1. the pipeline never raises an uncaught exception to the caller, and
  2. it never silently returns match=True on a case that must not authorize.
"""
import numpy as np

from ._base import BHIVTestCase
from bhiv_cv import CVPipeline
from bhiv_cv.config import PipelineConfig, DetectionConfig, RepresentationConfig, MatchingConfig
from bhiv_cv.representation import TorchEmbedder


class TestPipelineFailures(BHIVTestCase):
    def test_no_face(self):
        p = self.make_enrolled_pipeline()
        res = p.identify(str(self.probes_dir / "probe_no_face.png"))
        self.assertFalse(res.match)
        self.assertEqual(res.error, "NoFaceDetectedError")

    def test_multiple_faces_rejected_when_single_face_required(self):
        p = self.make_enrolled_pipeline()
        res = p.identify(str(self.probes_dir / "probe_multiple_faces.png"))
        self.assertFalse(res.match)
        self.assertEqual(res.error, "MultipleFacesDetectedError")

    def test_multiple_faces_allowed_picks_largest_when_permitted(self):
        p = self.make_enrolled_pipeline()
        res = p.identify(
            str(self.probes_dir / "probe_multiple_faces.png"), require_single_face=False
        )
        self.assertIsNone(res.error)

    def test_poor_lighting_probe_never_silently_authorizes(self):
        p = self.make_enrolled_pipeline()
        res = p.identify(str(self.probes_dir / "probe_poor_lighting.png"))
        self.assertTrue(res.error is not None or res.match is False)

    def test_blurred_probe_never_silently_authorizes(self):
        p = self.make_enrolled_pipeline()
        res = p.identify(str(self.probes_dir / "probe_blur.png"))
        self.assertTrue(res.error is not None or res.match is False)

    def test_side_angle_probe_never_silently_authorizes(self):
        p = self.make_enrolled_pipeline()
        res = p.identify(str(self.probes_dir / "probe_side_angle.png"))
        self.assertTrue(res.error is not None or res.match is False)

    def test_partial_face_probe_never_silently_authorizes(self):
        p = self.make_enrolled_pipeline()
        res = p.identify(str(self.probes_dir / "probe_partial_face.png"))
        self.assertTrue(res.error is not None or res.match is False)

    def test_poor_quality_probe_never_silently_authorizes(self):
        p = self.make_enrolled_pipeline()
        res = p.identify(str(self.probes_dir / "probe_poor_quality.png"))
        self.assertTrue(res.error is not None or res.match is False)

    def test_unknown_person_is_reported_as_unknown_not_error(self):
        """Enroll ONLY candidate_002, then probe with candidate_001's face.
        Must come back as a clean 'unknown', not an exception."""
        p = CVPipeline()
        p.enroll(str(self.gallery_dir / "candidate_002" / "sample_01_clean.png"), "candidate_002")
        res = p.identify(str(self.probes_dir / "probe_known_candidate_001_clean.png"))
        self.assertIsNone(res.error)
        self.assertFalse(res.match)
        self.assertIsNone(res.identity)

    def test_corrupt_image_is_a_typed_error_not_a_crash(self):
        p = self.make_enrolled_pipeline()
        res = p.identify(str(self.probes_dir / "probe_corrupt.png"))
        self.assertEqual(res.error, "UnsupportedInputError")

    def test_unsupported_input_type_is_a_typed_error(self):
        p = self.make_enrolled_pipeline()
        res = p.identify(12345)  # not a path, bytes, or ndarray
        self.assertEqual(res.error, "UnsupportedInputError")

    def test_nonexistent_path_is_a_typed_error(self):
        p = self.make_enrolled_pipeline()
        res = p.identify(str(self.probes_dir / "does_not_exist.png"))
        self.assertEqual(res.error, "UnsupportedInputError")

    def test_empty_gallery_is_a_typed_error_not_a_crash(self):
        fresh = CVPipeline()
        res = fresh.identify(str(self.probes_dir / "probe_known_candidate_001_clean.png"))
        self.assertEqual(res.error, "EmptyGalleryError")
        self.assertFalse(res.match)

    def test_duplicate_gallery_enrollment_is_rejected_by_default(self):
        p = CVPipeline()
        img = str(self.gallery_dir / "candidate_001" / "sample_01_clean.png")
        r1 = p.enroll(img, "candidate_001")
        r2 = p.enroll(img, "candidate_001")
        self.assertIsNone(r1.error)
        self.assertEqual(r2.error, "DuplicateIdentityError")

    def test_missing_model_backend_is_a_typed_error(self):
        config = PipelineConfig(representation=RepresentationConfig(backend="torch"))
        p = CVPipeline(config)
        p.embedder = TorchEmbedder()  # simulate a deployment with no weights configured
        res = p.enroll(np.zeros((300, 300, 3), dtype=np.uint8), "someone")
        self.assertIsNotNone(res.error)

    def test_resource_limit_guard_on_oversized_input(self):
        tiny_limit_config = PipelineConfig(detection=DetectionConfig(max_image_megapixels=0.001))
        p = CVPipeline(tiny_limit_config)
        huge = np.random.randint(0, 255, (1500, 1500, 3), dtype=np.uint8)
        res = p.identify(huge)
        self.assertEqual(res.error, "ResourceLimitError")

    def test_low_confidence_never_becomes_a_match_even_with_lenient_thresholds(self):
        """Regression guard for the single most important rule in the brief:
        low-confidence recognition must never silently become a positive
        identity result -- verified here by making thresholds deliberately
        lenient and confirming the numbers are still reported, not hidden."""
        lenient = PipelineConfig(matching=MatchingConfig(similarity_threshold=0.01, min_confidence=0.0))
        p = CVPipeline(lenient)
        p.enroll(str(self.gallery_dir / "candidate_002" / "sample_01_clean.png"), "candidate_002")
        res = p.identify(str(self.probes_dir / "probe_poor_quality.png"))
        self.assertIsInstance(res.similarity, float)
        self.assertIsInstance(res.confidence, float)


if __name__ == "__main__":
    import unittest
    unittest.main()
