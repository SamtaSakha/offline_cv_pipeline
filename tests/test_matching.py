import unittest

import numpy as np

from bhiv_cv.matching import match, MatchingConfig
from bhiv_cv.gallery import Gallery
from bhiv_cv.exceptions import EmptyGalleryError, DuplicateIdentityError


class TestMatching(unittest.TestCase):
    def test_empty_gallery_raises(self):
        with self.assertRaises(EmptyGalleryError):
            match(np.zeros(10), {})

    def test_identical_embedding_matches_with_high_similarity(self):
        template = np.array([0.5, 0.5, 0.0, 0.0])
        gallery = {"alice": [template]}
        result = match(template.copy(), gallery)
        self.assertTrue(result.match)
        self.assertEqual(result.identity, "alice")
        self.assertGreater(result.similarity, 0.99)

    def test_very_different_embedding_is_unknown(self):
        template = np.array([1.0, 0.0, 0.0, 0.0])
        probe = np.array([0.0, 0.0, 0.0, 1.0])
        gallery = {"alice": [template]}
        result = match(probe, gallery, MatchingConfig(similarity_threshold=0.55))
        self.assertFalse(result.match)
        self.assertIsNone(result.identity)

    def test_close_but_distinct_identities_reduce_confidence(self):
        """Two near-identical gallery templates for DIFFERENT identities
        should suppress the margin-based confidence even when similarity is
        high -- the 'similar-looking person' scenario from Day 4."""
        probe = np.array([0.5, 0.495, 0.0, 0.005])
        gallery = {
            "alice": [np.array([0.5, 0.5, 0.0, 0.0])],
            "alice_lookalike": [np.array([0.5, 0.49, 0.0, 0.01])],
        }
        result = match(probe, gallery, MatchingConfig(min_confidence=0.9))
        self.assertLess(result.confidence, 0.9)
        self.assertFalse(result.match)

    def test_gallery_rejects_duplicate_identity_by_default(self):
        g = Gallery()
        g.enroll("alice", np.zeros(4))
        with self.assertRaises(DuplicateIdentityError):
            g.enroll("alice", np.ones(4))

    def test_gallery_allows_update_when_explicitly_requested(self):
        g = Gallery()
        g.enroll("alice", np.zeros(4))
        g.enroll("alice", np.ones(4), allow_update=True)
        self.assertEqual(len(g.as_dict()["alice"]), 2)

    def test_gallery_save_and_load_roundtrip(self):
        import tempfile, os
        g = Gallery()
        g.enroll("alice", np.array([0.1, 0.2, 0.3]))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gallery.json")
            g.save(path)
            loaded = Gallery.load(path)
        self.assertEqual(loaded.identities(), ["alice"])
        np.testing.assert_allclose(loaded.as_dict()["alice"][0], [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
