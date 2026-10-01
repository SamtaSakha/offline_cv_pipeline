import unittest

import numpy as np

from bhiv_cv.representation import LBPGridEmbedder, TorchEmbedder
from bhiv_cv.exceptions import ModelNotLoadedError


class TestRepresentation(unittest.TestCase):
    def test_embedding_is_deterministic(self):
        embedder = LBPGridEmbedder()
        face = np.random.RandomState(0).randint(0, 255, (200, 200), dtype=np.uint8)
        v1 = embedder.embed(face)
        v2 = embedder.embed(face)
        np.testing.assert_array_equal(v1, v2)

    def test_embedding_has_expected_dimension(self):
        embedder = LBPGridEmbedder()
        face = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        v = embedder.embed(face)
        self.assertEqual(v.shape[0], embedder.embedding_dim)

    def test_different_faces_give_different_embeddings(self):
        embedder = LBPGridEmbedder()
        face_a = np.random.RandomState(1).randint(0, 255, (200, 200), dtype=np.uint8)
        face_b = np.random.RandomState(2).randint(0, 255, (200, 200), dtype=np.uint8)
        v_a = embedder.embed(face_a)
        v_b = embedder.embed(face_b)
        self.assertFalse(np.allclose(v_a, v_b))

    def test_embedding_rejects_empty_input(self):
        embedder = LBPGridEmbedder()
        with self.assertRaises(ValueError):
            embedder.embed(np.array([]))

    def test_torch_embedder_stub_raises_model_not_loaded(self):
        embedder = TorchEmbedder()
        with self.assertRaises(ModelNotLoadedError):
            embedder.embed(np.zeros((200, 200), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
