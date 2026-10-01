# CV_MODEL_EVALUATION.md

## Candidates investigated

| Candidate | Type | Architecture | CPU-suitable? | Weight download required? |
|---|---|---|---|---|
| **FaceNet** (`facenet-pytorch`, InceptionResnetV1, `vggface2`/`casia-webface` pretrained) | Deep embedding | Inception-ResNet, 512-d embedding | Yes, but slower cold-start (~30–90MB weights, seconds to load) | Yes — pulled from a hosted URL at first run |
| **ArcFace** (via `insightface`) | Deep embedding | ResNet backbone + additive angular margin loss | Yes | Yes — pulled from model zoo |
| **dlib `face_recognition`** | Deep embedding | Small ResNet, 128-d embedding | Yes | Yes — bundled `.dat` file (~100MB) downloaded separately |
| **OpenCV Haar Cascade + LBPH / classical LBP** | Classical CV (not a trained model) | Handcrafted Haar features (detection) + Local Binary Pattern texture histograms (representation) | Yes, very lightweight | **No — ships inside `opencv-python`/`scikit-image`** |

## Environment constraint that shaped the decision

This sprint's development container has **no outbound network access**
(confirmed: `pip install`, weight downloads, and `curl` to PyPI all fail —
see `evidence/test_results_raw.txt` header and the container's egress
policy). That immediately rules out FaceNet, ArcFace, and dlib as *runnable,
reproducible-today* choices, since all three require pulling pretrained
weights from the internet on first run.

This is treated as a real engineering constraint, not an excuse: a baseline
that cannot be rebuilt from clean instructions in the target environment
fails Day 7's "rebuild from clean instructions" requirement outright,
regardless of its accuracy on paper.

## Selected baseline: OpenCV Haar Cascade (detection) + grid-based LBP (representation)

**What it is:** Haar Cascade is a classical sliding-window detector using
handcrafted rectangular features (Viola–Jones, 2001), bundled as XML files
inside `opencv-python`. Local Binary Patterns (Ahonen et al., 2006) is a
classical texture descriptor: each pixel is compared to its neighbors to
produce a binary pattern, and a histogram of patterns over a spatial grid
becomes the face's "embedding." Neither has trained weights in the deep
-learning sense — they are fixed, published mathematical procedures.

**Input/output contract:** `detect(image) -> [BoundingBox]`;
`embed(face_crop) -> 640-d float vector` (8×8 grid × 10 LBP bins, see
`evidence/benchmark_results.json → per_identity_storage_cost`).

**Why this is the right call for this sprint, not just the available one:**
1. **Zero training-dataset provenance question.** There is no dataset behind
   the "model" to investigate licensing for — see `LICENSE_AND_PROVENANCE.md`.
   This directly avoids the exact trap the brief calls out ("it's on GitHub,
   so it's free").
2. **Fully reproducible offline**, satisfying Day 7's rebuild-from-clean
   requirement in this exact environment.
3. **CPU-only, small, fast per-identity storage** — 640 floats
   (5.0 KB/1000 identities, see benchmark evidence) vs. hundreds of MB of
   deep-model weights.
4. **Interpretable** — every number in the embedding is a texture-histogram
   bin, not an opaque learned feature, which makes failure analysis (Day 4)
   tractable rather than a black box.

**Honest trade-off (not hidden):** LBP is measurably less discriminative
than a modern deep embedding, especially under large pose/illumination
variation or with large galleries (hundreds+ of identities). This is visible
directly in this sprint's own failure evidence: the "poor lighting" and
"blurred" probes in `FAILURE_TESTS.md` produce weaker/absent detections,
where a deep detector (e.g. an MTCNN/RetinaFace-class model) would likely be
more robust.

## Upgrade path (documented, not built this sprint)

`src/bhiv_cv/representation.py` defines an `Embedder` abstract interface.
`LBPGridEmbedder` implements it today; `TorchEmbedder` is a stub showing
exactly where a PyTorch-based deep embedding model plugs in later —
swapping backends requires no change to detection, preprocessing, matching,
gallery, or the integration contract. When network/GPU access is available:
1. Add `facenet-pytorch` (MIT licensed) or a similarly permissively licensed
   model to `requirements.txt`.
2. Implement `TorchEmbedder.embed()` to run the model and return a vector.
3. Re-run `tests/` and `scripts/benchmark.py` unchanged — they test the
   interface, not the backend.
4. Re-do the licence/provenance check in `LICENSE_AND_PROVENANCE.md` for the
   new model's specific weight license AND its training dataset's license —
   these are two separate questions (see that doc).

## What BHIV can legally learn from vs. use vs. must independently implement

| | Learn from (read/study) | Use as-is (ship in a product) | Must independently implement |
|---|---|---|---|
| Haar Cascade | ✅ published algorithm | ✅ Apache-2.0 via OpenCV, no restriction | — |
| LBP | ✅ published algorithm (2006 paper) | ✅ it's math, not code someone else owns | — |
| FaceNet/ArcFace/dlib *models* (if adopted later) | ✅ architecture/paper | ⚠️ only after checking THIS SPECIFIC weight file's license (varies by release) | Training data pipeline, if BHIV wants a model with a fully known/owned provenance chain |
