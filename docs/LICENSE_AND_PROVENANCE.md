# LICENSE_AND_PROVENANCE.md

No hand-waving rule applied throughout: "it's on GitHub" / "it's on PyPI" is
never treated as equivalent to "free to use commercially." Each dependency
below is checked individually for (a) code license, (b) any bundled model
license, and (c) any bundled dataset/sample-image provenance.

## 1. Software dependencies

| Package | License | Commercial use? | Modification/redistribution? |
|---|---|---|---|
| `opencv-python` / `opencv-contrib-python` | Apache 2.0 | Yes | Yes, with attribution retained in NOTICE |
| `numpy` | BSD-3-Clause | Yes | Yes |
| `scikit-image` | BSD-3-Clause | Yes | Yes |
| `psutil` | BSD-3-Clause | Yes | Yes |
| Haar cascade XML files (bundled in `opencv-python`) | Intel/OpenCV contribution, distributed under OpenCV's Apache-2.0 | Yes | Yes |

No copyleft (GPL/AGPL) dependencies are used anywhere in this pipeline.
None of the above require attribution beyond retaining the standard
license file, and none restrict commercial use.

## 2. "Model" provenance — the actual answer for this baseline

The representation backend (`LBPGridEmbedder`) is **not a trained model**.
Local Binary Patterns is a fixed, published mathematical transform (Ahonen,
Hadid & Pietikäinen, 2006) with no training data behind it at all. There is
therefore **no dataset license, no training-data consent question, and no
model-weight license** to evaluate for the representation stage — this is
the single strongest provenance property of this baseline and is the main
reason it was selected for a sprint that explicitly calls out sloppy
provenance handling as a repeat problem.

Haar Cascade detection is similarly not "trained" in the deep-learning
sense per se (it is boosted from labeled examples originally, but the
resulting cascade is a fixed, published, redistributable artifact shipped
under OpenCV's own license — not a third-party dataset the pipeline
depends on at runtime).

## 3. Sample/demo images used in this repo (`data/gallery`, `data/probes`)

These images exist **only to make the pipeline runnable and testable in
this sprint**. They are not, and must never become, real enrolled
identities in any production HIAS deployment.

| Image | Source | Status |
|---|---|---|
| "astronaut" (`skimage.data.astronaut`) | Official NASA portrait of astronaut Eileen Collins, bundled offline with `scikit-image` | US federal government work — public domain. Standard, decades-old CV research/teaching benchmark image. |
| "grace_hopper" (`matplotlib` sample data) | Official US Navy portrait of Grace Hopper, bundled offline with `matplotlib` | US federal government work — public domain. Also used by the official PyTorch tutorials as a standard demo image. |

**Why these and not a real face dataset (e.g. LFW, CASIA-WebFace):** this
sprint's environment has no network access to download a dataset, and
academic face datasets carry their own non-trivial licensing/consent
questions (LFW, for example, restricts use to non-commercial research and
was built from web-scraped news photos without explicit subject consent —
exactly the kind of provenance risk this document exists to catch). Using
two public-domain, government-work photographs sidesteps that risk entirely
while still exercising the pipeline against real human faces.

**Explicit non-goal:** this repo does not ship, and must never be extended
to ship, a scraped or academic face dataset without a fresh, standalone
license review before that dataset touches any BHIV system.

## 4. Commercial-use / redistribution summary for what THIS sprint ships

Everything in this repository (code, the two demo images, the generated
synthetic gallery/probe variants) can be used commercially and redistributed
inside BHIV without further legal review, **provided**:
- The demo images (`data/gallery`, `data/probes`) are treated as
  throwaway test fixtures, never as production enrollment data.
- Any future swap to a deep-learning `TorchEmbedder` backend (see
  `CV_MODEL_EVALUATION.md`) triggers a fresh review of that specific model's
  weight license AND its training dataset's license — both, not just one.
