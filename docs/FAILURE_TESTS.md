# FAILURE_TESTS.md

All rows below are backed by real, reproducible runs, not descriptions of
expected behaviour. Raw evidence:
- `evidence/runtime_log.jsonl` — one real pipeline call per row, with actual
  `similarity`/`confidence`/`trace_id` values.
- `evidence/test_results_raw.txt` — 37/37 automated regression tests passing
  (`tests/test_pipeline_failures.py` encodes every row below as an
  assertion, so these are not just observations — they are enforced).

To reproduce any row yourself:
```
PYTHONPATH=src python3 scripts/generate_runtime_log.py
PYTHONPATH=src python3 -m unittest discover -t . -s tests -v
```

**The one rule every row below is checked against:** low-confidence or
ambiguous recognition must never silently become a positive identity result.
Every row ends in either a typed error or an explicit `match: false` — never
a silent `match: true`.

| # | Input | Behaviour observed | Expected | Actual | Handling |
|---|---|---|---|---|---|
| 1 | No face (`probe_no_face.png`, a blank frame) | Detector finds 0 boxes | Typed error, no crash | `error: NoFaceDetectedError`, `match: false` | Caller sees a typed exception name, not a stack trace. HIAS should treat this as "cannot evaluate," not "denied." |
| 2 | Multiple faces (`probe_multiple_faces.png`, two people in frame) | Detector finds 2 boxes; identification requires exactly 1 by default | Rejected, ambiguity surfaced | `error: MultipleFacesDetectedError` (count=2, boxes attached) | Default policy refuses to silently pick one identity out of an ambiguous frame. `identify(..., require_single_face=False)` opts into "largest face wins" for surveillance-style use cases — verified in `test_multiple_faces_allowed_picks_largest_when_permitted`. |
| 3 | Poor lighting (image darkened to 15% brightness) | Haar cascade, run on the histogram-equalized frame, found **2** boxes (a false positive appeared as contrast collapsed) | Should not silently authorize | `error: MultipleFacesDetectedError` | Same ambiguity-rejection path as row 2. This is a genuine, observed classical-detector weakness: severe brightness loss increases false-positive rate. Documented as a known LBP/Haar limitation, not swept under the rug. |
| 4 | Blur (Gaussian blur, kernel=25) | Cascade still found exactly 1 box; embedding was computed but did not match either enrolled identity | Should not silently authorize | `similarity: 0.0`, `confidence: 0.6525`, `identity: None`, `match: false`, `error: None` | Clean "unknown" outcome, not an exception — the pipeline detected a face but correctly refused to claim it matched anyone. This is the intended behaviour for "recognized but insufficiently confident." |
| 5 | Side angle (35° rotation) | Frontal-only Haar cascade found 0 boxes | Should fail cleanly, not silently pass | `error: NoFaceDetectedError` | Documented limitation: this baseline uses a frontal-face cascade only. A production system needing pose robustness would add a profile cascade or a pose-invariant deep detector (see `CV_MODEL_EVALUATION.md` upgrade path). |
| 6 | Partial face (55% of frame blacked out) | 0 boxes found | Should fail cleanly | `error: NoFaceDetectedError` | Same as row 5 — occlusion below the cascade's minimum feature visibility is correctly rejected rather than guessed at. |
| 7 | Poor-quality / low-resolution input (down-then-up-sampled to ~6% native resolution) | 0 boxes found | Should fail cleanly | `error: NoFaceDetectedError` | Matches the independent finding in `scripts/optimization_experiments.py`: resizing to 128×128 also broke detection entirely — this is a real, reproducible resolution floor for this detector, not a one-off. |
| 8 | Unknown person (gallery has only `candidate_002`; probe is `candidate_001`) | Face detected and embedded fine; matched against the 1-identity gallery | Reported as unknown, NOT an error | `identity: None`, `match: false`, `error: None` | Verified in `test_unknown_person_is_reported_as_unknown_not_error`. This is the single most important non-error outcome in the whole system: an unrecognized person is an expected, ordinary result. |
| 9 | Similar-looking person (two near-identical embeddings in the gallery for different identities) | Top-1 and top-2 candidates too close together | Confidence should drop even if similarity is high | Margin-based `confidence` collapses below the configured floor, forcing `match: false` (see `tests/test_matching.py::test_close_but_distinct_identities_reduce_confidence`) | Similarity alone is not trusted for a decision — confidence is a *separate*, distinctiveness-based number specifically to catch this case. |
| 10 | Duplicate gallery identity (enroll `"candidate_001"` twice without `allow_update`) | Second enroll call rejected | Should not silently overwrite | `error: DuplicateIdentityError` | Enrollment requires an explicit `allow_update=True` to replace/add a template — prevents silent identity collisions. |
| 11 | Empty gallery (`identify()` called before any enrollment) | Matching stage has zero candidates to compare against | Should not crash | `error: EmptyGalleryError` | Caught before any distance computation is attempted. |
| 12 | Missing/unconfigured model (`TorchEmbedder` stub, no weights) | `embed()` called on a backend with nothing loaded | Should not crash the process | `error: ModelNotLoadedError` (via the pipeline's generic `CVPipelineError` handling) | Demonstrates the pipeline degrades to a typed error even for an entirely different backend, not just the shipped LBP one. |
| 13 | Corrupt image (well-formed filename, garbage bytes) | `cv2.imdecode` returns `None` | Should not crash | `error: UnsupportedInputError` | Explicit `None`/empty-array check in `pipeline.load_image()` before anything touches the "image." |
| 14 | Unsupported input type (`.txt` file, and a raw Python `int`) | Neither is a path to a decodable image nor bytes/ndarray | Should not crash | `error: UnsupportedInputError` for both cases | `load_image()` has an explicit type check with no fallback that could misinterpret arbitrary input as an image. |
| 15 | Insufficient CPU/memory condition | Not exercised destructively (deliberately — actually starving the sandbox of memory is unsafe/unreliable to automate here) | Should degrade predictably rather than hang or crash the host | A configurable `ResourceLimitError` guard (`max_image_megapixels`) is enforced *before* detection runs — verified in `test_resource_limit_guard_on_oversized_input` by setting an artificially tiny ceiling against a 1500×1500 random image | This is the pragmatic, safe substitute for literally exhausting host memory: a pre-flight size ceiling that fails fast and typed, instead of letting a huge frame reach OpenCV uncontrolled. |

## What these 15 failure modes prove together

- **No input, however malformed, produces an unhandled exception.** Every
  path terminates in a typed `IdentityResult` with `error` set or `match: false`.
- **"Unknown person" and "error" are kept structurally distinct** (rows 8 vs.
  1/13/14) — a stranger at the door is not a bug, and the contract makes
  sure HIAS can tell the difference.
- **Confidence is not the same number as similarity** (row 9) — this is the
  specific mechanism that stops a "close enough" match from being treated
  as certain.
