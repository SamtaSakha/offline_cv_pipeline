# ARCHITECTURE.md

## Pipeline architecture (internal to the CV module)

```
Input Image
    │
    ▼
┌─────────────┐   src/bhiv_cv/detection.py
│  Detection  │   Haar Cascade → List[BoundingBox] (+ pseudo-confidence)
└──────┬──────┘
       ▼
┌─────────────┐   src/bhiv_cv/preprocessing.py
│Preprocessing│   crop + margin, grayscale, resize to 200×200, histogram equalize
└──────┬──────┘
       ▼
┌─────────────┐   src/bhiv_cv/representation.py
│Representation│  grid-based LBP → 640-d float embedding
└──────┬──────┘
       ▼
┌─────────────┐   src/bhiv_cv/gallery.py + src/bhiv_cv/matching.py
│  Matching   │   chi-square distance vs. enrolled templates → similarity + confidence
└──────┬──────┘
       ▼
┌─────────────────┐   src/bhiv_cv/contracts.py
│ IdentityResult   │   {identity, match, similarity, confidence, model_version, trace_id}
└──────┬──────────┘
       │  <-- THIS IS THE ONLY THING THAT LEAVES THE MODULE
       ▼
   [ BOUNDARY ]
       ▼
   HIAS Identity Layer  →  Deterministic Controller  →  Access Decision  →  Truth/Event Layer
```

## Module boundaries (single-responsibility, independently testable)

| Module | Owns | Does NOT own |
|---|---|---|
| `detection.py` | Finding face regions in a frame | Whose face it is, whether it's a "good" crop |
| `preprocessing.py` | Deterministic normalization of a crop | Detection, embedding |
| `representation.py` | Turning a crop into a comparable vector | What threshold counts as a match |
| `matching.py` | Comparing a vector against a gallery, scoring | Detection, embedding, deciding access |
| `gallery.py` | Storing/persisting enrolled templates | Matching logic |
| `contracts.py` | Defining the ONE object that crosses the HIAS boundary | Everything upstream of it |
| `pipeline.py` | Orchestration + converting internal exceptions to the bounded contract | Business/access logic |

Every stage is a pure function or a small class with an obvious single
input/output type (`tests/test_detection.py`, `test_preprocessing.py`,
`test_representation.py`, `test_matching.py` each test exactly one stage in
isolation — 37/37 passing, see `evidence/test_results_raw.txt`).

## The capability boundary — what this module is structurally incapable of doing

The `IdentityResult` dataclass (`contracts.py`) has exactly these fields:
`identity`, `match`, `similarity`, `confidence`, `model_version`, `trace_id`,
`timestamp`, `error`. **There is no field for an access decision** — no
`authorized`, `door_unlock`, `access_granted`, or similar. This is a
deliberate structural constraint, not a policy the module chooses to follow:
even a buggy caller cannot accidentally read an access decision out of this
object, because the object cannot express one.

`match: true` means "similarity and confidence cleared a configured
statistical threshold" — it is a *recognition* claim about identity, not an
*authorization* claim about access. The distinction is enforced end-to-end:
see `tests/test_pipeline_failures.py::test_low_confidence_never_becomes_a_match_even_with_lenient_thresholds`,
which regression-guards against exactly this failure mode.

## What the CV module owns vs. what HIAS must own

| | CV module (this repo) | HIAS (downstream) |
|---|---|---|
| Face detection, preprocessing, embedding | ✅ | — |
| Similarity/confidence scoring | ✅ | — |
| Reporting "unknown" vs. "recognized as X" | ✅ | — |
| Deciding if X is *allowed* through today, right now | ❌ never | ✅ always |
| Combining recognition with other signals (schedule, role, override) | ❌ | ✅ |
| Audit/event logging of what was decided and why | ❌ (only logs what was recognized, not what was decided) | ✅ |
| Physical actuation | ❌ | ✅ (or a further downstream consumer) |

## Configuration surface (nothing hard-coded)

All tunables live in `src/bhiv_cv/config.py` as frozen dataclasses:
`DetectionConfig`, `PreprocessConfig`, `RepresentationConfig`,
`MatchingConfig`. No real identity, hostel name, or organisation-specific
value appears anywhere in source code — the gallery is populated entirely
at runtime via `enroll()`.

## Versioning and provenance in the contract itself

Every `IdentityResult` carries `model_version` (currently
`"cv-baseline-lbp-v1"`) and a fresh `trace_id` (UUID) per call. This means:
- HIAS can log which model version produced a given result, so a future
  model swap (e.g. to a PyTorch backend) is auditable in the event history.
- Every call is independently traceable end-to-end without needing to
  correlate on timestamp alone.
