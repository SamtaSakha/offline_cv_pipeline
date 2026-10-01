# INTEGRATION.md

## The integration contract

This is the complete, only object that crosses from the CV module into
HIAS. Produced by `IdentityResult.to_json()` (`src/bhiv_cv/contracts.py`).

**Successful recognition:**
```json
{
  "identity": "candidate_001",
  "match": true,
  "similarity": 1.0,
  "confidence": 1.0,
  "model_version": "cv-baseline-lbp-v1",
  "trace_id": "b3ec7fa8-97f3-41f7-b377-ad36a5efdd0e",
  "timestamp": 1790420960.936254,
  "error": null
}
```
(real output, from `evidence/runtime_log.jsonl`)

**Unknown person (not an error — an expected outcome):**
```json
{
  "identity": null,
  "match": false,
  "similarity": 0.0,
  "confidence": 0.6525,
  "model_version": "cv-baseline-lbp-v1",
  "trace_id": "8142e38e-de4a-4e6d-b353-2988033e0085",
  "timestamp": 1790420960.7805338,
  "error": null
}
```
(real output for the blurred probe — see `FAILURE_TESTS.md` row 4)

**Typed failure (no face, corrupt input, ambiguous frame, etc.):**
```json
{
  "identity": null,
  "match": false,
  "similarity": 0.0,
  "confidence": 0.0,
  "model_version": "cv-baseline-lbp-v1",
  "trace_id": "e3f8997d-e9d0-423b-8d37-9317e33e3c2d",
  "timestamp": 1790420961.4202478,
  "error": "NoFaceDetectedError"
}
```
`error` is always one of the exception class names in
`src/bhiv_cv/exceptions.py`, so HIAS can branch on a stable string enum
rather than parsing free-text messages.

## Field reference

| Field | Type | Meaning |
|---|---|---|
| `identity` | `string \| null` | Enrolled identity string, or `null` if unrecognized/unknown. **Not** a person's real name unless the caller chose to enroll them under one — this module has no concept of "who a name belongs to." |
| `match` | `bool` | `true` only if similarity AND confidence both cleared their configured thresholds. **This is a recognition claim, not an access decision.** |
| `similarity` | `float, 0..1` | Closeness of the top-1 gallery candidate. |
| `confidence` | `float, 0..1` | Margin-based distinctiveness of the top-1 candidate vs. the runner-up (or absolute similarity if only one identity is enrolled). Deliberately a *separate* number from similarity — see `FAILURE_TESTS.md` row 9. |
| `model_version` | `string` | Which representation backend produced this result. Changes if `TorchEmbedder` or a future backend replaces the LBP baseline. |
| `trace_id` | `string (UUID)` | Unique per call. Use this to correlate CV-side logs with HIAS-side decision logs. |
| `timestamp` | `float (unix epoch)` | When the CV module produced the result. |
| `error` | `string \| null` | One of the `CVPipelineError` subclass names, or `null` on success (including "unknown person," which is a success, not an error). |

## Errors: full list HIAS must be prepared to branch on

From `src/bhiv_cv/exceptions.py`:
`UnsupportedInputError`, `NoFaceDetectedError`, `MultipleFacesDetectedError`,
`EmptyGalleryError`, `ModelNotLoadedError`, `DuplicateIdentityError`,
`ResourceLimitError`.

**Recommended HIAS-side handling policy (a suggestion, not something this
module enforces — access policy is HIAS's authority, not this module's):**
any non-null `error`, and any `match: false`, should route to the same
"cannot confirm identity" path as an unknown person. Recognition failure and
recognition absence should look the same to a downstream access decision —
only successful, high-confidence, positively-identified `match: true`
results should ever be treated differently.

## What the CV module owns vs. what it must never decide

Restated from `ARCHITECTURE.md` because it's the one rule everything else
serves: **this module reports identity-and-confidence information; it never
evaluates whether that information is *sufficient* for anything.** Whether
`candidate_001` matching at `similarity: 0.91, confidence: 0.94` should open
a specific door, at this specific time, under this specific policy, is
entirely a HIAS Deterministic Controller decision. The CV module has no
door/access/authorization vocabulary in its schema, code, or output, by
construction (see `ARCHITECTURE.md`, "structurally incapable").

## Call pattern assumed by this integration

One `identify()` call per access attempt (a single frame or a
caller-selected best frame from a short burst), not continuous
frame-by-frame streaming analysis. This shapes both the latency profile
(`BENCHMARK.md`, ~158ms/call is entirely acceptable for a per-attempt
challenge, not for 30fps video) and the "exactly one face expected by
default" policy in `identify()`.

## Inputs, outputs, and errors — end-to-end example

```python
from bhiv_cv import CVPipeline

pipeline = CVPipeline()
pipeline.enroll("path/to/enrollment_photo.jpg", "candidate_001")

result = pipeline.identify("path/to/live_capture.jpg")
hias_payload = result.to_json()   # <-- exactly what crosses the boundary
```

## Enrollment vs. identification integration paths

- **Enrollment** is assumed to be an administrative, higher-trust workflow
  (HR/security onboarding a new badge holder), not something exposed at a
  live access point. `enroll()` requires exactly one face by default and
  rejects duplicate identities unless `allow_update=True` is explicitly
  passed — see `Gallery` in `gallery.py`.
- **Identification** is the live, per-attempt path described above.

## Versioning path for a future deep-learning backend swap

1. Implement `TorchEmbedder.embed()` (stub already present in
   `representation.py`).
2. Bump `RepresentationConfig.model_version` to something like
   `"cv-torch-facenet-v1"`.
3. Every `IdentityResult` produced afterward carries the new
   `model_version` automatically — HIAS's event log now has a clean,
   queryable marker of exactly when the swap took effect, with zero contract
   schema change required on either side.
