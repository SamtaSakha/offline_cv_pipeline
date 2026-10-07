# Stage 4: BHIV CV Capability Design

## System Architecture and Conceptual Flow
To ensure a secure and scalable architecture, the Computer Vision (CV) module acts as a strict **Identity Provider**, not an **Authorization Engine**. 

**Conceptual Flow:**
`Raw Image Frame -> CV Runtime (Pipeline) -> IdentityResult -> HIAS Access Controller -> Hardware (Door Lock)`

Under this design, the CV module only answers the question: *"Who is this, and how sure are we?"*
It **must never** decide whether that person is allowed inside.

## Contract Definitions

### 1. Input Contract
The CV pipeline accepts raw images in various formats (File Path, Raw Bytes, NumPy array). This is handled by a unified decoder that protects the system from malformed inputs.
```python
def identify(source: Union[str, bytes, np.ndarray], require_single_face: bool = True) -> IdentityResult
```

### 2. Output Contract (`IdentityResult`)
Every call, whether successful or a catastrophic failure, returns a strictly typed `IdentityResult` object.
```json
{
  "identity": "candidate_001",
  "match": true,
  "similarity": 0.98,
  "confidence": 0.85,
  "model_version": "cv-baseline-lbp-v1",
  "trace_id": "uuid-1234",
  "error": null
}
```

### 3. Unknown Identity & Error States
- **Unknown Person:** If the face is detected but falls below the similarity threshold, `identity` is `null`, `match` is `false`, and `error` is `null`.
- **Error State:** If the image is blurry, corrupted, or has no face, `error` contains a typed string (e.g., `"NoFaceDetectedError"`) and `match` is forced to `false`.

### 4. Confidence vs. Similarity
- **Similarity:** Absolute distance to the closest gallery template. (Is it close to Candidate 1?)
- **Confidence:** Margin-based distinctiveness. (Is it close to Candidate 1, and *far away* from Candidate 2?). This prevents false matches between twins or similar-looking individuals.

### 5. Model Versioning & Replacement Strategy
The `IdentityResult` injects `model_version`. If we swap the underlying LBP engine for a Deep Learning model, the version increments to `cv-facenet-v1`. 
Because the output contract remains identical, downstream HIAS access controllers don't need any code changes. They just log the new version.

---

## Comparison: LBP Baseline vs. Deep Learning (e.g., FaceNet)

If we were to replace the LBP grid embedder with a Deep Learning alternative (like a PyTorch-based FaceNet/InceptionResnetV1 model), the trade-offs would be:

| Feature | Local Binary Patterns (LBP) | Deep Learning (FaceNet) |
| :--- | :--- | :--- |
| **Accuracy (Wild)** | Low. Breaks under bad lighting or rotation. | Very High. Handles pose and lighting variations easily. |
| **Model Size** | Zero. Mathematical transform. | ~90 MB to 250 MB weights download. |
| **Compute Cost** | Very low. Runs in milliseconds on CPU. | High. Requires GPU or heavy CPU computation. |
| **Provenance / IP** | Absolute clarity. Public domain math algorithm. | Complex. Models are trained on massive, scraped public face datasets (e.g., MS-Celeb-1M, VGGFace2) which carry heavy GDPR, copyright, and licensing risks. |

### Proposed Evaluation Method for Deep Learning
Because network access and weight downloads are restricted in this secure environment, we cannot live-test a PyTorch model today. If approved, the evaluation protocol would be:
1. Air-gap a clean machine with PyTorch installed.
2. Sideload the pre-trained `vggface2` `.pth` weight file.
3. Wrap it in a class that implements our `Embedder` interface (which expects an image and returns a 1D vector).
4. Re-run our Stage 2 failure scripts. We would expect the `probe_blur.png` and `probe_poor_lighting.png` to correctly yield matches rather than failing.
