# Next Stage Proposal (Test 3)

Based on the limitations discovered in Test 2, the pipeline is architecturally sound but algorithmically brittle. For Test 3, I propose the following roadmap:

## 1. Implement a Deep Learning Embedder
- **Action:** Replace `LBPGridEmbedder` with the stubbed `TorchEmbedder` using a lightweight, permissible-license model (e.g., MobileFaceNet).
- **Justification:** LBP failed 100% of the degraded image tests. A deep learning representation is required for real-world reliability against lighting and pose variations.

## 2. Upgrade the Detector
- **Action:** Replace the `cv2.CascadeClassifier` with an SSD-based or MTCNN face detector.
- **Justification:** The Haar Cascade failed to detect faces at side angles or in low light, meaning the matching engine was completely bypassed.

## 3. Implement Liveness Detection
- **Action:** Add a lightweight anti-spoofing stage to the pipeline.
- **Justification:** Currently, the system can be bypassed by holding a high-quality photograph of an enrolled identity up to the camera.

## 4. Integrate with Hardware Controller
- **Action:** Build a mock HIAS access-control script that consumes the `IdentityResult` JSON and simulated a relay/door-unlock event only when `match=true` and `confidence > 0.85`.
- **Justification:** Proves the separation of concerns designed in Stage 4.
