# AI Usage & Personal Contribution

*Note: This document was structurally drafted by an AI assistant (Antigravity), but records the actual collaborative workflow between the Engineer and the AI.*

## 1. What the AI Generated
- The AI identified the fatal bug in the Haar cascade loading (`cv2.data.haarcascades` fragility) and wrote the fix to load it from `data/models/`.
- The AI executed the test suites (`pytest`) and benchmarking scripts to gather performance data.
- The AI wrote a python script to iterate through the probe images and extract the failure results for Stage 2.
- The AI structured and formatted the markdown documentation for the review packet.

## 2. What I Personally Inspected
- I reviewed the pipeline's modular structure (`CVPipeline`, `FaceDetector`, `LBPGridEmbedder`) to understand the strict separation of concerns.
- I analyzed the JSON outputs of the `identify.py` script to understand how errors (like `NoFaceDetectedError`) are cleanly packaged into the `IdentityResult` contract instead of crashing.
- I read the benchmark results verifying the linear O(N) scaling of the LBP matching algorithm.

## 3. What I Independently Executed
- I initiated the review process and prompted the AI to break down the assignment.
- I orchestrated the progression from Stage 1 through Stage 4.

## 4. What Was Modified / Corrected
- We modified `src/bhiv_cv/detection.py` to point to a local `data/models/haarcascade_frontalface_default.xml` file because the dynamic OpenCV path failed on the host Windows machine.
- We modified the dataset by running `prepare_sample_data.py` to generate the test probes needed for failure validation.

## 5. What I Would Build Differently Next Time
- I would replace the Haar Cascade detector immediately. While LBP matching is adequate for constrained environments, the Haar Cascade failed to even detect faces in slightly dark or blurry images, meaning the matching engine was never even invoked. A lightweight deep-learning detector (like RetinaFace or MTCNN) is mandatory for real-world robustness.
