# Known Limitations

1. **Detection Fragility:**
   The OpenCV Haar Cascade is highly sensitive to rotation (side profiles), lighting conditions, and occlusion. It fails completely (raises `NoFaceDetectedError`) under mild real-world degradation.
2. **Linear Matching Scaling:**
   The gallery matching algorithm loops through every enrolled template calculating Chi-Square distance. At 10,000 identities, this takes ~165ms. While acceptable for a building lobby, this O(N) scaling will become a bottleneck for a system with 100,000+ identities. It requires vectorization or an indexing system (like FAISS/HNSW) for extreme scale.
3. **No Anti-Spoofing (Liveness Detection):**
   The pipeline has no mechanism to differentiate a 3D human face from a 2D photograph held up to the camera.
4. **Single-Face Policy Handling:**
   If multiple faces are detected during authorization, the pipeline conservatively rejects the frame entirely to prevent "tailgating" authorizations. This requires users to approach the camera alone.
