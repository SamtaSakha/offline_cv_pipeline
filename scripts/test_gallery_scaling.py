import time
import numpy as np
from bhiv_cv.pipeline import CVPipeline

pipeline = CVPipeline()

# Generate a fake embedding (LBP size is 640 for 8x8 grid with 10 bins)
fake_embedding = np.random.rand(640).astype(np.float64)

sizes = [2, 100, 500, 1000, 5000, 10000]

print("Gallery Size | Matching Latency (ms)")
print("-----------------------------------")
for size in sizes:
    # Build gallery
    gallery = {f"id_{i}": [np.random.rand(640).astype(np.float64)] for i in range(size)}
    
    # Warm up
    from bhiv_cv.matching import match
    for _ in range(10):
        match(fake_embedding, gallery)
    
    # Benchmark
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        match(fake_embedding, gallery)
        times.append((time.perf_counter() - t0) * 1000)
    
    mean_ms = np.mean(times)
    p95_ms = np.percentile(times, 95)
    print(f"{size:12} | {mean_ms:6.2f} (p95: {p95_ms:6.2f})")
