"""
Day 5: investigates at least two optimization approaches against the
detection-stage bottleneck identified in scripts/benchmark.py (detection was
~87% of total per-request latency). Writes real, measured trade-off numbers
to evidence/optimization_results.json.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bhiv_cv.detection import FaceDetector  # noqa: E402
from bhiv_cv.config import DetectionConfig  # noqa: E402
from bhiv_cv.pipeline import load_image  # noqa: E402

PROBE = ROOT / "data" / "probes" / "probe_known_candidate_001_clean.png"
N_RUNS = 40


def time_detect(detector, img, n=N_RUNS):
    times = []
    face_found = False
    for _ in range(n):
        t0 = time.perf_counter()
        boxes = detector.detect(img)
        times.append((time.perf_counter() - t0) * 1000)
        if len(boxes) >= 1:
            face_found = True
    return {
        "mean_ms": round(statistics.mean(times), 3),
        "p95_ms": round(sorted(times)[int(0.95 * (len(times) - 1))], 3),
        "still_detects_face": face_found,
    }


def main():
    original = load_image(str(PROBE))
    results = {}

    # Baseline (same config as benchmark.py)
    baseline_detector = FaceDetector()
    results["baseline_512x512"] = time_detect(baseline_detector, original)

    # --- Optimization 1: image-size reduction before detection -----------
    for scale, label in [(0.5, "resized_256x256"), (0.25, "resized_128x128")]:
        h, w = original.shape[:2]
        small = cv2.resize(original, (int(w * scale), int(h * scale)))
        results[f"opt1_{label}"] = time_detect(baseline_detector, small)

    # --- Optimization 2: coarser scan (larger scaleFactor / minNeighbors)--
    for sf, mn, label in [(1.2, 5, "scaleFactor_1.2"), (1.3, 3, "scaleFactor_1.3_minN_3")]:
        coarse_detector = FaceDetector(DetectionConfig(scale_factor=sf, min_neighbors=mn))
        results[f"opt2_{label}"] = time_detect(coarse_detector, original)

    # --- Combined: resize + coarser scan -----------------------------------
    h, w = original.shape[:2]
    small = cv2.resize(original, (int(w * 0.5), int(h * 0.5)))
    combined_detector = FaceDetector(DetectionConfig(scale_factor=1.2, min_neighbors=5))
    results["opt_combined_resize+coarse_scan"] = time_detect(combined_detector, small)

    out_path = ROOT / "evidence" / "optimization_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
