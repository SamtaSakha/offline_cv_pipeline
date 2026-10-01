"""
Measures the metrics required by Day 5 of the brief and writes them to
evidence/benchmark_results.json (docs/BENCHMARK.md is written FROM this
file's output, not the other way around -- the numbers in the doc are real).
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bhiv_cv import CVPipeline  # noqa: E402
from bhiv_cv.detection import FaceDetector  # noqa: E402
from bhiv_cv.preprocessing import crop_and_normalize  # noqa: E402
from bhiv_cv.representation import LBPGridEmbedder  # noqa: E402
from bhiv_cv.matching import match as run_match  # noqa: E402
from bhiv_cv.pipeline import load_image  # noqa: E402

PROBE = ROOT / "data" / "probes" / "probe_known_candidate_001_clean.png"
GALLERY_IMG_1 = ROOT / "data" / "gallery" / "candidate_001" / "sample_01_clean.png"
GALLERY_IMG_2 = ROOT / "data" / "gallery" / "candidate_002" / "sample_01_clean.png"

N_WARM_RUNS = 60
process = psutil.Process(os.getpid())


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def model_size_report():
    """LBP has no external weight file -- the 'model' is the source code
    (a fixed transform) plus the enrolled gallery. Report both honestly."""
    src_files = list((ROOT / "src" / "bhiv_cv").glob("*.py"))
    src_bytes = sum(f.stat().st_size for f in src_files)
    return {
        "external_pretrained_weights_mb": 0.0,
        "pipeline_source_code_kb": round(src_bytes / 1024, 1),
        "note": "LBP is a fixed mathematical transform, not a trained model "
                "with weight files, so there is no weight-file size to report. "
                "Cost per enrolled identity below is the real memory driver.",
    }


def per_identity_embedding_cost(embedder: LBPGridEmbedder):
    dim = embedder.embedding_dim
    bytes_per_identity = dim * 8  # float64
    return {
        "embedding_dimensions": dim,
        "bytes_per_enrolled_template": bytes_per_identity,
        "kb_per_1000_identities": round(bytes_per_identity * 1000 / 1024, 1),
    }


def measure_startup():
    t0 = time.perf_counter()
    pipeline = CVPipeline()
    t1 = time.perf_counter()
    return pipeline, (t1 - t0) * 1000  # ms


def measure_cold_and_warm_inference(pipeline):
    pipeline.enroll(str(GALLERY_IMG_1), "candidate_001")
    pipeline.enroll(str(GALLERY_IMG_2), "candidate_002")

    t0 = time.perf_counter()
    result_cold = pipeline.identify(str(PROBE))
    cold_ms = (time.perf_counter() - t0) * 1000
    assert result_cold.error is None, f"cold run failed: {result_cold.error}"

    warm_times = []
    for _ in range(N_WARM_RUNS):
        t0 = time.perf_counter()
        r = pipeline.identify(str(PROBE))
        warm_times.append((time.perf_counter() - t0) * 1000)
        assert r.error is None

    return cold_ms, warm_times


def measure_stage_breakdown():
    detector = FaceDetector()
    embedder = LBPGridEmbedder()
    img = load_image(str(PROBE))

    detect_times, prep_times, embed_times, match_times = [], [], [], []
    gallery = {"candidate_001": [embedder.embed(crop_and_normalize(
        img, detector.detect(img)[0]))]}

    for _ in range(N_WARM_RUNS):
        t0 = time.perf_counter()
        boxes = detector.detect(img)
        detect_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        face = crop_and_normalize(img, boxes[0])
        prep_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        embedding = embedder.embed(face)
        embed_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        run_match(embedding, gallery)
        match_times.append((time.perf_counter() - t0) * 1000)

    def summarize(name, values):
        return {
            "stage": name,
            "mean_ms": round(statistics.mean(values), 3),
            "p95_ms": round(percentile(values, 95), 3),
        }

    return [
        summarize("detection", detect_times),
        summarize("preprocessing", prep_times),
        summarize("representation", embed_times),
        summarize("matching", match_times),
    ]


def measure_memory_and_cpu(pipeline):
    cpu_start = process.cpu_times()
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    for _ in range(N_WARM_RUNS):
        pipeline.identify(str(PROBE))

    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    cpu_end = process.cpu_times()

    cpu_seconds_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    return {
        "rss_before_mb": round(mem_before_mb, 1),
        "rss_after_mb": round(mem_after_mb, 1),
        "rss_delta_mb_over_60_runs": round(mem_after_mb - mem_before_mb, 2),
        "cpu_seconds_for_60_identify_calls": round(cpu_seconds_used, 3),
        "avg_cpu_ms_per_identify_call": round(cpu_seconds_used * 1000 / N_WARM_RUNS, 3),
        "logical_cpu_count_on_this_host": psutil.cpu_count(logical=True),
    }


def main():
    results = {}
    results["environment"] = {
        "python": sys.version.split()[0],
        "cpu_count": psutil.cpu_count(logical=True),
        "note": "Single-CPU sandboxed container, no GPU, no network access. "
                "Numbers characterize the algorithm's relative behavior, not "
                "an absolute production-hardware SLA.",
    }
    results["model_size"] = model_size_report()

    pipeline, startup_ms = measure_startup()
    results["startup_time_ms"] = round(startup_ms, 3)
    results["per_identity_storage_cost"] = per_identity_embedding_cost(pipeline.embedder)

    cold_ms, warm_times = measure_cold_and_warm_inference(pipeline)
    results["cold_inference_ms"] = round(cold_ms, 3)
    results["warm_inference"] = {
        "n_runs": N_WARM_RUNS,
        "mean_ms": round(statistics.mean(warm_times), 3),
        "median_ms": round(statistics.median(warm_times), 3),
        "p95_ms": round(percentile(warm_times, 95), 3),
        "min_ms": round(min(warm_times), 3),
        "max_ms": round(max(warm_times), 3),
    }

    results["stage_breakdown"] = measure_stage_breakdown()

    fresh_pipeline_for_resources = CVPipeline()
    fresh_pipeline_for_resources.enroll(str(GALLERY_IMG_1), "candidate_001")
    fresh_pipeline_for_resources.enroll(str(GALLERY_IMG_2), "candidate_002")
    results["resource_usage"] = measure_memory_and_cpu(fresh_pipeline_for_resources)

    out_path = ROOT / "evidence" / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
