# BENCHMARK.md

All numbers below are measured, not estimated. Reproduce with:
```
PYTHONPATH=src python3 scripts/benchmark.py
PYTHONPATH=src python3 scripts/optimization_experiments.py
```
Raw output: `evidence/benchmark_results.json`, `evidence/optimization_results.json`.

**Environment (report this alongside every number — hardware changes all of
them):** single logical CPU, no GPU, sandboxed container, Python 3.12.3, no
network access. These numbers characterize the *algorithm's relative
behaviour and bottlenecks*, not an absolute production-hardware SLA.

## Model size

| Metric | Value |
|---|---|
| External pretrained weight files | **0 MB** (LBP is a fixed transform — see `CV_MODEL_EVALUATION.md`) |
| Pipeline source code | 22.3 KB |
| Embedding size per enrolled identity | 640 floats × 8 bytes = 5,120 bytes |
| Storage for 1,000 enrolled identities | ~5.0 MB |

## Startup and inference latency

| Metric | Value |
|---|---|
| Startup time (construct `CVPipeline`, load cascade) | 12.6 ms |
| Cold inference (first `identify()` call, includes any lazy init) | 159.4 ms |
| Warm inference — mean (n=60) | 158.2 ms |
| Warm inference — median | 157.0 ms |
| Warm inference — P95 | 164.3 ms |
| Warm inference — min / max | 155.5 ms / 177.4 ms |

Cold and warm latency are nearly identical (159.4ms vs 158.2ms mean) —
this baseline has **no meaningful cold-start penalty**, unlike a deep model
that must load large weight tensors onto a device on first use.

## Per-stage latency breakdown (mean of 60 runs, on the same probe image)

| Stage | Mean latency | P95 latency | % of total |
|---|---|---|---|
| Detection (Haar Cascade, 512×512 input) | 138.5 ms | 139.9 ms | **87.5%** |
| Representation (LBP embedding) | 10.0 ms | 10.3 ms | 6.3% |
| Preprocessing (crop/resize/equalize) | 0.16 ms | 0.18 ms | 0.1% |
| Matching (chi-square distance vs. 2 identities) | 0.05 ms | 0.05 ms | <0.1% |

**Detection is overwhelmingly the bottleneck** — the multi-scale sliding
window scan dominates total latency. This directly targeted the
optimization work below.

## CPU and memory

| Metric | Value |
|---|---|
| RSS before 60 `identify()` calls | 117.6 MB |
| RSS after 60 `identify()` calls | 117.6 MB (Δ 0.0 MB) |
| CPU time consumed by 60 `identify()` calls | 9.42 CPU-seconds |
| Average CPU time per call | 157.0 ms |

**No memory growth across repeated calls** — no leak under this workload.
CPU time per call (157ms) closely tracks wall-clock latency (158ms),
confirming the pipeline is CPU-bound, not I/O- or lock-bound, which is
expected for a single-threaded classical CV pipeline on one core.

## Optimization investigation (two approaches, as required)

### Approach 1: input image-size reduction before detection

| Variant | Mean latency | Still detects the face? |
|---|---|---|
| Baseline, 512×512 | 140.2 ms | ✅ |
| Resized to 256×256 | 34.3 ms (**4.1x faster**) | ✅ |
| Resized to 128×128 | 4.6 ms (**30x faster**) | ❌ **face detection fails** |

**Finding:** resizing helps a lot, but there's a real accuracy cliff. This
baseline's minimum reliable face size (given its `min_size=(40,40)` config
and the cascade's own resolution needs) means 128×128 downscaling destroys
enough facial detail that zero faces were found across all 40 repeated
runs. 256×256 is the safe, empirically-verified floor for this input.

### Approach 2: coarser multi-scale scan (larger `scaleFactor`, fewer neighbor checks)

| Variant | Mean latency | Still detects the face? |
|---|---|---|
| Baseline (`scaleFactor=1.08`, `minNeighbors=5`) | 140.2 ms | ✅ |
| `scaleFactor=1.2`, `minNeighbors=5` | 68.3 ms (**2.1x faster**) | ✅ |
| `scaleFactor=1.3`, `minNeighbors=3` | 47.8 ms (**2.9x faster**) | ✅ |

**Finding:** coarsening the scan step is a smaller, safer win on its own —
no accuracy cliff observed in this test, but real deployments should sweep
`minNeighbors` carefully since lowering it also raises false-positive rate
(see `FAILURE_TESTS.md` row 3, where a lighting change alone produced a
false positive at the *default* `minNeighbors=5`).

### Combined: moderate resize (256×256) + coarser scan

| Variant | Mean latency | Still detects the face? |
|---|---|---|
| Combined | **17.0 ms** (**8.2x faster than baseline**) | ✅ |

**This is the recommended production configuration** for this baseline:
an 8x latency reduction with no observed loss of detection on the test
probe, bringing detection from 138ms down to roughly the same order of
magnitude as the embedding stage.

## The trade-off, stated explicitly

**Accuracy ↔ latency ↔ memory ↔ CPU ↔ complexity:**
- Memory and per-identity storage cost (5KB/identity) are already
  negligible for this baseline and are not meaningfully affected by any of
  the above — the trade-off space here is really just **accuracy ↔ latency**.
- Latency scales roughly with (a) input pixel count fed to the detector and
  (b) how many scales/positions the cascade checks.
- Both levers buy real speed, but they are not free: aggressive downscaling
  has a hard failure cliff (row 7 of `FAILURE_TESTS.md` shows the same
  cliff appearing independently from real degraded input, not just from
  deliberately shrinking a clean image), and a coarser scan raises
  false-positive risk under adverse conditions (row 3).
- **Complexity cost of the recommended combined config is ~zero** — it's two
  config values (`DetectionConfig(scale_factor=1.2)` + a resize step before
  `detect()`), not an architecture change. This is the kind of first
  optimization to make before considering a heavier lever like swapping
  detector families or adding batching.
- Batching was not applicable here: this is a single-frame,
  single-identification-per-call access-control use case, not a bulk
  offline processing job — see `INTEGRATION.md` for the assumed call
  pattern.
