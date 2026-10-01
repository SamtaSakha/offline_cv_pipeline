"""
Runs the pipeline across every probe image in data/probes and writes a
JSONL runtime log to evidence/runtime_log.jsonl -- this is the raw evidence
that docs/FAILURE_TESTS.md and docs/REVIEW_PACKET.md cite.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bhiv_cv import CVPipeline  # noqa: E402

GALLERY_DIR = ROOT / "data" / "gallery"
PROBE_DIR = ROOT / "data" / "probes"
LOG_PATH = ROOT / "evidence" / "runtime_log.jsonl"


def main():
    pipeline = CVPipeline()
    entries = []

    for identity_dir in sorted(GALLERY_DIR.iterdir()):
        img = identity_dir / "sample_01_clean.png"
        result = pipeline.enroll(str(img), identity_dir.name)
        entries.append({"stage": "enroll", "identity": identity_dir.name,
                         "input": str(img.relative_to(ROOT)), "result": result.to_json()})

    probe_files = sorted(p for p in PROBE_DIR.iterdir() if p.suffix in (".png", ".txt"))
    for probe in probe_files:
        result = pipeline.identify(str(probe))
        entries.append({"stage": "identify", "input": str(probe.relative_to(ROOT)),
                         "result": result.to_json()})

    with open(LOG_PATH, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    print(f"Wrote {len(entries)} runtime log entries to {LOG_PATH}")
    for e in entries:
        tag = e["result"].get("error") or ("MATCH:" + str(e["result"]["identity"])) \
            if e["stage"] == "identify" else e["result"].get("error") or "enrolled"
        print(f"  [{e['stage']:8s}] {e['input']:55s} -> {tag}")


if __name__ == "__main__":
    main()
