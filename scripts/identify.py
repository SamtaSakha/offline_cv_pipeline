#!/usr/bin/env python3
"""
CLI: identify one probe image against a persisted gallery file, and print
exactly the integration-contract JSON object that would cross into HIAS.

Usage:
    python scripts/identify.py --image path/to/probe.png \
        --gallery data/gallery_store.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bhiv_cv import CVPipeline  # noqa: E402
from bhiv_cv.gallery import Gallery  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gallery", default=str(ROOT / "data" / "gallery_store.json"))
    ap.add_argument("--allow-multi-face", action="store_true")
    args = ap.parse_args()

    pipeline = CVPipeline()
    if Path(args.gallery).exists():
        pipeline.gallery = Gallery.load(args.gallery)

    result = pipeline.identify(args.image, require_single_face=not args.allow_multi_face)
    print(json.dumps(result.to_json(), indent=2))


if __name__ == "__main__":
    main()
