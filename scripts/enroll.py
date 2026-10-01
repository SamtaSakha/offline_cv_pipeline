#!/usr/bin/env python3
"""
CLI: enroll one image into the gallery file.

Usage:
    python scripts/enroll.py --image path/to/face.png --identity candidate_003 \
        --gallery data/gallery_store.json
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bhiv_cv import CVPipeline  # noqa: E402
from bhiv_cv.gallery import Gallery  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--identity", required=True)
    ap.add_argument("--gallery", default=str(ROOT / "data" / "gallery_store.json"))
    ap.add_argument("--allow-update", action="store_true")
    args = ap.parse_args()

    pipeline = CVPipeline()
    if Path(args.gallery).exists():
        pipeline.gallery = Gallery.load(args.gallery)

    result = pipeline.enroll(args.image, args.identity, allow_update=args.allow_update)
    print(result.to_json())

    if result.error is None:
        pipeline.gallery.save(args.gallery)
        print(f"Saved gallery to {args.gallery}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
