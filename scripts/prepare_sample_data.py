"""
Builds this repo's sample dataset from two public-domain, standard
computer-vision benchmark photographs that ship OFFLINE with widely used
Python libraries (no network access used or required):

  - "astronaut": official NASA portrait of astronaut Eileen Collins,
    bundled with scikit-image (skimage.data.astronaut). Public domain
    (US federal government work). One of the most widely used images in
    CV research/teaching for decades.

  - "grace_hopper": official US Navy portrait of Grace Hopper, bundled
    with matplotlib (mpl-data/sample_data/grace_hopper.jpg) and also used
    by the official PyTorch tutorials as a standard demo image. Public
    domain (US federal government work).

These give us two distinct, real, correctly-detectable human identities
with zero licensing ambiguity, without needing internet access to download
a face dataset (see docs/LICENSE_AND_PROVENANCE.md).

This script also synthesizes the degraded variants used in Day 4's
failure-mode testing (blur, low light, rotation, partial occlusion, etc.)
by applying standard, documented OpenCV transforms to these two base images.
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
from skimage import data

ROOT = Path(__file__).resolve().parent.parent
GALLERY_DIR = ROOT / "data" / "gallery"
PROBE_DIR = ROOT / "data" / "probes"


def save(img_bgr, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr)


def get_base_images():
    astronaut_rgb = data.astronaut()
    astronaut_bgr = cv2.cvtColor(astronaut_rgb, cv2.COLOR_RGB2BGR)

    hopper_path = Path(
        __import__("matplotlib").get_data_path()
    ) / "sample_data" / "grace_hopper.jpg"
    hopper_bgr = cv2.imread(str(hopper_path), cv2.IMREAD_COLOR)

    return {"candidate_001": astronaut_bgr, "candidate_002": hopper_bgr}


# ---- degradation transforms used for Day 4 failure analysis ---------------

def make_blurred(img, ksize=25):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def make_dark(img, factor=0.15):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def make_rotated(img, angle=35):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def make_partial(img, frac=0.55):
    """Occlude the right-hand fraction of the image (simulates partial face)."""
    out = img.copy()
    w = out.shape[1]
    cut = int(w * (1 - frac))
    out[:, cut:] = 0
    return out


def make_low_res(img, scale=0.06):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def make_multi_face_canvas(img_a, img_b):
    """Paste two known faces side by side -> a frame with 2 faces."""
    h = 512
    a = cv2.resize(img_a, (h, h))
    b = cv2.resize(img_b, (h, h))
    return np.hstack([a, b])


def main():
    if GALLERY_DIR.exists():
        shutil.rmtree(GALLERY_DIR)
    if PROBE_DIR.exists():
        shutil.rmtree(PROBE_DIR)

    base = get_base_images()

    # --- gallery: 3 clean enrollment samples per identity (crops/zooms) ---
    for identity, img in base.items():
        save(img, GALLERY_DIR / identity / "sample_01_clean.png")
        save(cv2.resize(img, (int(img.shape[1] * 0.9), int(img.shape[0] * 0.9))),
             GALLERY_DIR / identity / "sample_02_scaled.png")
        save(cv2.convertScaleAbs(img, alpha=1.1, beta=10),
             GALLERY_DIR / identity / "sample_03_brighter.png")

    a = base["candidate_001"]
    b = base["candidate_002"]

    # --- probes: clean, known and "unknown"-style test cases -------------
    save(a, PROBE_DIR / "probe_known_candidate_001_clean.png")
    save(b, PROBE_DIR / "probe_known_candidate_002_clean.png")

    # --- Day 4 required failure/edge cases --------------------------------
    save(make_blurred(a), PROBE_DIR / "probe_blur.png")
    save(make_dark(a), PROBE_DIR / "probe_poor_lighting.png")
    save(make_rotated(a), PROBE_DIR / "probe_side_angle.png")
    save(make_partial(a), PROBE_DIR / "probe_partial_face.png")
    save(make_low_res(a), PROBE_DIR / "probe_poor_quality.png")
    save(make_multi_face_canvas(a, b), PROBE_DIR / "probe_multiple_faces.png")

    # blank / no-face frame
    save(np.full((400, 400, 3), 200, dtype=np.uint8), PROBE_DIR / "probe_no_face.png")

    # unsupported / corrupt inputs (written directly, not via cv2.imwrite)
    (PROBE_DIR / "probe_corrupt.png").write_bytes(b"\x89PNGnotarealfileatall\x00\x01\x02")
    (PROBE_DIR / "probe_unsupported.txt").write_text("this is not an image")

    print(f"Gallery written to {GALLERY_DIR}")
    print(f"Probes written to {PROBE_DIR}")


if __name__ == "__main__":
    main()
