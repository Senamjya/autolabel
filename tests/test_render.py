"""Tests for review-image rendering: the footer, the histogram, and diagnostic PNGs."""

import os
from pathlib import Path

import cv2
import numpy as np

from _fixtures import (
    CYAN,
    ORANGE,
    WHITE,
    _compact_blob,
    _fibre_blob,
    _graded_blob,
    _imread,
    _params,
    _two_touching_blobs,
)
from autolabel.config import FONT, GREEN
from autolabel.detect import detect
from autolabel.measure import equiv_diameter_um
from autolabel.render import draw_footer, write_size_histogram

# where the diagnostic PNGs land. Override to point somewhere handy while eyeballing
DIAG_DIR = Path(
    os.environ.get("AUTOLABEL_DIAG_DIR", Path(__file__).parent / "_diagnostics")
)


def test_size_histogram_has_no_footer(tmp_path):
    """size_hist.png stays clean: drawing the footer still changes its bottom-left."""
    dst = tmp_path / "size_hist.png"
    write_size_histogram([100.0, 200.0, 300.0, 400.0], dst)
    hist = _imread(dst)
    footered = hist.copy()
    draw_footer(footered)
    h = hist.shape[0]
    corner = (slice(h - 30, h), slice(0, 300))
    # were a footer already present, redrawing it would be a near no-op there
    assert not np.array_equal(hist[corner], footered[corner])


def test_draw_footer_marks_bottom_left_only():
    """draw_footer alters the bottom-left corner and leaves the top-right untouched."""
    img = np.full((80, 400, 3), 100, np.uint8)
    before = img.copy()
    draw_footer(img)
    assert not np.array_equal(img[60:80, 0:200], before[60:80, 0:200])
    assert np.array_equal(img[0:20, 200:400], before[0:20, 200:400])


def _banner(canvas, lines, height=76):
    """Dim a top strip of canvas and write coloured legend lines over it."""
    canvas[:height] = (0.35 * canvas[:height]).astype(np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(canvas, text, (10, 24 + i * 22), FONT, 0.5, color, 1, cv2.LINE_AA)


def _zoom(img, z):
    """Nearest-neighbour upscale so individual pixels stay crisp and countable."""
    return cv2.resize(img, (0, 0), fx=z, fy=z, interpolation=cv2.INTER_NEAREST)


def _rect(canvas, box, z, color):
    """Draw a pixel box scaled by z onto a zoomed canvas."""
    x, y, w, h = box
    cv2.rectangle(canvas, (x * z, y * z), ((x + w) * z, (y + h) * z), color, 2)


def test_diagnostic_images_render():
    """Write annotated PNGs of the shape fixtures to tests/_diagnostics/ for eyeballing.

    Not a behaviour assertion: it regenerates the images a human inspects (override the
    directory with AUTOLABEL_DIAG_DIR). The behaviour itself is asserted by the tests above.
    """
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    # padded display box (green) vs true bbox (orange), on an irregular blob
    img, mask = _compact_blob()
    x, y, w, h = cv2.boundingRect(mask)
    pad, z = 2, 8
    d = detect(img, _params(pad=pad), disk=None)[0][0]
    canvas = _zoom(img, z)
    _rect(canvas, d.display_box, z, GREEN)
    _rect(canvas, (x, y, w, h), z, ORANGE)
    _banner(
        canvas,
        [
            (
                f"green display box: {d.display_box[2]}x{d.display_box[3]} px (padded {pad})",
                GREEN,
            ),
            (f"orange true bbox: {w}x{h} px", ORANGE),
            (f"measured extent: {d.bbox[2]}x{d.bbox[3]} px", WHITE),
        ],
    )
    cv2.imwrite(str(DIAG_DIR / "padding_split.png"), canvas)

    # core seed (orange) vs grown-to-halo (green), on a graded blob
    img = _graded_blob()
    core = detect(img, _params(threshold=100, grow_threshold=0), disk=None)[0][0]
    grown = detect(img, _params(threshold=100, grow_threshold=25), disk=None)[0][0]
    canvas = _zoom(img, 6)
    _rect(canvas, grown.display_box, 6, GREEN)
    _rect(canvas, core.display_box, 6, ORANGE)
    _banner(
        canvas,
        [
            (f"orange core seed: {core.area} px", ORANGE),
            (f"green grown to halo: {grown.area} px", GREEN),
            ("grow_threshold reaches the faded edge", WHITE),
        ],
    )
    cv2.imwrite(str(DIAG_DIR / "grow_halo.png"), canvas)

    # long axis (green bbox) vs area-equivalent-diameter disk (cyan), on a fibre
    img, mask = _fibre_blob()
    d = detect(img, _params(pad=0, grow_threshold=0), disk=None)[0][0]
    equiv = equiv_diameter_um(d.area, 1.0)
    canvas = _zoom(img, 8)
    _rect(canvas, d.display_box, 8, GREEN)
    bx, by, bw, bh = d.display_box
    cv2.circle(
        canvas,
        (int((bx + bw / 2) * 8), int((by + bh / 2) * 8)),
        int(equiv / 2 * 8),
        CYAN,
        2,
    )
    _banner(
        canvas,
        [
            (f"Feret length {d.length_px:.0f} px, width {d.width_px:.0f} px", WHITE),
            (f"cyan equiv-diameter disk: {equiv:.1f} px (same area {d.area})", CYAN),
            (
                f"green axis-aligned bbox long side: {max(d.bbox[2], d.bbox[3])} px",
                GREEN,
            ),
        ],
    )
    cv2.imwrite(str(DIAG_DIR / "fibre.png"), canvas)

    # two particles grown together, split at the neck into one green box per core
    img = _two_touching_blobs()
    dets = detect(img, _params(grow_threshold=25), disk=None)[0]
    canvas = _zoom(img, 4)
    for d in dets:
        _rect(canvas, d.display_box, 4, GREEN)
    _banner(
        canvas,
        [
            (f"{len(dets)} boxes from one merged grown blob", GREEN),
            ("split at the neck, one box per bright core", WHITE),
        ],
    )
    cv2.imwrite(str(DIAG_DIR / "split.png"), canvas)

    for name in ("padding_split.png", "grow_halo.png", "fibre.png", "split.png"):
        assert (DIAG_DIR / name).exists()
