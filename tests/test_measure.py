"""Tests for the size math: equivalent diameter, disk diameter, and Feret axes."""

import cv2
import numpy as np

from _fixtures import _disk_image, _fibre_blob, _params, _rotate_fixture
from autolabel.detect import detect, find_filter_mask
from autolabel.measure import disk_diameter_px, equiv_diameter_um


def test_equiv_diameter_um_inverts_disk_area():
    """A disk of known area reports its known equivalent diameter (2r, scaled)."""
    area_px = np.pi * 50.0**2  # radius 50 px
    assert equiv_diameter_um(area_px, 2.0) == 200.0


def test_disk_diameter_recovers_true_size_after_erosion():
    """A 240 px disk round-trips to within a few percent through erosion and add-back."""
    mask = find_filter_mask(_disk_image(radius=120))
    assert mask is not None
    assert abs(disk_diameter_px(mask) - 240.0) < 12.0


def test_fiber_long_axis_diverges_from_equivalent_diameter():
    """A fibre's long axis and equivalent diameter differ enough to catch a source swap."""
    img, mask = _fibre_blob()
    _, _, w, h = cv2.boundingRect(mask)
    dets, _ = detect(img, _params(pad=0, grow_threshold=0), disk=None)

    assert len(dets) == 1
    d = dets[0]
    assert d.area == int(mask.sum())
    _, _, bw, bh = d.bbox
    assert (bw, bh) == (w, h)

    equiv = equiv_diameter_um(d.area, 1.0)
    assert bw < equiv < bh  # equiv sits between the fibre's width and length
    assert max(bw, bh) > 2 * equiv  # long axis far exceeds it


def test_feret_length_is_orientation_invariant():
    """Rotating a fibre 45 deg leaves Feret length stable but collapses the bbox long side."""
    img0, mask0 = _fibre_blob()
    img45, _ = _rotate_fixture(mask0, 45)

    d0 = detect(img0, _params(pad=0, grow_threshold=0), disk=None)[0][0]
    d45 = detect(img45, _params(pad=0, grow_threshold=0), disk=None)[0][0]

    # Feret length is stable across orientation (a few px of rasterisation aside)
    assert abs(d0.length_px - d45.length_px) < 0.1 * d0.length_px
    # the axis-aligned bbox long side is not: it shrinks markedly once tilted
    assert max(d45.bbox[2], d45.bbox[3]) < 0.85 * max(d0.bbox[2], d0.bbox[3])
