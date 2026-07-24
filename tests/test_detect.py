"""Tests for pixel scoring, disk masking, and the detection pipeline."""

import cv2
import numpy as np

from _fixtures import _compact_blob, _graded_blob, _params, _two_touching_blobs
from autolabel.config import BLUE_RISE_LIMIT
from autolabel.detect import detect, find_filter_mask, luminance_score, redness_score


def test_redness_score_high_on_red_zero_on_paper():
    """Red pixels score well above the paper. Paper scores zero."""
    img = np.full((10, 10, 3), 100, np.uint8)  # neutral gray "paper"
    img[2:8, 2:8] = (0, 0, 255)  # a red square (BGR)
    score = redness_score(img)
    assert score[5, 5] > 100  # red rises well over the paper
    assert score[0, 0] == 0  # paper scores nothing


def test_redness_score_zeros_specular_blue():
    """A pixel whose blue rises past BLUE_RISE_LIMIT is zeroed as a glint."""
    img = np.full((10, 10, 3), 100, np.uint8)
    img[5, 5] = (100 + BLUE_RISE_LIMIT + 10, 100, 255)  # red but very blue
    assert redness_score(img)[5, 5] == 0


def test_luminance_score_is_brightest_channel():
    """luminance_score returns the brightest channel per pixel."""
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 1] = 200  # green channel brightest
    img[..., 0] = 50
    assert (luminance_score(img) == 200).all()


def test_find_filter_mask_none_without_disk():
    """A frame with no bright disk yields None."""
    assert find_filter_mask(np.zeros((100, 100, 3), np.uint8)) is None


def test_detect_separates_padded_box_from_true_bbox():
    """box is padded for display, while area and bbox report the blob's true extent."""
    img, mask = _compact_blob()
    x, y, w, h = cv2.boundingRect(mask)
    pad = 2
    dets, _ = detect(img, _params(pad=pad), disk=None)

    assert len(dets) == 1
    d = dets[0]
    assert d.display_box == (
        x - pad,
        y - pad,
        w + 2 * pad,
        h + 2 * pad,
    )  # padded to draw
    assert d.area == int(mask.sum())  # true pixel count, whatever the shape
    assert d.bbox == (x, y, w, h)  # true box, no padding


def test_detect_ignores_disk_when_mask_filter_off():
    """mask_filter=False skips the disk, even an all-zero one that would reject all."""
    img = np.full((50, 50, 3), 100, np.uint8)
    img[20:26, 20:26] = (0, 0, 255)
    disk = np.zeros((50, 50), np.uint8)
    dets, _ = detect(img, _params(mask_filter=False), disk=disk)
    assert len(dets) == 1


def test_detect_min_area_drops_small_blobs():
    """Blobs below min_area are dropped."""
    img = np.full((50, 50, 3), 100, np.uint8)
    img[10:12, 10:12] = (0, 0, 255)  # 4 px blob
    dets, _ = detect(img, _params(min_area=5), disk=None)
    assert dets == []


def test_grow_extends_seed_into_dimmer_halo():
    """grow_threshold extends a seed on the bright core out to the dim halo."""
    img = _graded_blob()
    # G and B stay at paper level, so redness_score here reduces to (red - paper)
    score = np.clip(img[..., 2].astype(np.int32) - 100, 0, 255)
    expected_core = int((score > 100).sum())
    expected_grown = int((score > 25).sum())
    assert 0 < expected_core < expected_grown  # the halo really is dimmer than the core

    core_only, _ = detect(img, _params(threshold=100, grow_threshold=0), disk=None)
    grown, _ = detect(img, _params(threshold=100, grow_threshold=25), disk=None)

    assert len(core_only) == 1 and len(grown) == 1
    assert core_only[0].area == expected_core  # seed catches only the bright core
    assert grown[0].area == expected_grown  # grow reaches the whole faded blob
    assert grown[0].area > core_only[0].area


def test_split_separates_touching_particles():
    """Two particles grown together at their halos split into two, one per bright core."""
    img = _two_touching_blobs()
    merged, _ = detect(img, _params(grow_threshold=25, split_touching=False), disk=None)
    split, _ = detect(img, _params(grow_threshold=25, split_touching=True), disk=None)

    assert len(merged) == 1  # the two halos are one connected grown blob
    assert len(split) == 2  # split at the neck between the cores
    # the split conserves every pixel of the merged blob
    assert sum(d.area for d in split) == merged[0].area
    # and yields two comparable particles, not one blob and a sliver
    small, large = sorted(d.area for d in split)
    assert small > 0.4 * large


def test_split_leaves_isolated_particle_unchanged():
    """Splitting is a no-op on a particle with a single core."""
    img = _graded_blob()
    off, _ = detect(img, _params(grow_threshold=25, split_touching=False), disk=None)
    on, _ = detect(img, _params(grow_threshold=25, split_touching=True), disk=None)

    assert len(off) == len(on) == 1
    assert off[0].display_box == on[0].display_box
    assert off[0].area == on[0].area


def test_detect_luminance_mode_finds_bright_blob():
    """Luminance detection finds a bright blob that a redness score would miss."""
    img = np.full((50, 50, 3), 100, np.uint8)
    img[20:30, 20:30] = (255, 255, 255)  # bright but not red, so redness scores zero
    assert redness_score(img)[25, 25] == 0  # redness alone would drop it
    dets, _ = detect(img, _params(luminance=True, threshold=200), disk=None)
    assert len(dets) == 1


def test_detect_otsu_mode_finds_blob_without_manual_threshold():
    """Otsu picks the threshold automatically and still isolates the blob."""
    img = np.full((50, 50, 3), 100, np.uint8)
    img[20:30, 20:30] = (0, 0, 255)
    dets, _ = detect(img, _params(use_otsu=True), disk=None)
    assert len(dets) == 1


def test_detect_hue_gate_keeps_orange_red_blob():
    """The hue gate passes an orange-red blob rather than dropping it."""
    img = np.full((50, 50, 3), 100, np.uint8)
    img[20:30, 20:30] = (0, 0, 255)  # pure red, hue sits in the orange-red band
    dets, _ = detect(img, _params(hue_gate=True), disk=None)
    assert len(dets) == 1
