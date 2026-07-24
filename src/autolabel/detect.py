"""Pixel scoring, filter-disk masking, and the two-stage detection pipeline."""

from __future__ import annotations

import cv2
import numpy as np

from .config import (
    BLUE_RISE_LIMIT,
    DISK_MARGIN,
    HUE_HIGH_MIN,
    HUE_LOW_MAX,
    Params,
)
from .measure import Detection, measure_blob, split_by_cores


def redness_score(img_bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel orange/red excess over the paper's median colour, clipped to 0-255.

    Subtracts the paper's median B/G/R (taken over `mask`, or the whole frame if None)
    from every pixel, then scores each pixel's red rise minus the larger of its green/blue
    rise. Pixels whose blue rises past BLUE_RISE_LIMIT are zeroed as specular glints.
    """
    b, g, r = (c.astype(np.float32) for c in cv2.split(img_bgr))
    sel = mask > 0 if mask is not None else np.ones(b.shape, dtype=bool)
    d_b, d_g, d_r = b - np.median(b[sel]), g - np.median(g[sel]), r - np.median(r[sel])

    score = d_r - np.maximum(np.maximum(d_g, d_b), 0)
    # zero pixels whose blue rises past the limit
    score = np.where(d_b < BLUE_RISE_LIMIT, score, 0)
    return np.clip(score, 0, 255).astype(np.uint8)


def luminance_score(img_bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel brightness (brightest channel, max over B/G/R), 0-255.

    Colour-agnostic alternative to redness_score for cyan/green-dominant batches; pair
    with a high --threshold (~230). `mask` is accepted for signature parity but unused.
    """
    return img_bgr.max(axis=2).astype(np.uint8)


def hue_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Binary mask (255 = keep) of pixels whose hue lies in the orange-red band.

    OpenCV stores hue as 0-179 and red straddles the wrap-around, so the band is the
    union of [0, HUE_LOW_MAX] and [HUE_HIGH_MIN, 179]. Gates out warm-but-not-orange
    specks (greenish or faint) that the redness score alone would let through.
    """
    hue = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 0]
    in_band = (hue <= HUE_LOW_MAX) | (hue >= HUE_HIGH_MIN)
    return np.where(in_band, 255, 0).astype(np.uint8)


def find_filter_mask(img_bgr: np.ndarray) -> np.ndarray | None:
    """Mask of the filter paper (255 = on paper), or None if no disk is found.

    Otsu-thresholds the HSV value channel, takes the largest bright connected component,
    fills its outer contour, and erodes the result by DISK_MARGIN.
    """
    value = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    _, bright = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if num <= 1:
        return None
    # label 0 is background, so pick the largest of the rest
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 0.05 * value.size:
        return None  # largest component covers under 5% of the frame

    blob = np.where(labels == largest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    mask = np.zeros(value.shape, np.uint8)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DISK_MARGIN, DISK_MARGIN))
    return cv2.erode(mask, kernel)


def detect(
    img_bgr: np.ndarray, p: Params, disk: np.ndarray | None
) -> tuple[list[Detection], np.ndarray]:
    """Return (detections, binary_mask). See Detection for the per-particle fields.

    `disk` is the filter mask from find_filter_mask, or None. When mask_filter is off it
    is ignored.

    Two-stage hysteresis: the strict `threshold` picks seeds, then each seed grows to its
    connected region above `grow_threshold` to cover the particle's faded edge (0 disables
    growing). Seeds must sit on the disk. The grown region may extend past it, so
    edge-hugging particles keep their full size.
    """
    if not p.mask_filter:
        disk = None
    score = (
        luminance_score(img_bgr, disk) if p.luminance else redness_score(img_bgr, disk)
    )

    def threshold_and_clean(cutoff: int) -> np.ndarray:
        """Binary mask of score >= cutoff (Otsu if 0), hue-gated and opened."""
        if cutoff > 0:
            _, mask = cv2.threshold(score, cutoff, 255, cv2.THRESH_BINARY)
        else:
            _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if p.hue_gate:
            mask = cv2.bitwise_and(mask, hue_mask(img_bgr))
        if p.open_ksize > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (p.open_ksize, p.open_ksize)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    seeds = threshold_and_clean(0 if p.use_otsu else p.threshold)
    if disk is not None:
        seeds = cv2.bitwise_and(seeds, disk)

    # grow path is not disk-masked (see docstring). Seeds stay disk-masked
    binary = threshold_and_clean(p.grow_threshold) if p.grow_threshold > 0 else seeds

    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # keep only the grown blobs that contain at least one seed pixel
    seeded = set(np.unique(labels[seeds > 0])) - {0}
    split = p.split_touching and p.grow_threshold > 0
    dets: list[Detection] = []
    for i in sorted(seeded):
        x, y, bw, bh = (int(v) for v in stats[i, :4])
        comp = labels[y : y + bh, x : x + bw] == i
        comp_seeds = ((seeds[y : y + bh, x : x + bw] > 0) & comp).astype(np.uint8)
        # one label per particle: the whole blob, or split at the neck between cores
        parts = split_by_cores(comp, comp_seeds, split)
        for part_id in np.unique(parts):
            if part_id == 0:
                continue
            det = measure_blob(
                (parts == part_id).astype(np.uint8), (x, y), binary.shape, p
            )
            if det is not None:
                dets.append(det)
    return dets, binary
