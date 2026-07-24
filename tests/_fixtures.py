"""Shared synthetic-image builders and helpers for the test modules.

Fixtures are deterministic: organic blobs from seeded noise, so ground truth can be
derived from the rendered mask rather than hardcoded.
"""

import dataclasses

import cv2
import numpy as np

from autolabel.config import Params

ORANGE = (0, 140, 255)  # BGR, for annotations
CYAN = (255, 200, 0)
WHITE = (245, 245, 245)


def _blob_field(shape, seed, sigma_x, sigma_y, env_x, env_y):
    """A smooth, centred, organic intensity field in [0, 1], deterministic per seed.

    Gaussian-blurred noise gives the irregular edge. A centred Gaussian envelope keeps the
    blob off the frame border.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    f = cv2.GaussianBlur(
        rng.random(shape).astype(np.float32), (0, 0), sigma_x, sigmaY=sigma_y
    )
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    env = np.exp(-(((xx - w / 2) / env_x) ** 2 + ((yy - h / 2) / env_y) ** 2))
    f *= env
    return (f - f.min()) / (f.max() - f.min() + 1e-9)


def _largest_component(mask):
    """Largest 8-connected component of a mask, as uint8 0/1."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return np.zeros(mask.shape, np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def _compact_blob():
    """A small irregular solid-red blob on paper-grey, for box-geometry checks."""
    f = _blob_field((48, 48), seed=5, sigma_x=4, sigma_y=4, env_x=9, env_y=9)
    mask = _largest_component(f >= 0.5)
    img = np.full((48, 48, 3), 100, np.uint8)
    img[..., 2] = np.where(mask > 0, 255, 100)
    return img, mask


def _graded_blob():
    """An irregular blob with a bright core fading to a dim halo, like a real particle."""
    f = _blob_field((60, 60), seed=3, sigma_x=7, sigma_y=9, env_x=16, env_y=16)
    mask = _largest_component(f >= 0.35)
    fn = np.clip((f - 0.35) / 0.65, 0.0, 1.0)  # 0 at the edge, 1 at the peak
    img = np.full((60, 60, 3), 100, np.uint8)
    img[..., 2] = np.where(mask > 0, (100 + fn * 155).astype(np.uint8), 100)
    return img


def _fibre_blob():
    """An elongated, tapered solid-red fibre on paper-grey."""
    f = _blob_field((64, 44), seed=2, sigma_x=1.7, sigma_y=6, env_x=4.5, env_y=24)
    mask = _largest_component(f >= 0.38)
    img = np.full((64, 44, 3), 100, np.uint8)
    img[..., 2] = np.where(mask > 0, 255, 100)
    return img, mask


def _two_touching_blobs():
    """Two graded blobs 36 px apart whose halos overlap into one grown region."""
    h, w = 80, 120
    yy, xx = np.mgrid[0:h, 0:w]
    red = np.full((h, w), 100.0)
    for cx in (w // 2 - 18, w // 2 + 18):
        g = np.exp(-(((xx - cx) / 18.0) ** 2 + ((yy - 40) / 18.0) ** 2))
        red = np.maximum(red, 100 + g * 155)
    img = np.full((h, w, 3), 100, np.uint8)
    img[..., 2] = np.clip(red, 0, 255).astype(np.uint8)
    return img


def _disk_image(radius: int, size: int = 400) -> np.ndarray:
    """A white filled disk of the given radius on a black frame."""
    img = np.zeros((size, size, 3), np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, (255, 255, 255), -1)
    return img


def _params(**over) -> Params:
    """Detection Params with test-friendly defaults, overridable by keyword."""
    base = Params(
        threshold=100,
        grow_threshold=0,
        min_area=1,
        open_ksize=0,
        pad=2,
        mask_filter=False,
    )
    return dataclasses.replace(base, **over)


def _imread(path):
    """cv2.imread that fails loudly instead of returning None on a missing file."""
    img = cv2.imread(str(path))
    assert img is not None
    return img


def _rotate_fixture(mask, degrees):
    """Rotate a mask about its centre and re-render it as a red-on-paper fixture."""
    h, w = mask.shape
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    m = (cv2.warpAffine(mask * 255, rot, (w, h), flags=cv2.INTER_NEAREST) > 127).astype(
        np.uint8
    )
    out = np.full((h, w, 3), 100, np.uint8)
    out[..., 2] = np.where(m > 0, 255, 100)
    return out, m
