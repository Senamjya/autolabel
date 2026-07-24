"""Per-particle geometry: equivalent diameter, Feret axes, and the Detection record."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import DISK_MARGIN, Params


def disk_diameter_px(mask: np.ndarray) -> float:
    """Diameter (px) of the filter disk, from an eroded mask produced by find_filter_mask.

    Uses the area-equivalent diameter d = 2*sqrt(area/pi), which a ragged edge perturbs
    far less than a bounding box. The mask was eroded by an ellipse kernel of width
    DISK_MARGIN, pulling the boundary in by DISK_MARGIN // 2 per side, so
    2 * (DISK_MARGIN // 2) is added back.
    """
    area = float(np.count_nonzero(mask))
    return 2.0 * np.sqrt(area / np.pi) + 2.0 * (DISK_MARGIN // 2)


def equiv_diameter_um(area_px: float, um_per_px: float) -> float:
    """Area-equivalent diameter (um) of a blob: the diameter of a disk of the same area.

    d = 2*sqrt(area/pi) in px, scaled to microns. Orientation-free, unlike a bbox side.
    """
    return 2.0 * np.sqrt(area_px / np.pi) * um_per_px


def feret_diameters(contour: np.ndarray) -> tuple[float, float]:
    """Max and min Feret (caliper) diameters (px) of a blob contour.

    The extreme widths of the shape over all orientations: max is the particle length,
    min its width. Both are orientation-free. Returns (0, 0) for a degenerate contour.
    """
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    if len(hull) < 2:
        return 0.0, 0.0
    # max Feret: the farthest-apart pair of hull vertices
    diffs = hull[:, None, :] - hull[None, :, :]
    d_max = float(np.sqrt((diffs**2).sum(-1)).max())
    # min Feret: the smallest perpendicular span of all vertices across any hull edge
    d_min = float("inf")
    n = len(hull)
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        length = float(np.hypot(edge[0], edge[1]))
        if length == 0:
            continue
        normal = np.array([-edge[1], edge[0]]) / length
        proj = hull @ normal
        d_min = min(d_min, float(proj.max() - proj.min()))
    return d_max, (d_min if np.isfinite(d_min) else 0.0)


@dataclass(frozen=True)
class Detection:
    """One suspected particle.

    bbox: unpadded pixel box (x, y, w, h).
    display_box: bbox grown by pad, for overlays and crops.
    area: pixel count.
    length_px, width_px: max/min Feret diameters.
    """

    bbox: tuple[int, int, int, int]
    display_box: tuple[int, int, int, int]
    area: int
    length_px: float
    width_px: float


def split_by_cores(comp: np.ndarray, comp_seeds: np.ndarray, split: bool) -> np.ndarray:
    """Label a grown blob's pixels 1..K by nearest bright core, or all 1 if not splitting.

    Two particles that grew together share one connected halo. With split on and more than
    one core inside, each pixel is assigned to its nearest core (a Euclidean Voronoi split
    within the blob), cutting them apart at the neck. `comp` and `comp_seeds` are the blob
    mask and the seed cores inside it. Returns an int32 label image (0 outside the blob).
    """
    if split:
        n, _ = cv2.connectedComponents(comp_seeds, connectivity=8)
        if n > 2:  # background plus at least two cores
            inv = np.where(comp_seeds > 0, 0, 255).astype(np.uint8)
            _, nearest = cv2.distanceTransformWithLabels(
                inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP
            )
            return np.where(comp, nearest, 0).astype(np.int32)
    return comp.astype(np.int32)


def measure_blob(
    mask: np.ndarray, origin: tuple[int, int], shape: tuple[int, int], p: Params
) -> Detection | None:
    """Build a Detection from one particle mask, or None if it fails the area filters.

    `mask` is one particle in crop coordinates, `origin` is that crop's (x, y) in the
    full frame, and `shape` is the full (H, W).
    """
    area = int(mask.sum())
    if area < p.min_area or (p.max_area and area > p.max_area):
        return None
    height, width = shape
    ox, oy = origin
    bx, by, bw, bh = cv2.boundingRect(mask)
    ax, ay = ox + bx, oy + by
    x0, y0 = max(0, ax - p.pad), max(0, ay - p.pad)
    x1, y1 = min(width, ax + bw + p.pad), min(height, ay + bh + p.pad)
    contours, _ = cv2.findContours(
        mask[by : by + bh, bx : bx + bw], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    length_px, width_px = feret_diameters(
        max(contours, key=lambda c: cv2.contourArea(c))
    )
    return Detection(
        bbox=(ax, ay, bw, bh),
        display_box=(x0, y0, x1 - x0, y1 - y0),
        area=area,
        length_px=length_px,
        width_px=width_px,
    )
