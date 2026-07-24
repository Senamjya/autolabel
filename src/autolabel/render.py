"""Review-image rendering: overlays, crop contact sheets, the footer, and the histogram."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import (
    BG,
    COLS,
    CREDIT_BAND,
    CROP,
    FONT,
    GAP,
    GREEN,
    TILE,
    ZOOM,
    Params,
)
from .provenance import footer_text


def draw_footer(img: np.ndarray) -> None:
    """Draw the small credit caption in the bottom-left corner, in place.

    Dims a tight box behind the text (as the crop tiles do for their labels) so the
    single pass of light ink stays legible over both light and dark backgrounds.
    """
    text = footer_text()
    scale, thickness, pad = 0.4, 1, 4
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = 8, img.shape[0] - 8  # text baseline
    x0, y0 = max(x - pad, 0), max(y - th - pad, 0)
    x1, y1 = min(x + tw + pad, img.shape[1]), min(y + base + pad, img.shape[0])
    img[y0:y1, x0:x1] = (0.4 * img[y0:y1, x0:x1]).astype(np.uint8)
    cv2.putText(img, text, (x, y), FONT, scale, (240, 240, 240), thickness, cv2.LINE_AA)


def write_overlay(
    img_bgr: np.ndarray,
    boxes,
    dst: Path,
    p: Params,
    disk: np.ndarray | None,
    credit: bool = False,
) -> None:
    """Save a full-frame copy with green detection boxes and the orange filter outline.

    `disk` is the precomputed filter mask (or None). The outline is drawn only when the
    filter is in use and a disk was found. The credit footer is drawn when `credit`
    is True.
    """
    vis = img_bgr.copy()
    if p.mask_filter and disk is not None:
        contours, _ = cv2.findContours(disk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 180, 255), 3)
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), GREEN, 2)
    if credit:
        draw_footer(vis)
    cv2.imwrite(str(dst), vis)


def crop_tile(
    img: np.ndarray, box: tuple[int, int, int, int], label: str = ""
) -> np.ndarray:
    """Zoomed crop centred on `box`, with the detection rectangle and optional size label."""
    H, W = img.shape[:2]
    x, y, w, h = box
    cx, cy = x + w // 2, y + h // 2
    # window of CROP px, shifted to stay inside the image
    x0 = min(max(cx - CROP // 2, 0), max(W - CROP, 0))
    y0 = min(max(cy - CROP // 2, 0), max(H - CROP, 0))
    win = img[y0 : y0 + CROP, x0 : x0 + CROP]
    tile = cv2.resize(win, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    # box position within the zoomed tile
    bx, by = (x - x0) * ZOOM, (y - y0) * ZOOM
    cv2.rectangle(tile, (bx, by), (bx + w * ZOOM, by + h * ZOOM), GREEN, 2)
    if label:
        # shrink the font if needed so the whole label fits the tile width
        scale = 0.6
        (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
        if tw > TILE - 16:
            scale *= (TILE - 16) / tw
            (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
        # dim a banner across the top so white text stays legible over any crop colour,
        # then white ink. The detection box keeps green to itself
        band = th + 12
        tile[:band] = (0.4 * tile[:band]).astype(np.uint8)
        cv2.putText(
            tile, label, (8, th + 6), FONT, scale, (245, 245, 245), 1, cv2.LINE_AA
        )
    return tile


def contact_sheet(tiles: list[np.ndarray], credit: bool = False) -> np.ndarray:
    """Arrange crop tiles into a padded grid on the dark background.

    With credit on, a short dark strip is appended below the grid and the credit
    footer is drawn into it, so the caption sits on clean background and never
    overlaps a tile.
    """
    n = len(tiles)
    rows = (n + COLS - 1) // COLS
    band = CREDIT_BAND if credit else 0
    sheet_w = COLS * TILE + (COLS + 1) * GAP
    sheet_h = rows * TILE + (rows + 1) * GAP + band
    sheet = np.full((sheet_h, sheet_w, 3), BG, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, COLS)
        y = GAP + r * (TILE + GAP)
        x = GAP + c * (TILE + GAP)
        sheet[y : y + TILE, x : x + TILE] = tile
    if credit:
        draw_footer(sheet)
    return sheet


def write_size_histogram(diams_um: list[float], dst: Path) -> None:
    """Render a size-distribution histogram (equivalent diameter) to a PNG.

    Bins are log-spaced: particle sizes span more than a decade and are right-skewed,
    so linear bins would crush the bulk into the first few columns. One hue, a dashed
    median marker, plain count/diameter axes. Drawn with cv2 to avoid a plotting dep.
    """
    a = np.asarray(diams_um, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:  # single value (or all equal): give the axis a little width
        lo, hi = lo * 0.9, hi * 1.1 or 1.0

    W, H = 900, 520
    L, R, T, B = 74, 30, 58, 66  # plot-area margins
    pw, ph = W - L - R, H - T - B
    ink, muted = (232, 236, 236), (120, 130, 130)
    bar, accent = (200, 170, 70), (60, 150, 240)  # teal bars, orange median (BGR)
    canvas = np.full((H, W, 3), BG, np.uint8)

    nbins = int(np.clip(np.ceil(np.sqrt(a.size)) + 2, 6, 24))
    edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    counts, _ = np.histogram(a, bins=edges)
    cmax = int(counts.max()) or 1

    def x_of(v: float) -> int:
        """Log-scaled x pixel of a diameter value."""
        return int(
            L + (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) * pw
        )

    def y_of(c: float) -> int:
        """Y pixel of a count."""
        return int(T + ph - (c / cmax) * ph)

    # y grid + count ticks
    ystep = max(1, int(np.ceil(cmax / 5)))
    for c in range(0, cmax + 1, ystep):
        y = y_of(c)
        cv2.line(canvas, (L, y), (L + pw, y), muted, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(c), (L - 28, y + 5), FONT, 0.45, ink, 1, cv2.LINE_AA)

    # bars, 2px surface gap between them
    for i, count in enumerate(counts):
        c = int(count)
        if c == 0:
            continue
        x0, x1 = x_of(float(edges[i])) + 1, x_of(float(edges[i + 1])) - 1
        cv2.rectangle(canvas, (x0, y_of(c)), (x1, T + ph), bar, -1)

    # x ticks at 'nice' round diameters that fall inside the range
    for v in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if lo <= v <= hi:
            x = x_of(v)
            cv2.line(canvas, (x, T + ph), (x, T + ph + 5), muted, 1, cv2.LINE_AA)
            cv2.putText(
                canvas, str(v), (x - 12, T + ph + 22), FONT, 0.45, ink, 1, cv2.LINE_AA
            )

    # dashed median marker
    med = float(np.median(a))
    xm = x_of(med)
    for y in range(T, T + ph, 10):
        cv2.line(canvas, (xm, y), (xm, min(y + 5, T + ph)), accent, 1, cv2.LINE_AA)
    # dark outline first so the label stays legible when it crosses a bar
    cv2.putText(
        canvas, f"median {med:.0f} um", (xm + 6, T + 16), FONT, 0.45, BG, 3, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        f"median {med:.0f} um",
        (xm + 6, T + 16),
        FONT,
        0.45,
        accent,
        1,
        cv2.LINE_AA,
    )

    # title + axis labels
    cv2.putText(
        canvas,
        f"Particle size distribution (n={a.size})",
        (L, 34),
        FONT,
        0.7,
        ink,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "equivalent diameter (um, log scale)",
        (L + pw // 2 - 130, H - 16),
        FONT,
        0.5,
        ink,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "count", (L - 58, T - 10), FONT, 0.5, ink, 1, cv2.LINE_AA
    )  # atop the y-axis
    cv2.imwrite(str(dst), canvas)
