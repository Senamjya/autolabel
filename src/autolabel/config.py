"""Tunable constants and the detection Params dataclass."""

from __future__ import annotations

from dataclasses import dataclass

import cv2

IMAGE_SUFFIXES = {".png", ".tif", ".tiff"}

# fixed subfolder names under the output root
CROPS_DIR = "crops"
OVERLAYS_DIR = "overlays"
SIZES_CSV = "sizes.csv"  # per-particle size table at the output root
SIZE_HIST = "size_hist.png"  # size-distribution histogram at the output root
REPORT_TXT = "report.txt"  # human-readable per-image size report at the output root
RUN_JSON = "run.json"  # machine-readable provenance record at the output root

BLUE_RISE_LIMIT = 40  # blue rise over the paper above which a pixel is zeroed
DISK_MARGIN = 61  # ellipse kernel (px) the filter mask is eroded by

# physical diameter of the filter paper disk, for px -> micron scale
FILTER_DIAMETER_MM = 70.0

# orange-red hue band, in OpenCV's 0-179 hue scale. Red wraps, so it is two ranges
HUE_LOW_MAX = 20  # upper edge of the low (orange) half of the band
HUE_HIGH_MIN = 160  # lower edge of the high (magenta-red) half of the band

GREEN = (0, 255, 0)  # detection box colour, in overlays and crop tiles
FONT = cv2.FONT_HERSHEY_SIMPLEX

CROP = 110  # source crop size (px) around each particle, before zoom
ZOOM = 2  # magnification for the contact-sheet tiles
COLS = 8  # tiles per row in the contact sheet
GAP = 6  # px gap between tiles
BG = (14, 23, 25)  # dark contact-sheet background (BGR)
CREDIT_BAND = 24  # height (px) of the footer strip below a credited contact sheet

TILE = CROP * ZOOM  # rendered tile size (px)


@dataclass
class Params:
    """Parameters that control what gets labelled."""

    threshold: int = 100  # min excess red over the paper to mark a pixel plastic
    use_otsu: bool = False  # threshold the redness score with Otsu instead
    grow_threshold: int = (
        25  # grow each seed to its connected region above this (0 = off)
    )
    luminance: bool = False  # detect by brightness (max channel) instead of redness
    hue_gate: bool = False  # also require the pixel's hue to sit in the orange-red band
    min_area: int = 4  # drop blobs smaller than this many pixels
    max_area: int = 0  # drop blobs larger than this (0 = no upper limit)
    open_ksize: int = 3  # morphological opening kernel (0 = off)
    pad: int = 1  # grow each box by this many px
    mask_filter: bool = True  # keep only detections on the filter disk
    split_touching: bool = True  # split blobs that grew together, one per bright core
