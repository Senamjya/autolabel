"""Tabular and textual output: sizes.csv, report.txt, and the console summaries."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .config import FILTER_DIAMETER_MM


def scale_summary_lines(
    px_per_mm: list[float], n_images: int, min_area: int
) -> list[str]:
    """Lines describing the median image scale and the min_area cutoff in microns."""
    if not px_per_mm:
        return ["No filter disk found in any image, so no scale can be reported."]
    median_px_mm = float(np.median(px_per_mm))
    median_um_px = 1000.0 / median_px_mm
    # a blob of N px has equivalent diameter 2*sqrt(N/pi) px
    min_diam_um = 2.0 * np.sqrt(min_area / np.pi) * median_um_px
    return [
        (
            f"Scale   {median_px_mm:.2f} px/mm, {median_um_px:.2f} um/px   "
            f"(median over {len(px_per_mm)}/{n_images} images, {FILTER_DIAMETER_MM:g} mm disk)"
        ),
        f"        min_area {min_area} px ~ {min_diam_um:.0f} um",
    ]


def print_scale_summary(px_per_mm: list[float], n_images: int, min_area: int) -> None:
    """Print the median image scale and what the min_area cutoff means in microns."""
    print("\n" + "\n".join(scale_summary_lines(px_per_mm, n_images, min_area)))


def write_sizes_csv(rows: list[dict], dst: Path) -> None:
    """Write the per-particle size table. One row per detection, micron columns blank without a disk."""
    fields = [
        "image",
        "particle",
        "x",
        "y",
        "w",
        "h",
        "area_px",
        "um_per_px",
        "equiv_diam_um",
        "length_um",
        "width_um",
        "aspect_ratio",
        "tool_version",
    ]
    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    dst: Path,
    image_reports: Sequence[tuple[str, float | None, list[dict]]],
    px_per_mm: list[float],
    diams_um: list[float],
    total: int,
    n_images: int,
    min_area: int,
) -> None:
    """Write the human-readable per-image size report.

    Opens with the same aggregate scale and size summary shown on the console, then
    one section per image listing each particle's area and (when a disk gave a scale)
    its equivalent diameter, Feret length/width and aspect ratio. `image_reports` is
    one (name, um_per_px, rows) tuple per processed image, rows being the sizes.csv dicts.
    """
    lines = ["Autolabel particle report", "=" * 25, ""]
    lines.append(f"Images processed:    {n_images}")
    lines.append(f"Suspected particles: {total}")
    lines.append("")
    lines.extend(scale_summary_lines(px_per_mm, n_images, min_area))
    lines.append("")
    lines.extend(size_summary_lines(diams_um, len(diams_um), total))
    lines += ["", "Per-image detail", "-" * 16, ""]

    cols = ("#", "area_px", "equiv_um", "length_um", "width_um", "aspect")
    widths = (4, 9, 9, 10, 9, 7)
    header = "  ".join(f"{c:>{w}}" for c, w in zip(cols, widths, strict=True))
    for name, um_per_px, rows in image_reports:
        scale = f"  ({um_per_px:.2f} um/px)" if um_per_px is not None else ""
        lines.append(f"{name}  -  {len(rows)} particle(s){scale}")
        if not rows:
            lines += ["  (none)", ""]
            continue
        lines.append("  " + header)
        for row in rows:
            cells = (
                str(row["particle"]),
                str(row["area_px"]),
                row["equiv_diam_um"] or "-",
                row["length_um"] or "-",
                row["width_um"] or "-",
                row["aspect_ratio"] or "-",
            )
            lines.append(
                "  "
                + "  ".join(f"{c:>{w}}" for c, w in zip(cells, widths, strict=True))
            )
        lines.append("")
    dst.write_text("\n".join(lines) + "\n")


def size_summary_lines(diams_um: list[float], n_sized: int, n_total: int) -> list[str]:
    """Lines describing the particle size distribution over sized particles."""
    if not diams_um:
        return ["Sizes: no particle could be sized (no filter disk found for scale)."]
    a = np.array(diams_um)
    return [
        (
            f"Sizes   median {np.median(a):.0f} um, range {a.min():.0f}-{a.max():.0f} um, "
            f"mean {a.mean():.0f} um   ({n_sized}/{n_total} sized)"
        ),
    ]
