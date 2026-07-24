"""Command-line front-end: argument parsing and the per-run orchestration."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import cv2

from .config import (
    CROPS_DIR,
    FILTER_DIAMETER_MM,
    IMAGE_SUFFIXES,
    OVERLAYS_DIR,
    REPORT_TXT,
    RUN_JSON,
    SIZE_HIST,
    SIZES_CSV,
    Params,
)
from .detect import detect, find_filter_mask
from .measure import disk_diameter_px, equiv_diameter_um
from .provenance import (
    __version__,
    _VersionAction,
    run_metadata,
    write_run_json,
)
from .render import contact_sheet, crop_tile, write_overlay, write_size_histogram
from .report import (
    print_scale_summary,
    scale_summary_lines,
    size_summary_lines,
    write_report,
    write_sizes_csv,
)


def process(args: argparse.Namespace) -> int:
    """Detect and measure particles in every image, writing crops, sizes and overlays.

    Returns a process exit code (0 on success, 1 if no images were found).
    """
    # Build Params straight from the parsed args: field name == flag dest for all but
    # the flags whose CLI spelling differs (--auto-threshold, --no-mask-filter,
    # --no-split).
    overrides = {
        "use_otsu": args.auto_threshold,
        "mask_filter": not args.no_mask_filter,
        "split_touching": not args.no_split,
    }
    p = Params(
        **{
            f.name: overrides[f.name] if f.name in overrides else getattr(args, f.name)
            for f in fields(Params)
        }
    )

    in_dir = Path(args.input)
    root = Path(args.out)
    overlays_dir = root / OVERLAYS_DIR
    crops_dir = root / CROPS_DIR
    # --scale-only measures the disk and reports scale. It writes nothing at all
    if not args.scale_only:
        root.mkdir(parents=True, exist_ok=True)
        if not args.no_overlays:
            overlays_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_crops:
            crops_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(f for f in in_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"No images found in {in_dir}", file=sys.stderr)
        return 1

    total = 0
    px_per_mm: list[float] = []  # one entry per image a disk was found in
    size_rows: list[dict] = []  # one row per particle, written to sizes.csv
    diams_um: list[float] = []  # every particle's equivalent diameter, for the summary
    # one (name, um_per_px, rows) per processed image, for the per-image report.txt
    image_reports: list[tuple[str, float | None, list[dict]]] = []
    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ! could not read {img_path.name}, skipping", file=sys.stderr)
            continue

        disk = find_filter_mask(img)
        # microns per px, from this image's own disk (None when no disk was found)
        um_per_px: float | None = None
        if disk is not None:
            scale = disk_diameter_px(disk) / FILTER_DIAMETER_MM
            px_per_mm.append(scale)
            um_per_px = 1000.0 / scale
            if args.scale_only:
                print(
                    f"  {img_path.name:45s} -> {scale:8.2f} px/mm  ({um_per_px:5.2f} um/px)"
                )
        elif args.scale_only:
            print(f"  {img_path.name:45s} -> no disk found")
        if args.scale_only:
            continue

        dets, _ = detect(img, p, disk)
        boxes = [d.display_box for d in dets]
        total += len(dets)
        print(f"  {img_path.name:45s} -> {len(dets):4d} suspected particle(s)")

        # per-particle size: micron columns need a scale, but aspect ratio does not
        tile_labels: list[str] = []
        img_rows: list[dict] = []  # this image's size rows, for the report section
        for i, d in enumerate(dets, start=1):
            x, y, bw, bh = d.bbox  # true particle box, no display padding
            diam_um = (
                equiv_diameter_um(d.area, um_per_px) if um_per_px is not None else None
            )
            if diam_um is not None:
                diams_um.append(diam_um)
            # orientation-free length/width (Feret). Aspect ratio is scale-independent
            aspect = d.length_px / d.width_px if d.width_px > 0 else None
            img_rows.append(
                {
                    "image": img_path.name,
                    "particle": i,
                    "x": x,
                    "y": y,
                    "w": bw,
                    "h": bh,
                    "area_px": d.area,
                    "um_per_px": f"{um_per_px:.4f}" if um_per_px is not None else "",
                    "equiv_diam_um": f"{diam_um:.1f}" if diam_um is not None else "",
                    "length_um": f"{d.length_px * um_per_px:.1f}"
                    if um_per_px is not None
                    else "",
                    "width_um": f"{d.width_px * um_per_px:.1f}"
                    if um_per_px is not None
                    else "",
                    "aspect_ratio": f"{aspect:.2f}" if aspect is not None else "",
                    "tool_version": __version__,
                }
            )
            tile_labels.append(
                f"diameter = {diam_um:.0f} um"
                if diam_um is not None
                else f"area = {d.area} px"
            )
        size_rows.extend(img_rows)
        image_reports.append((img_path.name, um_per_px, img_rows))

        credit = args.credit
        if not args.no_overlays:
            write_overlay(
                img,
                boxes,
                overlays_dir / f"{img_path.stem}_overlay.png",
                p,
                disk,
                credit,
            )
        if not args.no_crops and boxes:
            sheet = contact_sheet(
                [
                    crop_tile(img, b, lbl)
                    for b, lbl in zip(boxes, tile_labels, strict=True)
                ],
                credit=credit,
            )
            cv2.imwrite(str(crops_dir / f"{img_path.stem}_crops.png"), sheet)

    if args.scale_only:
        print_scale_summary(px_per_mm, len(images), p.min_area)
        return 0

    write_sizes_csv(size_rows, root / SIZES_CSV)
    write_report(
        root / REPORT_TXT,
        image_reports,
        px_per_mm,
        diams_um,
        total,
        len(images),
        p.min_area,
    )
    write_run_json(
        run_metadata(
            args, len(images), total, {name: um for name, um, _ in image_reports}
        ),
        root / RUN_JSON,
    )
    if diams_um:
        write_size_histogram(diams_um, root / SIZE_HIST)

    print(
        f"\nDone: {total} suspected particle{'s' if total != 1 else ''} "
        f"across {len(images)} image{'s' if len(images) != 1 else ''}.\n"
    )
    for line in scale_summary_lines(px_per_mm, len(images), p.min_area):
        print(line)
    for line in size_summary_lines(diams_um, len(diams_um), total):
        print(line)

    # aligned manifest of what landed under the output root
    outputs = [
        (SIZES_CSV, "per-particle size table"),
        (REPORT_TXT, "per-image report"),
        (RUN_JSON, "provenance record"),
    ]
    if diams_um:
        outputs.append((SIZE_HIST, "size distribution"))
    if not args.no_overlays:
        outputs.append((f"{OVERLAYS_DIR}/", "full-frame review images"))
    if not args.no_crops:
        outputs.append((f"{CROPS_DIR}/", "zoomed contact sheets"))
    width = max(len(name) for name, _ in outputs)
    print(f"\nWrote to {root}/:")
    for name, desc in outputs:
        print(f"  {name:<{width}}   {desc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    ap = argparse.ArgumentParser(
        prog="autolabel",
        description="An offline batch tool that detects and measures Nile red "
        "fluorescence microplastic particles on filter-paper photographs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", help="Folder of fluorescence images (PNG or TIFF)")

    detection = ap.add_argument_group("detection")
    detection.add_argument(
        "--threshold",
        type=int,
        default=Params.threshold,
        help="Minimum red excess over the paper for a pixel to count as plastic. "
        "Raise it if bare paper gets boxed, lower it if faint particles are missed",
    )
    detection.add_argument(
        "--auto-threshold",
        action="store_true",
        help="Choose the threshold automatically, overriding --threshold. A quick "
        "first pass, but it can misfire when very few particles are present",
    )
    detection.add_argument(
        "--luminance",
        action="store_true",
        help="Detect by brightness instead of redness, for cyan or green-dominant "
        "batches where particles do not glow red. Pair with a high --threshold "
        "(~230) and --grow-threshold 0",
    )
    detection.add_argument(
        "--grow-threshold",
        type=int,
        default=Params.grow_threshold,
        help="Grow each detection out to its connected region above this score, so a "
        "box covers the whole particle and not just its bright core (0 = off, which "
        "also disables splitting)",
    )
    detection.add_argument(
        "--hue-gate",
        action="store_true",
        help="Also require an orange-red hue. Redundant when an orange long-pass "
        "filter is fitted, so off by default",
    )

    cleanup = ap.add_argument_group("filtering and cleanup")
    cleanup.add_argument(
        "--open-ksize",
        type=int,
        default=Params.open_ksize,
        help="Size in pixels of the opening kernel that removes single-pixel noise "
        "(0 = off)",
    )
    cleanup.add_argument(
        "--min-area",
        type=int,
        default=Params.min_area,
        help="Drop blobs smaller than N pixels. Raise it if speckle noise gets boxed",
    )
    cleanup.add_argument(
        "--max-area",
        type=int,
        default=Params.max_area,
        help="Drop blobs larger than N pixels (0 = off). Set a cap if a huge artefact "
        "gets boxed",
    )
    cleanup.add_argument(
        "--no-split",
        action="store_true",
        help="Keep particles that grew together as a single box, instead of "
        "splitting them at the neck",
    )
    cleanup.add_argument(
        "--pad",
        type=int,
        default=Params.pad,
        help="Grow each box by N pixels. Raise it if boxes clip the fluorescence halo",
    )
    cleanup.add_argument(
        "--no-mask-filter",
        action="store_true",
        help="Keep detections that fall outside the filter disk, instead of "
        "restricting to it",
    )

    output = ap.add_argument_group("output")
    output.add_argument(
        "--out",
        default="out",
        help="Output root. crops/ and overlays/ go inside it, with sizes.csv and "
        "size_hist.png at the top",
    )
    output.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip the full-frame annotated overlay images (written to "
        "<out>/overlays/ by default)",
    )
    output.add_argument(
        "--no-crops",
        action="store_true",
        help="Skip the per-image zoomed contact sheets (written to <out>/crops/ by "
        "default)",
    )
    output.add_argument(
        "--credit",
        action="store_true",
        help="Stamp the author and version footer into the overlay and crop images",
    )

    info = ap.add_argument_group("info and early exit")
    info.add_argument(
        "--scale-only",
        action="store_true",
        help="Report px/mm and um/px from the filter disk, then exit without writing "
        "anything",
    )
    info.add_argument(
        "--version",
        action=_VersionAction,
        default=argparse.SUPPRESS,
        help="Print the tool version, author and ORCID, then exit",
    )
    return ap


def main() -> int:
    """Parse arguments and run the pipeline. This is the console-script entry point."""
    return process(build_parser().parse_args())
