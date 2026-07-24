"""Integration tests for the CLI orchestration: run.json, sizes.csv, and the footer toggle."""

import csv
import json

import cv2
import numpy as np

from _fixtures import _disk_image, _imread
from autolabel.cli import build_parser, process
from autolabel.config import CREDIT_BAND, CROPS_DIR, OVERLAYS_DIR, RUN_JSON, SIZES_CSV
from autolabel.provenance import __version__


def _make_input_dir(root):
    """A folder with one PNG holding a red particle on grey (detectable, no disk)."""
    d = root / "in"
    d.mkdir(parents=True)
    img = np.full((200, 200, 3), 100, np.uint8)
    img[90:104, 90:104] = (0, 0, 255)  # a red square, well above the redness threshold
    cv2.imwrite(str(d / "sample.png"), img)
    return d


def _run(root, *extra):
    """Run process() on a one-image input dir under root and return the output root."""
    in_dir = _make_input_dir(root)
    out = root / "out"
    args = build_parser().parse_args([str(in_dir), "--out", str(out), *extra])
    assert process(args) == 0
    return out


def test_process_writes_run_json(tmp_path):
    """A run writes run.json with the provenance keys and this run's argument set."""
    out = _run(tmp_path)
    meta = json.loads((out / RUN_JSON).read_text())
    assert meta["version"] == __version__
    assert meta["image_count"] == 1 and meta["particle_count"] >= 1
    assert meta["arguments"]["no_overlays"] is False
    assert "sample.png" in meta["um_per_px"]


def test_process_sizes_csv_carries_tool_version(tmp_path):
    """Every sizes.csv row from a run carries the tool version."""
    out = _run(tmp_path)
    with (out / SIZES_CSV).open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and all(r["tool_version"] == __version__ for r in rows)


def test_credit_adds_footer_default_omits_it(tmp_path):
    """--credit footers the crop sheet (taller by CREDIT_BAND). The default does not."""
    credited = _run(tmp_path / "a", "--credit")
    plain = _run(tmp_path / "b")
    sheet_c = _imread(credited / CROPS_DIR / "sample_crops.png")
    sheet_p = _imread(plain / CROPS_DIR / "sample_crops.png")
    assert sheet_c.shape[0] == sheet_p.shape[0] + CREDIT_BAND


def test_overlay_footer_toggles_bottom_left(tmp_path):
    """The footer changes the overlay's bottom-left. The default leaves it clean."""
    ov_c = _imread(
        _run(tmp_path / "a", "--credit") / OVERLAYS_DIR / "sample_overlay.png"
    )
    ov_p = _imread(_run(tmp_path / "b") / OVERLAYS_DIR / "sample_overlay.png")
    assert ov_c.shape == ov_p.shape
    h = ov_c.shape[0]
    corner = (slice(h - 30, h), slice(0, 300))
    assert not np.array_equal(ov_c[corner], ov_p[corner])  # footer only when credited


def test_scale_only_reports_and_writes_nothing(tmp_path):
    """--scale-only reports the disk scale and exits without creating the output tree."""
    d = tmp_path / "in"
    d.mkdir()
    cv2.imwrite(str(d / "disk.png"), _disk_image(radius=120))
    out = tmp_path / "out"
    args = build_parser().parse_args([str(d), "--out", str(out), "--scale-only"])
    assert process(args) == 0
    assert not out.exists()  # scale-only writes nothing


def test_process_skips_unreadable_image(tmp_path):
    """A file that cannot be decoded is skipped, and the run still completes."""
    d = tmp_path / "in"
    d.mkdir()
    good = np.full((200, 200, 3), 100, np.uint8)
    good[90:104, 90:104] = (0, 0, 255)
    cv2.imwrite(str(d / "good.png"), good)
    (d / "bad.png").write_bytes(b"not a real png")
    out = tmp_path / "out"
    args = build_parser().parse_args([str(d), "--out", str(out)])
    assert process(args) == 0
    with (out / SIZES_CSV).open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert any(r["image"] == "good.png" for r in rows)
    assert not any(r["image"] == "bad.png" for r in rows)
