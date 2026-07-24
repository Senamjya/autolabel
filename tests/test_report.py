"""Tests for the tabular and textual output: sizes.csv, report.txt, and summaries."""

import csv

from autolabel.provenance import __version__
from autolabel.report import (
    scale_summary_lines,
    size_summary_lines,
    write_report,
    write_sizes_csv,
)


def test_scale_summary_lines_report_scale_and_cutoff():
    """With disks found, the lines report median px/mm, um/px and the min_area cutoff."""
    lines = scale_summary_lines([30.0, 32.0], n_images=3, min_area=4)
    assert lines[0].startswith("Scale")
    assert "px/mm" in lines[0] and "um/px" in lines[0]
    assert "median over 2/3 images" in lines[0]
    assert lines[1].strip().startswith("min_area 4 px ~")


def test_scale_summary_lines_no_disk():
    """With no disks, a single explanatory line is returned (no crash on empty input)."""
    assert scale_summary_lines([], n_images=3, min_area=4) == [
        "No filter disk found in any image, so no scale can be reported."
    ]


def test_size_summary_lines_report_distribution():
    """The size line reports the n sized, median, range and mean."""
    lines = size_summary_lines([100.0, 200.0, 300.0], n_sized=3, n_total=5)
    assert lines[0].startswith("Sizes")
    assert "3/5 sized" in lines[0]
    assert "median 200 um" in lines[0]
    assert "range 100-300 um" in lines[0]


def test_size_summary_lines_none_sized():
    """With nothing sized, a single explanatory line is returned."""
    assert size_summary_lines([], n_sized=0, n_total=0) == [
        "Sizes: no particle could be sized (no filter disk found for scale)."
    ]


def _row(particle, area_px, equiv="", length="", width="", aspect=""):
    """A sizes.csv-shaped row dict, as process() builds and write_report() consumes."""
    return {
        "particle": particle,
        "area_px": area_px,
        "equiv_diam_um": equiv,
        "length_um": length,
        "width_um": width,
        "aspect_ratio": aspect,
    }


def _size_row():
    """A complete sizes.csv row dict as process() builds it."""
    return {
        "image": "S0001_NR.png",
        "particle": 1,
        "x": 10,
        "y": 10,
        "w": 5,
        "h": 5,
        "area_px": 20,
        "um_per_px": "31.5000",
        "equiv_diam_um": "160.0",
        "length_um": "200.0",
        "width_um": "120.0",
        "aspect_ratio": "1.67",
        "tool_version": __version__,
    }


def test_write_report_has_header_and_per_image_sections(tmp_path):
    """The report opens with the aggregate summary, then one section per image."""
    rows_a = [_row(1, 2076, "1615.0", "2449.8", "1243.9", "1.97")]
    image_reports = [
        ("A_NR.png", 31.4, rows_a),
        ("B_NR.png", 31.5, []),  # an image with no detections
    ]
    dst = tmp_path / "report.txt"
    write_report(
        dst,
        image_reports,
        px_per_mm=[31.4, 31.5],
        diams_um=[1615.0],
        total=1,
        n_images=2,
        min_area=4,
    )
    text = dst.read_text()

    # aggregate header, driven by the same summary builders as the console
    assert text.startswith("Autolabel particle report")
    assert "Images processed:    2" in text
    assert "Suspected particles: 1" in text
    assert "Scale   " in text and "median over 2/2 images" in text
    assert "Sizes   median" in text and "1/1 sized" in text

    # per-image sections: A carries a scale and a particle row, B is empty
    assert "A_NR.png  -  1 particle(s)  (31.40 um/px)" in text
    assert "1615.0" in text and "2449.8" in text and "1.97" in text
    assert "B_NR.png  -  0 particle(s)  (31.50 um/px)" in text
    assert "  (none)" in text
    assert text.endswith("\n")


def test_write_report_blanks_render_as_dash_without_scale(tmp_path):
    """An image with no disk scale has blank um cells. They render as '-', with no um/px tag."""
    rows = [_row(1, 120)]  # only area_px filled, micron columns blank
    dst = tmp_path / "report.txt"
    write_report(
        dst,
        [("C_NR.png", None, rows)],
        px_per_mm=[],
        diams_um=[],
        total=1,
        n_images=1,
        min_area=4,
    )
    text = dst.read_text()

    assert "C_NR.png  -  1 particle(s)\n" in text  # no "(.. um/px)" suffix
    assert "No filter disk found in any image" in text
    # the blank micron cells collapse to '-', the filled area_px stays
    particle_line = next(ln for ln in text.splitlines() if ln.strip().startswith("1"))
    assert "120" in particle_line
    assert particle_line.split().count("-") == 4  # equiv, length, width, aspect


def test_write_sizes_csv_tool_version_column(tmp_path):
    """sizes.csv ends with a tool_version column and parses with a bare csv reader."""
    dst = tmp_path / "sizes.csv"
    write_sizes_csv([_size_row(), _size_row()], dst)
    with dst.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        assert fieldnames is not None and fieldnames[-1] == "tool_version"
        rows = list(reader)
    assert rows and all(r["tool_version"] == __version__ for r in rows)
    assert dst.read_text().splitlines()[0].startswith("image,")  # no comment lines
