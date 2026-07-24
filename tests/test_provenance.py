"""Tests for authorship metadata and the run.json provenance record."""

from pathlib import Path

from autolabel.cli import build_parser
from autolabel.provenance import AUTHOR, ORCID, __version__, run_metadata

REPO_ROOT = Path(__file__).parent.parent


def _pyproject_version():
    """The project version string from pyproject.toml, parsed without tomllib (3.10)."""
    for line in (REPO_ROOT / "pyproject.toml").read_text().splitlines():
        if line.strip().startswith("version") and "=" in line:
            return line.split("=", 1)[1].strip().strip('"')
    raise AssertionError("no version line found in pyproject.toml")


def test_pyproject_version_mirrors_dunder():
    """pyproject.toml's version matches the authoritative __version__."""
    assert _pyproject_version() == __version__


def test_run_metadata_has_expected_keys():
    """run_metadata records the full provenance set with a null-safe per-image scale."""
    args = build_parser().parse_args(["some/input"])
    meta = run_metadata(
        args, n_images=2, total=5, um_per_px_by_image={"a.png": 1.5, "b.png": None}
    )
    assert set(meta) == {
        "tool",
        "version",
        "author",
        "orcid",
        "affiliation",
        "timestamp_utc",
        "input_dir",
        "arguments",
        "image_count",
        "particle_count",
        "um_per_px",
    }
    assert meta["version"] == __version__
    assert meta["author"] == AUTHOR and meta["orcid"] == ORCID
    assert meta["image_count"] == 2 and meta["particle_count"] == 5
    assert meta["um_per_px"]["b.png"] is None  # no disk stays null, not an error
    assert meta["timestamp_utc"].endswith("Z")  # UTC
