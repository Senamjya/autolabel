"""Authorship metadata and the provenance recorded in every run.

This module is the single source of the tool version, author and ORCID used by
the --version flag, the image footers and run.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

# Authoritative version. pyproject.toml mirrors this.
__version__ = "1.0.0"

TOOL_NAME = "autolabel"
AUTHOR = "Senam Julian Yao Asmussen"
ORCID = "https://orcid.org/0009-0001-4885-4801"
AFFILIATION = "Norwegian University of Science and Technology (NTNU)"


def version_text() -> str:
    """Multi-line name, version, author and ORCID, for the --version flag."""
    return f"{TOOL_NAME} {__version__}\n{AUTHOR}, {AFFILIATION}\n{ORCID}"


class _VersionAction(argparse.Action):
    """Print version_text() verbatim and exit, preserving its line breaks.

    argparse's built-in version action reflows the string and collapses newlines.
    This prints it as written.
    """

    def __init__(self, option_strings, dest, **kwargs):
        """Register as a zero-argument flag."""
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Print the version block to stdout and exit cleanly."""
        print(version_text())
        parser.exit()


def footer_text() -> str:
    """The credit caption rendered into overlay and crop images."""
    return f"{TOOL_NAME} v{__version__} - {AUTHOR}, NTNU"


def run_metadata(
    args: argparse.Namespace,
    n_images: int,
    total: int,
    um_per_px_by_image: dict[str, float | None],
) -> dict:
    """Assemble the provenance record written to run.json.

    Records who produced the run and which version, the UTC time, the resolved CLI
    arguments, image and particle counts, and each image's micron scale (null with
    no disk).
    """
    return {
        "tool": TOOL_NAME,
        "version": __version__,
        "author": AUTHOR,
        "orcid": ORCID,
        "affiliation": AFFILIATION,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_dir": str(args.input),
        "arguments": vars(args),
        "image_count": n_images,
        "particle_count": total,
        "um_per_px": um_per_px_by_image,
    }


def write_run_json(metadata: dict, dst: Path) -> None:
    """Write the provenance record as indented JSON."""
    dst.write_text(json.dumps(metadata, indent=2) + "\n")
