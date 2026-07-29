# Autolabel

An offline batch tool that detects and measures Nile red fluorescence microplastic
particles on filter-paper photographs.

## Requirements

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

## Installation

Install uv by following its
[installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Get the code with git:

```bash
git clone https://github.com/Senamjya/autolabel.git
cd autolabel
uv sync
```

Or grab the [latest release](https://github.com/Senamjya/autolabel/releases/latest),
unzip the source zip, and run `uv sync` inside the unpacked folder:

```bash
cd autolabel-*/
uv sync
```

## Usage

Drop your captures into `data/` and run:

```bash
uv run autolabel data/
```

Everything lands in `out/` (see [Output](#output)). Always eyeball the overlays and crops
before trusting the numbers. Every green box is a detected particle, the orange outline is
the filter disk.

### Options

**Detection**

| Flag                 | Default | Description                                                                                                                                                                  |
| -------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--threshold N`      | `100`   | Minimum red excess over the paper for a pixel to count as plastic. Raise it if bare paper gets boxed, lower it if faint particles are missed                                 |
| `--auto-threshold`   | off     | Choose the threshold automatically, overriding `--threshold`. A quick first pass, but it can misfire when very few particles are present                                     |
| `--luminance`        | off     | Detect by brightness instead of redness, for cyan or green-dominant batches where particles do not glow red. Pair with a high `--threshold` (~230) and `--grow-threshold 0`  |
| `--grow-threshold N` | `25`    | Grow each detection out to its connected region above this score, so a box covers the whole particle and not just its bright core (`0` = off, which also disables splitting) |
| `--hue-gate`         | off     | Also require an orange-red hue. Redundant when an orange long-pass filter is fitted, so off by default                                                                       |

**Filtering and cleanup**

| Flag               | Default | Description                                                                              |
| ------------------ | ------- | ---------------------------------------------------------------------------------------- |
| `--open-ksize N`   | `3`     | Size in pixels of the opening kernel that removes single-pixel noise (`0` = off)         |
| `--min-area N`     | `4`     | Drop blobs smaller than N pixels. Raise it if speckle noise gets boxed                   |
| `--max-area N`     | `0`     | Drop blobs larger than N pixels (`0` = off). Set a cap if a huge artefact gets boxed     |
| `--no-split`       | off     | Keep particles that grew together as a single box, instead of splitting them at the neck |
| `--pad N`          | `1`     | Grow each box by N pixels. Raise it if boxes clip the fluorescence halo                  |
| `--no-mask-filter` | off     | Keep detections that fall outside the filter disk, instead of restricting to it          |

**Output**

| Flag            | Default | Description                                                                                         |
| --------------- | ------- | --------------------------------------------------------------------------------------------------- |
| `--out DIR`     | `out`   | Output root. `crops/` and `overlays/` go inside it, with `sizes.csv` and `size_hist.png` at the top |
| `--no-overlays` | off     | Skip the full-frame annotated overlay images (written to `<out>/overlays/` by default)              |
| `--no-crops`    | off     | Skip the per-image zoomed contact sheets (written to `<out>/crops/` by default)                     |
| `--credit`      | off     | Stamp the author and version footer into the overlay and crop images                                |

**Info and early exit**

| Flag            | Default | Description                                                                              |
| --------------- | ------- | ---------------------------------------------------------------------------------------- |
| `--scale-only`  | off     | Report px/mm and um/px from the filter disk, then exit without writing anything          |
| `--replot DIR`  | -       | Redraw `size_hist.png` from an existing run's `sizes.csv` in DIR, then exit (no detection) |
| `--version`     | -       | Print the tool version, author and ORCID, then exit                                      |

## Output

`out/sizes.csv` has one row per particle:

| Column                  | Meaning                                             |
| ----------------------- | --------------------------------------------------- |
| `image`, `particle`     | source image and its 1-based particle index         |
| `x`, `y`, `w`, `h`      | particle bounding box in pixels (unpadded)          |
| `area_px`               | particle area in pixels                             |
| `um_per_px`             | this image's scale (blank if no disk was found)     |
| `equiv_diam_um`         | area-equivalent diameter, the orientation-free size |
| `length_um`, `width_um` | max / min Feret diameters (major and minor axes)    |
| `aspect_ratio`          | length / width, for telling fibres from fragments   |
| `tool_version`          | version of Autolabel that produced the row          |

`out/run.json` records the tool version, author, ORCID, UTC timestamp, the resolved
arguments, and each image's scale, so a result can be traced back to the exact run.

## Project structure

```
src/autolabel/       # source
├── config.py        # constants and the Params dataclass
├── provenance.py    # version/author metadata, run.json
├── measure.py       # geometry: Feret, equivalent and disk diameter
├── detect.py        # pixel scoring, disk mask, detect()
├── render.py        # overlays, crops, histogram
├── report.py        # sizes.csv, report.txt, summaries
├── cli.py           # argparse, process(), main()
├── __init__.py      # package marker
└── __main__.py      # python -m autolabel
tests/               # tests
data/                # input images
out/                 # generated results
```

Treat all of `out/` as disposable. Re-running the tool regenerates it.

## Development

```bash
uv run pytest                              # tests
uv run coverage run -m pytest              # tests with coverage
uv run coverage report                     # coverage summary
uv run ruff check .                        # lint
uv run ty check                            # type-check
```

## Citation

Written by Senam Julian Yao Asmussen (NTNU), ORCID
[0009-0001-4885-4801](https://orcid.org/0009-0001-4885-4801).

If you use this tool,
please cite it. `CITATION.cff` at the repo root holds the full citation metadata.

Run `uv run autolabel --version` to print the version, author and ORCID.

## License

Copyright 2026 Norwegian University of Science and Technology (NTNU).

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text
and [NOTICE](NOTICE) for attribution.
