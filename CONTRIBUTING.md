# Contributing

Thanks for taking the time to contribute.

## Development setup

Requires Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Checks

Run these before opening a pull request. CI runs the same set on Python 3.10 and 3.13.

```bash
uv run pytest                 # tests
uv run coverage run -m pytest # tests with coverage
uv run coverage report        # coverage summary
uv run ruff check .           # lint
uv run ty check               # type-check
```

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/), and
they are load-bearing: releases are automated with
[release-please](https://github.com/googleapis/release-please), which reads the commit
history on `main` to decide the next version and to build the changelog. A stray
non-conventional message will not break the build, but it will be left out of the
changelog and may lead to a wrong version bump.

Format:

```
<type>: <short summary>
```

Types that affect the release:

| Type                                   | Version bump | Example                              |
| -------------------------------------- | ------------ | ------------------------------------ |
| `feat`                                 | minor        | `feat: add --replot flag`            |
| `fix`                                  | patch        | `fix: guard against empty size list` |
| `feat!` or a `BREAKING CHANGE:` footer | major        | `feat!: drop Python 3.9 support`     |

Other types do not trigger a release and are fine to use freely: `docs`, `test`,
`refactor`, `chore`, `build`, `ci`, `style`, `perf`.

Keep the summary in the imperative mood and lower case, with no trailing period.

This repository squash-merges pull requests, so the individual commits on a branch
never reach `main`. The squashed commit takes its message from the pull request title,
which means **the PR title is the commit release-please reads**. Make every PR title a
valid Conventional Commit. The commit messages within a branch still help review, but
they do not affect versioning.

## Releases

You do not bump the version by hand. On every push to `main`, release-please keeps a
release pull request open that bumps the version (in `provenance.py` and
`CITATION.cff`) and updates `CHANGELOG.md`. Merging that pull request tags the release
and creates a GitHub Release, which mints the Zenodo DOI. Nothing is released until
that pull request is deliberately merged.
