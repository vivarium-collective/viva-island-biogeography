# Contributing to viva-island-biogeography

## Development setup

uv is required. Install with `brew install uv` or `pip install uv`.

    uv venv .venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"
    pytest

Run the workspace lint and a showcase study from the repo root:

    python scripts/lint-workspace.py
    python workspace/studies/island-archipelago/sims/run.py

## Releasing to PyPI

Tag a commit with `git tag v<VERSION>` and push the tag. The
`.github/workflows/release.yml` workflow publishes to PyPI automatically
using trusted publishing (no tokens needed after initial setup).

PyPI trusted publishing must be configured once per repo. See
https://docs.pypi.org/trusted-publishers/ and
[docs/conventions/distribution.md](https://github.com/vivarium-collective/viva-superpowers/blob/main/docs/conventions/distribution.md).
