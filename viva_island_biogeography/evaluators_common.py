"""Shared helpers for the workspace's derived-scalar computers.

A derived-scalar computer has signature ``fn(reader, test, ws_root) -> float``
(see ``discovery_engine.study.evaluator.load_workspace_derived_scalars``). The
``reader`` is a :class:`viva_emitters.RunReader` opened on a persisted run store
whose leaf observables are the flattened emitted paths, e.g. ``richness.island1``
and ``islands.island1.Species_3``. These helpers turn those leaf series into the
per-island equilibrium quantities the island-biogeography metrics are built from,
so each study's computer recomputes its published number from the REAL run store
(not a recorded/authored value).

Nothing here invents a mechanism or a parameter: the island geometry is the same
deterministic ``island_specs`` the composites and canonical runs use, and the
tail-window reductions mirror each study's ``sims/run.py``.
"""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Any

import numpy as np


def island_names(reader: Any) -> list[str]:
    """Island names present in the store, in deterministic ``island_specs`` order.

    Derived from the ``richness.<island>`` leaf observables the archipelago-family
    composites emit (one per island), then ordered to match ``island_specs`` so a
    computer's per-island vectors align with the reconstructed areas.
    """
    obs = reader.observables()
    names = sorted({o.split(".", 1)[1] for o in obs
                    if o.startswith("richness.") and o.count(".") == 1})
    spec_order = [s["name"] for s in _specs(len(names))]
    ordered = [n for n in spec_order if n in names]
    # Any name not in the preset ordering (defensive) appended in sorted order.
    ordered += [n for n in names if n not in ordered]
    return ordered


@lru_cache(maxsize=None)
def _specs(n_islands: int) -> tuple:
    from viva_island_biogeography.composites._common import island_specs
    # area in island_specs is independent of the RNG (only x/y position uses it),
    # so Random(0) is safe and deterministic for area reconstruction.
    return tuple(island_specs(n_islands, 1, random.Random(0)))


def island_areas(names: list[str]) -> np.ndarray:
    """Reconstruct each island's area from the deterministic ``island_specs`` geometry."""
    by_name = {s["name"]: float(s["area"]) for s in _specs(len(names))}
    return np.array([by_name[n] for n in names], dtype=float)


def _series_values(reader: Any, path: str) -> np.ndarray:
    df = reader.series(path)
    return df["value"].to_numpy()


def tail_mean(reader: Any, path: str, tail: int) -> float:
    """Mean of a leaf observable over the last ``tail`` recorded steps (the
    equilibrium window each study's run uses)."""
    v = _series_values(reader, path)
    if v.size == 0:
        raise ValueError(f"empty series for {path!r}")
    return float(np.mean(v[-tail:]))


def per_island_total_abundance(reader: Any, names: list[str], tail: int) -> np.ndarray:
    """Tail-mean total abundance per island = sum over that island's species leaves."""
    obs = reader.observables()
    out = []
    for n in names:
        leaves = [o for o in obs if o.startswith(f"islands.{n}.")]
        out.append(float(np.sum([tail_mean(reader, o, tail) for o in leaves])))
    return np.array(out, dtype=float)


def per_island_richness(reader: Any, names: list[str], tail: int) -> np.ndarray:
    """Tail-mean species richness per island from the ``richness.<island>`` leaves."""
    return np.array([tail_mean(reader, f"richness.{n}", tail) for n in names], dtype=float)
