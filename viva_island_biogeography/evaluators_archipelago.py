"""Derived-scalar computers for the ``island-archipelago`` study.

Recomputes the study's two published metrics from a persisted run store, exactly
as ``workspace/studies/island-archipelago/sims/run.py`` computes them:

- ``capacity_area_corr`` = corr(tail-mean total abundance per island, area)
- ``richness_area_corr``  = corr(tail-mean richness per island, log area)

Tail window = last 25 recorded steps (``TAIL`` in that run). Both read only
emitted observables (``islands.*``, ``richness.*``) plus the deterministic island
geometry, so the grade is earned by the run, not asserted.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .evaluators_common import (
    island_areas, island_names, per_island_richness, per_island_total_abundance)

_TAIL = 25  # matches sims/run.py TAIL (equilibrium window)


def _capacity_area_corr(reader: Any, test: Any, ws_root: Any) -> float:
    names = island_names(reader)
    eq_total = per_island_total_abundance(reader, names, _TAIL)
    area = island_areas(names)
    return float(np.corrcoef(eq_total, area)[0, 1])


def _richness_area_corr(reader: Any, test: Any, ws_root: Any) -> float:
    names = island_names(reader)
    eq_rich = per_island_richness(reader, names, _TAIL)
    area = island_areas(names)
    return float(np.corrcoef(eq_rich, np.log(area))[0, 1])


def register(reg: dict) -> None:
    from discovery_engine.study.evaluator import declares_sources
    reg["capacity_area_corr"] = declares_sources("islands", "richness")(_capacity_area_corr)
    reg["richness_area_corr"] = declares_sources("richness")(_richness_area_corr)
