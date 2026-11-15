"""Derived-scalar computers for the ``island-food-web`` study.

Recomputes the study's two metrics from the persisted run store exactly as
``workspace/studies/island-food-web/sims/run.py`` does. The run persists its whole
ensemble into one store: 6 food-web seeds (simulation_id ``<run>-fw-s<k>``) and 6
matched random-matrix archipelago seeds (``<run>-rand-s<k>``).

Per ensemble member, equilibrium richness of an island = mean over the last 25
recorded steps of the number of species with abundance > 0 (``islands.*``); the
ensemble richness is the mean over seeds.

- ``richness_area_corr``        = corr(food-web ensemble richness per island, log area)
- ``richness_gain_over_random`` = mean food-web richness - mean random richness
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from .evaluators_common import island_areas

_TAIL = 25  # matches sims/run.py TAIL


def _ensemble(reader: Any, tag: str):
    """(island names in island_specs order, ensemble-mean richness per island)."""
    by_sim: dict[str, list] = {}
    for sim, _step, _t, state in reader._sqlite_rows():
        by_sim.setdefault(sim, []).append(state["islands"])
    pat = re.compile(rf"-{tag}-s\d+$")
    runs = [rows for sim, rows in by_sim.items() if pat.search(sim)]
    if not runs:
        raise ValueError(f"no '{tag}' ensemble members in store")
    from .evaluators_common import _specs
    names = [s["name"] for s in _specs(len(runs[0][0]))]
    rich = []
    for rows in runs:
        tail = rows[-_TAIL:]
        rich.append(np.array([
            np.mean([sum(1 for c in r[n].values() if c > 0) for r in tail])
            for n in names]))
    return names, np.mean(rich, axis=0)


def _richness_area_corr(reader: Any, test: Any, ws_root: Any) -> float:
    names, fw = _ensemble(reader, "fw")
    return float(np.corrcoef(fw, np.log(island_areas(names)))[0, 1])


def _richness_gain_over_random(reader: Any, test: Any, ws_root: Any) -> float:
    _, fw = _ensemble(reader, "fw")
    _, rand = _ensemble(reader, "rand")
    return float(fw.mean() - rand.mean())


def register(reg: dict) -> None:
    from discovery_engine.study.evaluator import declares_sources
    reg["fw_richness_area_corr"] = declares_sources("islands")(_richness_area_corr)
    reg["richness_gain_over_random"] = declares_sources("islands")(_richness_gain_over_random)
