"""Derived-scalar computers for the ``distance-immigration`` study.

Recomputes the study's two metrics from the persisted run store exactly as
``workspace/studies/distance-immigration/sims/run.py`` does. The run holds one
``MigrationProcess`` per seed over a held-full-occupancy archipelago and emits the
cumulative per-island per-species migration exchange (running totals; a tick's
exchange is the difference of consecutive totals) as ``exchange_s<seed>.<island>.<species>``.

- ``immigration_distance_corr`` = corr(immigration received per island, mean
  Euclidean distance to the other islands), received = sum of positive exchange
  deltas averaged over the seed ensemble.
- ``total_net_migration`` = ABSOLUTE value of the net exchange summed over all
  seeds, islands, species and ticks. Conservation (each emigration -1 is matched by
  an immigration +1) makes this exactly 0; it is returned as an absolute value so
  the supported op ``<`` against a tiny epsilon expresses "net flux ~ 0".
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

_N_ISLANDS = 8  # matches sims/run.py N_ISLANDS


def _specs():
    from viva_island_biogeography.composites._common import island_specs
    return island_specs(_N_ISLANDS, 30, random.Random(0))


def _sums(reader: Any):
    """(received per island averaged over seeds, total net flux).

    Reads the emitted history once (the run store holds ~1.4k per-species leaves;
    per-leaf ``series`` calls would re-parse every row each time).
    """
    names = [s["name"] for s in _specs()]
    rows = [state for _sim, _step, _t, state in reader._sqlite_rows()]
    seeds = sorted({int(k[len("exchange_s"):]) for k in rows[0]
                    if k.startswith("exchange_s")})
    received = np.zeros(len(names))
    total = 0.0
    for sd in seeds:
        prev = None
        for st in rows:
            cur = st[f"exchange_s{sd}"]
            for i, n in enumerate(names):
                for sp, c in cur[n].items():
                    # cumulative running total -> this tick's exchange
                    v = c - (prev[n][sp] if prev else 0.0)
                    total += v
                    if v > 0:
                        received[i] += v
            prev = cur
    received /= max(len(seeds), 1)
    return names, received, total


def _immigration_distance_corr(reader: Any, test: Any, ws_root: Any) -> float:
    specs = _specs()
    names, received, _ = _sums(reader)
    meta = {s["name"]: s for s in specs}
    mean_dist = np.array([
        np.mean([math.hypot(meta[a]["x"] - meta[b]["x"], meta[a]["y"] - meta[b]["y"])
                 for b in names if b != a]) for a in names])
    return float(np.corrcoef(received, mean_dist)[0, 1])


def _total_net_migration(reader: Any, test: Any, ws_root: Any) -> float:
    return abs(_sums(reader)[2])


def register(reg: dict) -> None:
    from discovery_engine.study.evaluator import declares_sources
    srcs = tuple(f"exchange_s{sd}" for sd in range(6))
    reg["immigration_distance_corr"] = declares_sources(*srcs)(_immigration_distance_corr)
    reg["total_net_migration"] = declares_sources(*srcs)(_total_net_migration)
