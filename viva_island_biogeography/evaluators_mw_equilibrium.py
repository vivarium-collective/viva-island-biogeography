"""Derived-scalar computers for the ``mw-equilibrium-reproduction`` study.

Both metrics are ENSEMBLE quantities a single-seed trajectory cannot
reconstruct, so the canonical run (``sims/run.py``) runs all ``N_SEEDS`` seeds,
reduces them to the per-island arrays below, and persists them via
``run_persistence.persist_observables_store``. These computers read those real
emitted observables back and apply the SAME reduction run.py prints:

- ``eq_track_corr``  = corr(ensemble-mean equilibrium richness ``S_obs`` per
  island, analytic ``S_star`` per island) across the area gradient.
- ``turnover_ratio`` = gross colonization+extinction events / net richness change
  on the representative mid-gradient island (``turnover_gross`` / ``turnover_net``).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _map_values(reader: Any, prefix: str) -> dict[str, float]:
    """Read every ``<prefix>.<key>`` leaf observable as ``{key: value}`` (one row each)."""
    out: dict[str, float] = {}
    for o in reader.observables():
        if o.startswith(prefix + "."):
            key = o[len(prefix) + 1:]
            out[key] = float(reader.series(o)["value"].to_numpy()[-1])
    return out


def _eq_track_corr(reader: Any, test: Any, ws_root: Any) -> float:
    s_obs = _map_values(reader, "S_obs")
    s_star = _map_values(reader, "S_star")
    names = sorted(set(s_obs) & set(s_star))
    obs = np.array([s_obs[n] for n in names])
    star = np.array([s_star[n] for n in names])
    return float(np.corrcoef(obs, star)[0, 1])


def _turnover_ratio(reader: Any, test: Any, ws_root: Any) -> float:
    gross = _map_values(reader, "turnover_gross")
    net = _map_values(reader, "turnover_net")
    # a single representative island is persisted
    rep = next(iter(gross))
    return float(gross[rep] / max(1.0, net[rep]))


def register(reg: dict) -> None:
    from discovery_engine.study.evaluator import declares_sources
    reg["eq_track_corr"] = declares_sources("S_obs", "S_star")(_eq_track_corr)
    reg["turnover_ratio"] = declares_sources("turnover_gross", "turnover_net")(_turnover_ratio)
