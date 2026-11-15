"""Derived-scalar computers for ``species-area-and-isolation``.

Both metrics are 8-seed ensemble quantities; ``sims/run.py`` persists the real
ensemble per-island arrays (``S_obs_area``/``area``, ``S_obs_iso``/``distance``)
and these recompute the same reductions run.py prints.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _map_values(reader: Any, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for o in reader.observables():
        if o.startswith(prefix + "."):
            out[o[len(prefix) + 1:]] = float(reader.series(o)["value"].to_numpy()[-1])
    return out


def _pair(reader: Any, a: str, b: str):
    va, vb = _map_values(reader, a), _map_values(reader, b)
    names = sorted(set(va) & set(vb))
    return (np.array([va[n] for n in names]), np.array([vb[n] for n in names]))


def _sar_z(reader: Any, test: Any, ws_root: Any) -> float:
    S, A = _pair(reader, "S_obs_area", "area")
    mask = S > 0
    return float(np.polyfit(np.log(A[mask]), np.log(S[mask]), 1)[0])


def _spearman(x, y) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    return float(np.corrcoef(rx, ry)[0, 1])


def _isolation_corr(reader: Any, test: Any, ws_root: Any) -> float:
    S, D = _pair(reader, "S_obs_iso", "distance")
    return _spearman(D, S)


def register(reg: dict) -> None:
    from discovery_engine.study.evaluator import declares_sources
    reg["sar_z"] = declares_sources("S_obs_area", "area")(_sar_z)
    reg["isolation_corr_spearman"] = declares_sources("S_obs_iso", "distance")(_isolation_corr)
