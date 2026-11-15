"""Derived-scalar computers for ``neutral-vs-equilibrium``.

``sims/run.py`` persists the real 8-seed ensemble per-island richness and area
for both the neutral and MacArthur-Wilson composites; these recompute
``z_neutral`` and ``both_sar`` exactly as run.py does.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_SAR_Z_MIN = 0.05


def _map_values(reader: Any, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for o in reader.observables():
        if o.startswith(prefix + "."):
            out[o[len(prefix) + 1:]] = float(reader.series(o)["value"].to_numpy()[-1])
    return out


def _z(reader: Any, s_prefix: str, a_prefix: str) -> float:
    vs, va = _map_values(reader, s_prefix), _map_values(reader, a_prefix)
    names = sorted(set(vs) & set(va))
    S = np.array([vs[n] for n in names])
    A = np.array([va[n] for n in names])
    mask = S > 0
    if mask.sum() < 2 or np.ptp(np.log(A[mask])) <= 0:
        return 0.0
    return float(np.polyfit(np.log(A[mask]), np.log(S[mask]), 1)[0])


def _z_neutral(reader: Any, test: Any, ws_root: Any) -> float:
    return _z(reader, "S_obs_neutral", "area_neutral")


def _both_sar(reader: Any, test: Any, ws_root: Any) -> float:
    zn = _z(reader, "S_obs_neutral", "area_neutral")
    ze = _z(reader, "S_obs_eq", "area_eq")
    return float((zn > _SAR_Z_MIN) and (ze > _SAR_Z_MIN))


def register(reg: dict) -> None:
    from discovery_engine.study.evaluator import declares_sources
    reg["z_neutral"] = declares_sources("S_obs_neutral", "area_neutral")(_z_neutral)
    reg["both_sar"] = declares_sources(
        "S_obs_neutral", "area_neutral", "S_obs_eq", "area_eq")(_both_sar)
