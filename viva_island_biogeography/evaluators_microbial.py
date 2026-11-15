"""Derived-scalar computers for the ``microbial-island-biogeography`` study.

Recomputes, exactly as ``workspace/studies/microbial-island-biogeography/sims/run.py``:

- ``z_micro``          = log-log taxa-area slope of the microbial regime
- ``z_flatter_margin`` = z_macro - z_micro  (macro regime = the study's own
  MACRO setup, run over the same area gradient and seeds)

The run persists every seed of both regimes into one store as ``gen=<k>``
simulations: macro seeds are gens 0..N_SEEDS-1, microbial seeds N_SEEDS..2N-1.
Per island: tail-mean (last 25% of ticks) richness, averaged over seeds, then
``polyfit(log A, log S)`` slope over islands with S > 0. Areas come from the
emitted ``island_meta.<island>.area``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_TAIL_FRAC = 0.25   # matches sims/run.py TAIL_FRAC
_N_SEEDS = 6        # matches sims/run.py N_SEEDS (macro gens first, then micro)


def _islands(reader: Any) -> list[str]:
    obs = reader.observables()
    return sorted({o.split(".", 1)[1] for o in obs
                   if o.startswith("richness.") and o.count(".") == 1})


def _regime_S(reader: Any, names: list[str], gens: range) -> np.ndarray:
    per_seed = {g: [] for g in gens}
    for n in names:
        df = reader.series(f"richness.{n}")          # read once per island
        for g in gens:
            v = df.filter(df["generation"] == g)["value"].to_numpy()
            t = v[max(0, int(len(v) * (1 - _TAIL_FRAC))):]
            per_seed[g].append(float(np.mean(t)))
    return np.mean(np.vstack([per_seed[g] for g in gens]), axis=0)


def _areas(reader: Any, names: list[str]) -> np.ndarray:
    return np.array([float(reader.series(f"island_meta.{n}.area")["value"].to_numpy()[-1])
                     for n in names])


def _fit_z(A: np.ndarray, S: np.ndarray) -> float:
    mask = S > 0
    return float(np.polyfit(np.log(A[mask]), np.log(S[mask]), 1)[0])


def _zs(reader: Any) -> tuple[float, float]:
    names = _islands(reader)
    A = _areas(reader, names)
    z_macro = _fit_z(A, _regime_S(reader, names, range(0, _N_SEEDS)))
    z_micro = _fit_z(A, _regime_S(reader, names, range(_N_SEEDS, 2 * _N_SEEDS)))
    return z_macro, z_micro


def _z_micro(reader: Any, test: Any, ws_root: Any) -> float:
    names = _islands(reader)
    return _fit_z(_areas(reader, names),
                  _regime_S(reader, names, range(_N_SEEDS, 2 * _N_SEEDS)))


def _z_flatter_margin(reader: Any, test: Any, ws_root: Any) -> float:
    z_macro, z_micro = _zs(reader)
    return z_macro - z_micro


def register(reg: dict) -> None:
    from discovery_engine.study.evaluator import declares_sources
    reg["z_micro"] = declares_sources("richness", "island_meta")(_z_micro)
    reg["z_flatter_margin"] = declares_sources("richness", "island_meta")(_z_flatter_margin)
