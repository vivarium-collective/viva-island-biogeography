"""Derived-scalar computer for the ``general-dynamic-model`` study.

Recomputes ``richness-is-humped`` exactly as
``workspace/studies/general-dynamic-model/sims/run.py`` does, from the persisted
store (all ensemble seeds are persisted as separate ``gen=<k>`` simulations in
one store): ensemble-mean focal-island richness, centred 15-tick moving average
to locate the peak, then the middle-third / 0.75*peak early+late margin gate.
Returns the gate indicator 1.0 (hump gate passes) or 0.0 (fails); derived-scalar
computers must be numeric, so the test's pass_if is ``== 1.0``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_MARGIN_FRAC = 0.75   # matches sims/run.py
_EDGE_FRAC = 0.10
_SMOOTH_W = 15


def _smooth(y, w=_SMOOTH_W):
    y = np.asarray(y, dtype=float)
    if len(y) < w or w < 2:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def _richness_is_humped(reader: Any, test: Any, ws_root: Any) -> float:
    obs = reader.observables()
    names = sorted({o.split(".", 1)[1] for o in obs
                    if o.startswith("richness.") and o.count(".") == 1})
    focal = f"island_{len(names) // 2:02d}"          # FOCAL_IDX = N_ISLANDS // 2
    df = reader.series(f"richness.{focal}")
    runs = [g["value"].to_numpy() for _, g in
            sorted(df.group_by("generation"), key=lambda kv: kv[0])]
    L = min(len(r) for r in runs)
    mean_rich = np.vstack([r[:L] for r in runs]).mean(axis=0)
    peak_idx = int(np.argmax(_smooth(mean_rich)))
    peak = float(mean_rich[peak_idx])
    edge = max(1, int(L * _EDGE_FRAC))
    early = float(np.mean(mean_rich[:edge]))
    late = float(np.mean(mean_rich[-edge:]))
    hump = ((L / 3.0) < peak_idx < (2.0 * L / 3.0)
            and early < _MARGIN_FRAC * peak and late < _MARGIN_FRAC * peak)
    return 1.0 if hump else 0.0


def register(reg: dict) -> None:
    from discovery_engine.study.evaluator import declares_sources
    reg["richness-is-humped"] = declares_sources("richness")(_richness_is_humped)
