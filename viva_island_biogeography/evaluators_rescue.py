"""Derived-scalar computers for the ``rescue-effect`` study.

These are ENSEMBLE quantities: ``sims/run.py`` runs all 8 seeds, reduces them to
per-island ensemble tail-mean richness (``S_obs``) and tail turnover events/gen
(``T_obs``), plus each island's ``rescue_strength`` and the analytic M-W S*
(``mw_sstar``), and persists them via ``persist_observables_store``. The computers
read those emitted observables and apply the same reduction run.py prints:
no-rescue / with-rescue islands = rescue_strength nearest 0.0 / 0.8;
richness_margin = S_with - S_no; turnover_drop = T_no - T_with;
sanity_err_r0 = |S_no - analytic M-W S*|.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_RESCUE_WITH = 0.8


def _map(reader: Any, prefix: str) -> dict[str, float]:
    out = {}
    for o in reader.observables():
        if o.startswith(prefix + "."):
            out[o[len(prefix) + 1:]] = float(reader.series(o)["value"].to_numpy()[-1])
    return out


def _pair(reader: Any, prefix: str) -> tuple[float, float]:
    """(no-rescue, with-rescue) value of an ensemble per-island observable."""
    vals, st = _map(reader, prefix), _map(reader, "rescue_strength")
    names = sorted(set(vals) & set(st))
    s = np.array([st[n] for n in names])
    i_no = int(np.argmin(np.abs(s - 0.0)))
    i_with = int(np.argmin(np.abs(s - _RESCUE_WITH)))
    return vals[names[i_no]], vals[names[i_with]]


def _richness_margin(reader: Any, test: Any, ws_root: Any) -> float:
    no, with_ = _pair(reader, "S_obs")
    return float(with_ - no)


def _turnover_drop(reader: Any, test: Any, ws_root: Any) -> float:
    no, with_ = _pair(reader, "T_obs")
    return float(no - with_)


def _sanity_err_r0(reader: Any, test: Any, ws_root: Any) -> float:
    no, _ = _pair(reader, "S_obs")
    return float(abs(no - _map(reader, "mw_sstar")["analytic"]))


def register(reg: dict) -> None:
    from discovery_engine.study.evaluator import declares_sources
    reg["richness_margin"] = declares_sources("S_obs", "rescue_strength")(_richness_margin)
    reg["turnover_drop"] = declares_sources("T_obs", "rescue_strength")(_turnover_drop)
    reg["sanity_err_r0"] = declares_sources("S_obs", "rescue_strength", "mw_sstar")(_sanity_err_r0)
