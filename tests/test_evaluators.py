"""Guards for the run-earned evidence layer: derived-scalar evaluators + the
run-persistence helper that lets the discovery-engine grade behavior tests by
CODE against real emitted observables.

Fast (no heavy simulation): registration/collision checks, a persist→read
roundtrip, and one computer end-to-end on a synthetic store.
"""
from __future__ import annotations

import numpy as np
import pytest

from viva_island_biogeography import evaluators
from viva_island_biogeography.core import build_core

# Every behavior-test field across the nine studies. If a study's measure field
# is renamed, update this list — a drift here means a test will SKIP (agent
# bucket) in the real workspace.
EXPECTED_FIELDS = {
    "capacity_area_corr", "richness_area_corr",              # island-archipelago
    "immigration_distance_corr", "total_net_migration",      # distance-immigration
    "fw_richness_area_corr", "richness_gain_over_random",    # island-food-web
    "richness-is-humped",                                    # general-dynamic-model
    "z_micro", "z_flatter_margin",                           # microbial
    "eq_track_corr", "turnover_ratio",                       # mw-equilibrium
    "z_neutral", "both_sar",                                 # neutral-vs-equilibrium
    "richness_margin", "sanity_err_r0", "turnover_drop",     # rescue-effect
    "sar_z", "isolation_corr_spearman",                      # species-area
}


def _registry() -> dict:
    reg: dict = {}
    evaluators.register_derived_scalars(reg)
    return reg


def test_all_study_fields_register():
    reg = _registry()
    missing = EXPECTED_FIELDS - set(reg)
    assert not missing, f"unregistered derived-scalar fields: {sorted(missing)}"


def test_no_field_name_collision_across_studies():
    # The workspace registry is global by field name; two studies registering the
    # same field would silently overwrite one another (the bug that made
    # island-archipelago's richness_area_corr SKIP once food-web also used it).
    counts: dict[str, int] = {}
    import importlib
    for name in evaluators._STUDY_EVALUATOR_MODULES:
        try:
            mod = importlib.import_module(f"viva_island_biogeography.{name}")
        except ModuleNotFoundError:
            continue
        local: dict = {}
        mod.register(local)
        for field in local:
            counts[field] = counts.get(field, 0) + 1
    clashes = {f: n for f, n in counts.items() if n > 1}
    assert not clashes, f"derived-scalar field names registered by >1 study: {clashes}"


def test_every_computer_is_callable():
    reg = _registry()
    for field, fn in reg.items():
        assert callable(fn), f"{field} computer is not callable"


def test_persist_observables_store_roundtrip(tmp_path):
    from viva_island_biogeography.run_persistence import persist_observables_store
    from viva_emitters import RunReader
    core = build_core()
    obs = {"S_obs": {"a": 4.2, "b": 8.6}, "S_star": {"a": 4.0, "b": 8.1}}
    db = persist_observables_store(
        tmp_path, "run1", obs, core=core,
        spec_id="test", params={"seed": 0}, n_steps=1)
    r = RunReader.open(db)
    got = set(r.observables())
    assert {"S_obs.a", "S_obs.b", "S_star.a", "S_star.b"} <= got
    assert r.series("S_obs.b")["value"].to_numpy()[-1] == pytest.approx(8.6)
    # the run is registered so the engine can find the store
    assert (tmp_path / "runs.db").exists()


def test_mw_eq_track_corr_computer_on_synthetic_store(tmp_path):
    # Persist S_obs that perfectly tracks S_star -> the computer must recompute
    # correlation ~1.0 from the real emitted observables (not an authored value).
    from viva_island_biogeography.run_persistence import persist_observables_store
    from viva_island_biogeography import evaluators_mw_equilibrium as mw
    from viva_emitters import RunReader
    core = build_core()
    islands = [f"i{k}" for k in range(6)]
    star = {n: 10.0 + 3.0 * k for k, n in enumerate(islands)}
    obs = {"S_obs": dict(star), "S_star": dict(star),
           "turnover_gross": {"i3": 40.0}, "turnover_net": {"i3": 2.0}}
    db = persist_observables_store(tmp_path, "r", obs, core=core,
                                   spec_id="mw", params={}, n_steps=1)
    reg: dict = {}
    mw.register(reg)
    reader = RunReader.open(db)
    assert reg["eq_track_corr"](reader, {}, str(tmp_path)) == pytest.approx(1.0)
    assert reg["turnover_ratio"](reader, {}, str(tmp_path)) == pytest.approx(20.0)
