"""Unit tests for the island-biogeography processes (offline, fast)."""

from __future__ import annotations

import numpy as np

from process_bigraph import Composite, gather_emitter_results

from viva_island_biogeography.core import build_core
from viva_island_biogeography.processes import (
    IslandProcess,
    MigrationProcess,
    lotka_volterra_step,
    migration_rate,
)


def test_migration_rate_decays_with_distance():
    assert migration_rate(0) == 1.0
    assert migration_rate(1) == 0.5
    assert migration_rate(9) == 0.1
    assert migration_rate(10) < migration_rate(1)


def test_lotka_volterra_matches_reference_formula():
    # Two species, no interactions: new = count + growth*count, int-truncated.
    species = ["A", "B"]
    M = np.zeros((2, 2))
    out = lotka_volterra_step({"A": 100.0, "B": 0.0}, species, M,
                              growth_rate=0.03, capacity=1e9)
    # 100 + 0.03*100 = 103
    assert out["A"] == 103.0
    # absent species stays 0
    assert out["B"] == 0.0


def test_lotka_volterra_capacity_rescale():
    species = ["A", "B"]
    M = np.zeros((2, 2))
    # projected total 200 but capacity 100 -> scale by 0.5
    out = lotka_volterra_step({"A": 100.0, "B": 100.0}, species, M,
                              growth_rate=0.0, capacity=100.0)
    assert out["A"] + out["B"] <= 100.0


def test_island_process_inputs_outputs_are_dicts():
    core = build_core()
    proc = IslandProcess(config={
        "name": "isle", "species_list": ["A", "B"],
        "interaction_matrix": [[0.0, 0.0], [0.0, 0.0]],
        "mean_growth_rate": 0.1, "area": 10.0,
    }, core=core)
    assert isinstance(proc.inputs(), dict)
    assert isinstance(proc.outputs(), dict)
    assert set(proc.inputs()) == {"abundance", "immigration", "growth_modifier"}
    assert set(proc.outputs()) == {"abundance", "richness"}


def test_island_process_emits_growth_delta():
    core = build_core()
    proc = IslandProcess(config={
        "name": "isle", "species_list": ["A", "B"],
        "interaction_matrix": [[0.0, 0.0], [0.0, 0.0]],
        "mean_growth_rate": 0.1, "area": 1e6, "individuals_per_area": 1.0,
        "introduction_probability": 0.0,
    }, core=core)
    out = proc.update(
        {"abundance": {"A": 100.0, "B": 0.0}, "immigration": {}, "growth_modifier": 1.0},
        1.0,
    )
    # 100 -> 110, delta = +10
    assert abs(out["abundance"]["A"] - 10.0) < 1e-9


def test_island_process_growth_modifier_scales_growth():
    core = build_core()
    cfg = dict(name="isle", species_list=["A"], interaction_matrix=[[0.0]],
               mean_growth_rate=0.1, area=1e6, individuals_per_area=1.0,
               introduction_probability=0.0)
    proc = IslandProcess(config=cfg, core=core)
    out = proc.update({"abundance": {"A": 100.0}, "immigration": {},
                       "growth_modifier": 2.0}, 1.0)
    # growth doubled: 100 -> 100 + 0.2*100 = 120, delta +20
    assert abs(out["abundance"]["A"] - 20.0) < 1e-9


def test_island_process_immigration_passes_through():
    core = build_core()
    proc = IslandProcess(config={
        "name": "isle", "species_list": ["A"], "interaction_matrix": [[0.0]],
        "mean_growth_rate": 0.0, "area": 1e6, "individuals_per_area": 1.0,
        "introduction_probability": 0.0,
    }, core=core)
    out = proc.update({"abundance": {"A": 10.0}, "immigration": {"A": 5.0},
                       "growth_modifier": 1.0}, 1.0)
    # no growth, LV delta 0, + immigration 5
    assert abs(out["abundance"]["A"] - 5.0) < 1e-9


def test_migration_conserves_individuals():
    """Every emigration is matched by an immigration: net exchange == 0."""
    core = build_core()
    names = ["a", "b", "c"]
    proc = MigrationProcess(config={
        "island_names": names,
        "positions": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        "seed": 5,
    }, core=core)
    ab = {n: {"A": 50.0, "B": 30.0} for n in names}
    for _ in range(100):
        out = proc.update({"abundances": ab}, 1.0)
        net = sum(sum(v.values()) for v in out["exchange"].values())
        assert abs(net) < 1e-12


def test_migration_never_over_emigrates():
    """Working-copy decrement keeps emigration within what an island holds."""
    core = build_core()
    names = ["a", "b", "c", "d"]
    proc = MigrationProcess(config={
        "island_names": names,
        "positions": [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]],
        "seed": 1,
    }, core=core)
    # island 'a' has only 1 individual of species A; many close neighbors would
    # each want to pull it, but only one can leave.
    ab = {"a": {"A": 1.0}, "b": {"A": 0.0}, "c": {"A": 0.0}, "d": {"A": 0.0}}
    for _ in range(50):
        out = proc.update({"abundances": ab}, 1.0)
        emigrated = -out["exchange"].get("a", {}).get("A", 0.0)
        assert emigrated <= 1.0


def test_short_composite_run_executes():
    core = build_core()
    from viva_island_biogeography.composites.archipelago import island_archipelago
    doc = island_archipelago(core, n_islands=3, n_species=10, seed=0)
    sim = Composite({"state": doc}, core=core)
    sim.run(5.0)
    rows = gather_emitter_results(sim)[("emitter",)]
    assert len(rows) >= 5
    assert set(rows[-1]["islands"].keys()) == {"mainland", "island1", "island2"}
