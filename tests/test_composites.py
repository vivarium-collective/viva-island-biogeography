"""Tests for the composite generators and their registration/discovery."""

from __future__ import annotations

import numpy as np

from process_bigraph import Composite, gather_emitter_results

from viva_island_biogeography.core import build_core


def test_generators_are_registered():
    from process_bigraph.composite_generator import _REGISTRY
    want = {"island_archipelago", "island_food_web"}
    found = {eid.split(".")[-1] for eid in _REGISTRY}
    missing = want - found
    assert not missing, f"missing generators {missing}; have {sorted(found)[:8]}"


def test_generators_discoverable():
    from process_bigraph.composite_generator import discover_generators
    gens = discover_generators()
    matches = [g for g in gens if "viva_island_biogeography" in g]
    assert matches, f"no island-biogeography generators discovered; got {list(gens)[:5]}"


def test_archipelago_builds_and_runs():
    core = build_core()
    from viva_island_biogeography.composites.archipelago import island_archipelago
    doc = island_archipelago(core, n_islands=4, n_species=12, seed=2)
    sim = Composite({"state": doc}, core=core)
    sim.run(10.0)
    rows = gather_emitter_results(sim)[("emitter",)]
    assert len(rows) >= 10
    # every island fills toward (does not exceed) its carrying capacity
    ipa = 100.0
    from viva_island_biogeography.composites._common import island_specs
    import random
    specs = {s["name"]: s for s in island_specs(4, 12, random.Random(2))}
    last = rows[-1]["islands"]
    for name, ab in last.items():
        cap = specs[name]["area"] * ipa
        assert sum(ab.values()) <= cap * 1.01


def test_food_web_builds_and_runs():
    core = build_core()
    from viva_island_biogeography.composites.food_web import island_food_web
    doc = island_food_web(core, n_islands=3, n_species=12, seed=1)
    sim = Composite({"state": doc}, core=core)
    sim.run(10.0)
    rows = gather_emitter_results(sim)[("emitter",)]
    assert len(rows) >= 10
    # food web matrix is antisymmetric on predator-prey pairs
    from viva_island_biogeography.composites.food_web import (
        generate_food_web, interaction_matrix_from_food_web,
    )
    import random
    species = [f"Species_{i}" for i in range(12)]
    web = generate_food_web(species, random.Random(1))
    M = interaction_matrix_from_food_web(species, web, 0.1)
    idx = {s: i for i, s in enumerate(species)}
    for pred, prey_list in web.items():
        for prey in prey_list:
            p, q = idx[pred], idx[prey]
            assert np.isclose(M[p][q], -M[q][p])


def test_archipelago_seed_is_reproducible():
    core1 = build_core()
    core2 = build_core()
    from viva_island_biogeography.composites.archipelago import island_archipelago
    rows = []
    for core in (core1, core2):
        doc = island_archipelago(core, n_islands=3, n_species=10, seed=7)
        sim = Composite({"state": doc}, core=core)
        sim.run(8.0)
        rows.append(gather_emitter_results(sim)[("emitter",)][-1]["islands"])
    assert rows[0] == rows[1]
