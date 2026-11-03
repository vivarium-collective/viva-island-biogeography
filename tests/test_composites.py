"""Tests for the composite generators and their registration/discovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from process_bigraph import Composite, gather_emitter_results

from viva_island_biogeography.core import build_core

# Runs one composite generator in a fresh interpreter and prints its emitter
# output as canonical JSON. Each subprocess gets its own hash salt, so any
# str-hash-based seeding shows up as a difference between the two runs.
_DUMP_SCRIPT = """
import json, sys
from process_bigraph import Composite, gather_emitter_results
import viva_island_biogeography.composites as gens
from viva_island_biogeography.core import build_core

name, kwargs = sys.argv[1], json.loads(sys.argv[2])
core = build_core()
doc = getattr(gens, name)(core, **kwargs)
sim = Composite({"state": doc}, core=core)
sim.run(10.0)
results = {str(k): v for k, v in gather_emitter_results(sim).items()}
print(json.dumps(results, sort_keys=True, default=str))
"""

_SEEDED_GENERATORS = [
    ("island_archipelago", {"n_islands": 4, "n_species": 12, "seed": 2}),
    ("mw_equilibrium", {"n_islands": 4, "n_pool": 12, "seed": 2}),
    ("general_dynamic_model", {"n_islands": 3, "n_pool": 12, "seed": 2}),
    ("neutral_metacommunity", {"n_islands": 4, "n_pool": 12, "seed": 2}),
    ("rescue_effect", {"n_islands": 4, "n_pool": 12, "seed": 2}),
]


@pytest.mark.parametrize(
    "generator,kwargs", _SEEDED_GENERATORS, ids=[g for g, _ in _SEEDED_GENERATORS]
)
def test_output_identical_across_interpreter_starts(generator, kwargs):
    """Same seed must give the same run in every interpreter start.

    Regression: per-process RNG seeds were built with ``hash((seed, name))``,
    which Python salts per process (PYTHONHASHSEED), so runs differed between
    interpreter starts even though the in-process reproducibility checks passed.
    """
    outputs = []
    for hashseed in ("1", "2"):
        proc = subprocess.run(
            [sys.executable, "-c", _DUMP_SCRIPT, generator, json.dumps(kwargs)],
            env={**os.environ, "PYTHONHASHSEED": hashseed},
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        outputs.append(proc.stdout.strip().splitlines()[-1])
    assert json.loads(outputs[0]), "generator produced no emitter output"
    assert outputs[0] == outputs[1]


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
