"""Shared archipelago document builder for the composite generators.

Both the Lotka-Volterra archipelago and the food-web variant assemble the same
topology — N :class:`IslandProcess` nodes + one :class:`MigrationProcess`
coupler, wired through a shared ``islands`` abundance tree, plus a RAM emitter —
differing only in how the species-interaction matrix is generated. This module
holds the reusable pieces so each generator supplies just its interaction
matrix.
"""

from __future__ import annotations

import random

import numpy as np

from ..processes import generate_random_interaction_matrix, seed_initial_abundance


# Preset island geometry from the reference ``run_island_map`` (mainland + 3).
_PRESET = [
    # name,      x,   y,   area, initial_species, initial_individuals
    ("mainland", 0.0, 50.0, 1600.0, 20, 100),
    ("island1", 45.0, 25.0, 100.0, 20, 10),
    ("island2", 65.0, 75.0, 25.0, 20, 10),
    ("island3", 75.0, 65.0, 15.0, 20, 10),
]


def island_specs(n_islands: int, n_species: int, rng: random.Random):
    """Return ``n_islands`` island specs (dicts).

    The first up-to-4 follow the reference preset (mainland + island1-3); any
    beyond that are placed deterministically farther out with exponentially
    smaller area, mimicking the reference's ``generate_islands`` spirit while
    staying reproducible under ``rng``.
    """
    specs = []
    for i in range(min(n_islands, len(_PRESET))):
        name, x, y, area, init_sp, init_ind = _PRESET[i]
        specs.append({
            "name": name, "x": x, "y": y, "area": area,
            "initial_species": min(init_sp, n_species),
            "initial_individuals": init_ind,
        })
    for i in range(len(_PRESET), n_islands):
        angle = rng.uniform(0, 2 * np.pi)
        radius = 90.0 + 25.0 * (i - len(_PRESET) + 1)
        specs.append({
            "name": f"island{i}",
            "x": radius * float(np.cos(angle)),
            "y": 50.0 + radius * float(np.sin(angle)),
            "area": max(10.0, 80.0 * (0.8 ** (i - len(_PRESET) + 1))),
            "initial_species": max(1, min(10, n_species)),
            "initial_individuals": 10,
        })
    return specs


def build_archipelago_document(
    *,
    n_islands: int,
    n_species: int,
    growth_rate: float,
    introduction_probability: float,
    interaction_matrix: np.ndarray,
    interval: float = 1.0,
    seed: int = 0,
    individuals_per_area: float = 100.0,
    geometry_seed: int = 0,
):
    """Assemble the full process-bigraph document for an archipelago.

    ``interaction_matrix`` is an ``n_species x n_species`` ndarray supplied by
    the caller (random for the baseline, food-web-derived for the variant).

    Island **geometry** (positions + areas of any islands beyond the fixed
    preset) is drawn from ``geometry_seed``, deliberately decoupled from the
    dynamics ``seed`` so that a seed ensemble over the stochastic dynamics keeps
    the archipelago fixed — essential for across-island area/distance analyses.
    """
    species_list = [f"Species_{i}" for i in range(n_species)]
    specs = island_specs(n_islands, n_species, random.Random(geometry_seed))
    matrix_list = interaction_matrix.tolist()

    islands_store: dict[str, dict] = {}
    environment_store: dict[str, float] = {}
    immigration_store: dict[str, dict] = {}
    richness_store: dict[str, float] = {}
    nodes: dict[str, dict] = {}

    island_names = [s["name"] for s in specs]
    positions = [[s["x"], s["y"]] for s in specs]

    for idx, spec in enumerate(specs):
        name = spec["name"]
        seed_rng = random.Random((seed, name).__hash__())
        capacity = spec["area"] * individuals_per_area
        islands_store[name] = seed_initial_abundance(
            species_list,
            spec["initial_species"],
            spec["initial_individuals"],
            interaction_matrix,
            growth_rate,
            capacity,
            seed_rng,
        )
        environment_store[name] = 1.0
        immigration_store[name] = {}
        richness_store[name] = 0.0

        nodes[f"island_{name}"] = {
            "_type": "process",
            "address": "local:IslandProcess",
            "config": {
                "name": name,
                "species_list": species_list,
                "interaction_matrix": matrix_list,
                "mean_growth_rate": growth_rate,
                "area": spec["area"],
                "individuals_per_area": individuals_per_area,
                "introduction_probability": introduction_probability,
                "x": spec["x"],
                "y": spec["y"],
                "seed": seed + idx,
            },
            "interval": interval,
            "inputs": {
                "abundance": ["islands", name],
                "immigration": ["immigration", name],
                "growth_modifier": ["environment", name],
            },
            "outputs": {
                "abundance": ["islands", name],
                "richness": ["richness", name],
            },
        }

    nodes["migration"] = {
        "_type": "process",
        "address": "local:MigrationProcess",
        "config": {
            "island_names": island_names,
            "positions": positions,
            "seed": seed,
        },
        "interval": interval,
        "inputs": {"abundances": ["islands"]},
        "outputs": {"exchange": ["islands"]},
    }

    document = {
        "islands": islands_store,
        "environment": environment_store,
        "immigration": immigration_store,
        "richness": richness_store,
        **nodes,
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "islands": "map[string,map[string,float]]",
                    "richness": "map[string,float]",
                    "time": "float",
                }
            },
            "inputs": {
                "islands": ["islands"],
                "richness": ["richness"],
                "time": ["global_time"],
            },
        },
    }
    return document


def archipelago_emitter_paths(n_islands: int):
    """Dotted store-paths a persisted ParquetEmitter should keep for studies."""
    return ["islands", "richness"]
