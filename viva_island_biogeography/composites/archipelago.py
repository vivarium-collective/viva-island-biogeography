"""Composite generator: the classic Lotka-Volterra island archipelago.

A mainland plus ``n_islands - 1`` smaller, more distant islands, each running
the discrete Lotka-Volterra dynamics of :class:`IslandProcess`, coupled by a
single distance-stochastic :class:`MigrationProcess`. Interactions are a random
matrix (reference ``generate_random_interaction_matrix``). This is the direct
port of ``experiments/island_lotka_volterra.py::run_island_map``.
"""

from __future__ import annotations

import random

from process_bigraph.composite_generator import composite_generator

from ..processes import generate_random_interaction_matrix
from ._common import build_archipelago_document, archipelago_emitter_paths


@composite_generator(
    name="island_archipelago",
    description=(
        "Lotka-Volterra island biogeography: a mainland + smaller distant islands, "
        "each with discrete LV population dynamics, coupled by distance-stochastic "
        "inter-island migration and random species introduction."
    ),
    parameters={
        "n_islands": {"type": "integer", "default": 4,
                      "description": "Total islands including the mainland"},
        "n_species": {"type": "integer", "default": 30,
                      "description": "Size of the global species pool"},
        "growth_rate": {"type": "float", "default": 0.03,
                        "description": "Mean per-generation Lotka-Volterra growth rate"},
        "interaction_weight": {"type": "float", "default": 0.1,
                               "description": "Spread of the random interaction matrix"},
        "introduction_probability": {"type": "float", "default": 0.01,
                                      "description": "Per-island per-generation chance of a new-species introduction"},
        "interval": {"type": "float", "default": 1.0,
                     "description": "Generations per tick (discrete model: keep 1.0)"},
        "seed": {"type": "integer", "default": 0,
                 "description": "RNG seed for reproducible stochastic dynamics"},
    },
    emitters=[{
        "address": "local:ParquetEmitter",
        "paths": ["islands", "richness"],
    }],
)
def island_archipelago(
    core=None,
    *,
    n_islands=4,
    n_species=30,
    growth_rate=0.03,
    interaction_weight=0.1,
    introduction_probability=0.01,
    interval=1.0,
    seed=0,
):
    import numpy as np
    rng = np.random.default_rng(seed)
    interaction_matrix = generate_random_interaction_matrix(
        n_species, weight=interaction_weight, rng=rng)
    return build_archipelago_document(
        n_islands=n_islands,
        n_species=n_species,
        growth_rate=growth_rate,
        introduction_probability=introduction_probability,
        interaction_matrix=interaction_matrix,
        interval=interval,
        seed=seed,
    )
