"""Composite generator: the food-web interaction variant.

Same archipelago topology as :mod:`archipelago`, but the species-interaction
matrix is derived from a randomly-generated trophic food web (producers ->
primary consumers -> secondary consumers) with antisymmetric predator-prey
couplings, rather than a symmetric random matrix. Faithful port of
``experiments/island_food_web.py`` (``generate_food_web`` +
``generate_interaction_matrix_from_food_web``).
"""

from __future__ import annotations

import random

import numpy as np
from process_bigraph.composite_generator import composite_generator

from ..core import register_processes
from ..types import register_types
from ._common import build_archipelago_document


def generate_food_web(species_list, rng: random.Random):
    """Assign species to trophic levels and build predator->prey links.

    Port of the reference ``generate_food_web``.
    """
    trophic_levels = {"producers": [], "primary_consumers": [], "secondary_consumers": []}
    for species in species_list:
        chosen = rng.choice(list(trophic_levels.keys()))
        trophic_levels[chosen].append(species)

    food_web = {}
    producers = trophic_levels["producers"]
    primary = trophic_levels["primary_consumers"]
    for consumer in primary:
        if producers:
            food_web[consumer] = rng.sample(producers, rng.randint(1, len(producers)))
    for consumer in trophic_levels["secondary_consumers"]:
        if primary:
            food_web[consumer] = rng.sample(primary, rng.randint(1, len(primary)))
    return food_web


def interaction_matrix_from_food_web(species_list, food_web, predator_prey_interaction=0.1):
    """Antisymmetric predator-prey interaction matrix (reference port)."""
    n = len(species_list)
    matrix = np.zeros((n, n))
    index = {s: i for i, s in enumerate(species_list)}
    for predator, prey_list in food_web.items():
        p = index[predator]
        for prey in prey_list:
            q = index[prey]
            matrix[p][q] = predator_prey_interaction   # prey helps predator
            matrix[q][p] = -predator_prey_interaction  # predator hurts prey
    return matrix


@composite_generator(
    name="island_food_web",
    description=(
        "Food-web island biogeography: identical archipelago topology, but species "
        "interactions come from a random trophic food web (producers/primary/secondary "
        "consumers) with antisymmetric predator-prey couplings."
    ),
    parameters={
        "n_islands": {"type": "integer", "default": 4,
                      "description": "Total islands including the mainland"},
        "n_species": {"type": "integer", "default": 30,
                      "description": "Size of the global species pool"},
        "growth_rate": {"type": "float", "default": 0.03,
                        "description": "Mean per-generation Lotka-Volterra growth rate"},
        "predator_prey_interaction": {"type": "float", "default": 0.1,
                                      "description": "Magnitude of each predator-prey coupling"},
        "introduction_probability": {"type": "float", "default": 0.05,
                                      "description": "Per-island per-generation chance of a new-species introduction"},
        "interval": {"type": "float", "default": 1.0,
                     "description": "Generations per tick (discrete model: keep 1.0)"},
        "seed": {"type": "integer", "default": 0,
                 "description": "RNG seed for reproducible food web + dynamics"},
    },
    emitters=[{
        "address": "local:ParquetEmitter",
        "paths": ["islands", "richness"],
    }],
    core_extensions=[register_types, register_processes],
)
def island_food_web(
    core=None,
    *,
    n_islands=4,
    n_species=30,
    growth_rate=0.03,
    predator_prey_interaction=0.1,
    introduction_probability=0.05,
    interval=1.0,
    seed=0,
):
    species_list = [f"Species_{i}" for i in range(n_species)]
    web_rng = random.Random(seed)
    food_web = generate_food_web(species_list, web_rng)
    interaction_matrix = interaction_matrix_from_food_web(
        species_list, food_web, predator_prey_interaction)
    return build_archipelago_document(
        n_islands=n_islands,
        n_species=n_species,
        growth_rate=growth_rate,
        introduction_probability=introduction_probability,
        interaction_matrix=interaction_matrix,
        interval=interval,
        seed=seed,
    )
