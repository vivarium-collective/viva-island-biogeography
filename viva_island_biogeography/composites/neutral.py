"""Composite generator: Hubbell's (2001) zero-sum neutral metacommunity.

Independent islands over a log-spaced area gradient, each a
:class:`~viva_island_biogeography.models_neutral.NeutralMetacommunityProcess`
holding a fixed ``J ∝ area`` local community that evolves by Moran
dispersal-drift from a shared metacommunity of ``n_pool`` species. There is no
inter-island coupling (as with :func:`mw_equilibrium`, each island is an
independent draw) and — crucially — no colonization/extinction rate asymmetry:
any species-area relationship here is produced by drift + dispersal alone, the
neutral-theory counterpoint to the MacArthur-Wilson equilibrium composite.

Mirrors ``mw_equilibrium.py``'s structure exactly: a per-island abundance store
seeded with ``J`` immigrants, a richness store, an ``island_meta`` store, one
RAMEmitter in the document, and a persisted ParquetEmitter keeping
``richness``/``abundance``/``island_meta``.
"""

from __future__ import annotations

import random

import numpy as np
from process_bigraph.composite_generator import composite_generator

from ..models_neutral import draw_community, register_neutral
from ..types import register_types


def _log_space(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1 or lo <= 0 or hi <= 0:
        return [hi] * max(1, n)
    return list(np.logspace(np.log10(lo), np.log10(hi), n))


def _lin_space(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1:
        return [lo] * max(1, n)
    return list(np.linspace(lo, hi, n))


def build_neutral_document(
    *,
    n_islands: int,
    n_pool: int,
    density: float,
    immigration_base: float,
    area_min: float,
    area_max: float,
    distance_min: float,
    distance_max: float,
    interval: float = 1.0,
    seed: int = 0,
):
    """Assemble the process-bigraph document for a neutral-metacommunity archipelago."""
    areas = _log_space(area_min, area_max, n_islands)
    distances = _lin_space(distance_min, distance_max, n_islands)
    species_list = [f"Species_{i}" for i in range(n_pool)]

    abundance_store: dict[str, dict] = {}
    richness_store: dict[str, float] = {}
    island_meta: dict[str, dict] = {}
    nodes: dict[str, dict] = {}

    for idx in range(n_islands):
        name = f"island_{idx:02d}"
        area = float(areas[idx])
        distance = float(distances[idx])
        local_J = max(1, int(round(area * density)))
        # Lay islands on a ring at radius = distance so the spatial viz reads;
        # distance from the mainland (origin) equals the isolation distance.
        angle = 2.0 * np.pi * idx / max(1, n_islands)
        x = distance * float(np.cos(angle))
        y = distance * float(np.sin(angle))

        # Seed the initial local community by drawing J immigrants (zero-sum).
        seed_rng = random.Random((seed + idx, name, "seed_community").__hash__())
        abundance_store[name] = draw_community(local_J, species_list, seed_rng)
        richness_store[name] = 0.0
        island_meta[name] = {
            "area": area, "distance": distance, "x": x, "y": y, "J": float(local_J),
        }

        nodes[name] = {
            "_type": "process",
            "address": "local:NeutralMetacommunityProcess",
            "config": {
                "name": name,
                "n_pool": n_pool,
                "density": density,
                "area": area,
                "distance": distance,
                "immigration_base": immigration_base,
                "seed": seed + idx,
            },
            "interval": interval,
            "inputs": {"abundance": ["abundance", name]},
            "outputs": {
                "abundance": ["abundance", name],
                "richness": ["richness", name],
            },
        }

    document = {
        "abundance": abundance_store,
        "richness": richness_store,
        "island_meta": island_meta,
        **nodes,
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "richness": "map[string,float]",
                    "abundance": "map[string,map[string,float]]",
                    "island_meta": "map[string,map[string,float]]",
                    "time": "float",
                }
            },
            "inputs": {
                "richness": ["richness"],
                "abundance": ["abundance"],
                "island_meta": ["island_meta"],
                "time": ["global_time"],
            },
        },
    }
    return document


@composite_generator(
    name="neutral_metacommunity",
    description=(
        "Hubbell (2001) zero-sum neutral metacommunity: independent fixed-J "
        "islands over an area gradient, each evolving by Moran dispersal-drift "
        "from a shared species pool. Richness and the species-abundance "
        "distribution emerge with no niche differences and no colonization/"
        "extinction rate asymmetry — the neutral counterpoint to MacArthur-Wilson."
    ),
    parameters={
        "n_islands": {"type": "integer", "default": 12,
                      "description": "Number of independent islands"},
        "n_pool": {"type": "integer", "default": 64,
                   "description": "Metacommunity species-pool size"},
        "density": {"type": "float", "default": 2.0,
                    "description": "Individuals per unit area (sets local size J = area*density)"},
        "immigration_base": {"type": "float", "default": 0.2,
                             "description": "Per-replacement immigration probability m at distance 0"},
        "area_min": {"type": "float", "default": 1.0,
                     "description": "Smallest island area (log-spaced across islands)"},
        "area_max": {"type": "float", "default": 100.0,
                     "description": "Largest island area"},
        "distance_min": {"type": "float", "default": 0.0,
                         "description": "Least isolated island distance (linear across islands)"},
        "distance_max": {"type": "float", "default": 0.0,
                         "description": "Most isolated island distance (0 = isolation held fixed)"},
        "interval": {"type": "float", "default": 1.0,
                     "description": "Generations per tick (discrete model: keep 1.0)"},
        "seed": {"type": "integer", "default": 0,
                 "description": "RNG seed; each island uses seed + its index"},
    },
    emitters=[{
        "address": "local:ParquetEmitter",
        "paths": ["richness", "abundance", "island_meta"],
    }],
    core_extensions=[register_types, register_neutral],
)
def neutral_metacommunity(
    core=None,
    *,
    n_islands=12,
    n_pool=64,
    density=2.0,
    immigration_base=0.2,
    area_min=1.0,
    area_max=100.0,
    distance_min=0.0,
    distance_max=0.0,
    interval=1.0,
    seed=0,
):
    return build_neutral_document(
        n_islands=n_islands,
        n_pool=n_pool,
        density=density,
        immigration_base=immigration_base,
        area_min=area_min,
        area_max=area_max,
        distance_min=distance_min,
        distance_max=distance_max,
        interval=interval,
        seed=seed,
    )
