"""Composite generator: the canonical MacArthur-Wilson equilibrium archipelago.

Independent islands, each a :class:`MacArthurWilsonEquilibriumProcess` drawing
from a shared mainland pool of ``n_pool`` species. Islands span a (log-spaced)
range of areas and a (linear-spaced) range of isolation distances, so a single
run yields:

* the **rate-crossing equilibrium** and **turnover** on each island
  (``colonization_min == colonization_max`` style replicates: set
  ``area_min == area_max`` and ``distance_min == distance_max`` to make all
  islands identical-but-independent replicates), and
* the **species-area law** S* ~ A**z and the **isolation decline** across the
  gradient of islands (vary the area / distance ranges).

Unlike :func:`island_archipelago` there is no inter-island migration coupler:
the original equilibrium theory treats each island as an independent draw from
the mainland, so the islands here are decoupled on purpose. The LV archipelago
remains the model for interaction-driven, coupled dynamics.
"""

from __future__ import annotations

import numpy as np
from process_bigraph.composite_generator import composite_generator

from ..core import register_processes
from ..types import register_types


def _log_space(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1 or lo <= 0 or hi <= 0:
        return [hi] * max(1, n)
    return list(np.logspace(np.log10(lo), np.log10(hi), n))


def _lin_space(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1:
        return [lo] * max(1, n)
    return list(np.linspace(lo, hi, n))


def build_mw_document(
    *,
    n_islands: int,
    n_pool: int,
    colonization_base: float,
    extinction_base: float,
    area_min: float,
    area_max: float,
    distance_min: float,
    distance_max: float,
    area_exponent: float,
    interval: float = 1.0,
    seed: int = 0,
):
    """Assemble the process-bigraph document for a MacArthur-Wilson archipelago."""
    areas = _log_space(area_min, area_max, n_islands)
    distances = _lin_space(distance_min, distance_max, n_islands)
    ref_area = max(area_max, area_min, 1.0)
    species_list = [f"Species_{i}" for i in range(n_pool)]

    presence_store: dict[str, dict] = {}
    richness_store: dict[str, float] = {}
    colonizations_store: dict[str, float] = {}
    extinctions_store: dict[str, float] = {}
    island_meta: dict[str, dict] = {}
    nodes: dict[str, dict] = {}

    for idx in range(n_islands):
        name = f"island_{idx:02d}"
        area = float(areas[idx])
        distance = float(distances[idx])
        # Lay islands on a ring at radius = distance so the spatial viz reads;
        # distance from the mainland (origin) equals the isolation distance.
        angle = 2.0 * np.pi * idx / max(1, n_islands)
        x = distance * float(np.cos(angle))
        y = distance * float(np.sin(angle))

        presence_store[name] = {s: 0.0 for s in species_list}
        richness_store[name] = 0.0
        colonizations_store[name] = 0.0
        extinctions_store[name] = 0.0
        island_meta[name] = {"area": area, "distance": distance, "x": x, "y": y}

        nodes[name] = {
            "_type": "process",
            "address": "local:MacArthurWilsonEquilibriumProcess",
            "config": {
                "name": name,
                "n_pool": n_pool,
                "colonization_base": colonization_base,
                "extinction_base": extinction_base,
                "area": area,
                "distance": distance,
                "area_exponent": area_exponent,
                "ref_area": ref_area,
                "x": x,
                "y": y,
                "seed": seed + idx,
            },
            "interval": interval,
            "inputs": {"presence": ["presence", name]},
            "outputs": {
                "presence": ["presence", name],
                "richness": ["richness", name],
                "cum_colonizations": ["colonizations", name],
                "cum_extinctions": ["extinctions", name],
            },
        }

    document = {
        "presence": presence_store,
        "richness": richness_store,
        "colonizations": colonizations_store,
        "extinctions": extinctions_store,
        "island_meta": island_meta,
        **nodes,
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "richness": "map[string,float]",
                    "colonizations": "map[string,float]",
                    "extinctions": "map[string,float]",
                    "presence": "map[string,map[string,float]]",
                    "island_meta": "map[string,map[string,float]]",
                    "time": "float",
                }
            },
            "inputs": {
                "richness": ["richness"],
                "colonizations": ["colonizations"],
                "extinctions": ["extinctions"],
                "presence": ["presence"],
                "island_meta": ["island_meta"],
                "time": ["global_time"],
            },
        },
    }
    return document


@composite_generator(
    name="mw_equilibrium",
    description=(
        "Canonical MacArthur-Wilson equilibrium model: independent islands "
        "drawing from a mainland species pool, each relaxing to S* set by area "
        "(extinction) and isolation (colonization). One run yields the "
        "rate-crossing equilibrium, the species-area power law, and turnover."
    ),
    parameters={
        "n_islands": {"type": "integer", "default": 24,
                      "description": "Number of independent islands"},
        "n_pool": {"type": "integer", "default": 60,
                   "description": "Mainland species-pool size P"},
        "colonization_base": {"type": "float", "default": 0.02,
                              "description": "Per-species colonization prob at distance 0"},
        "extinction_base": {"type": "float", "default": 0.012,
                            "description": "Per-species extinction prob at the reference (largest) area"},
        "area_min": {"type": "float", "default": 1.0,
                     "description": "Smallest island area (log-spaced across islands)"},
        "area_max": {"type": "float", "default": 100.0,
                     "description": "Largest island area (= reference area for extinction)"},
        "distance_min": {"type": "float", "default": 0.0,
                         "description": "Least isolated island distance (linear across islands)"},
        "distance_max": {"type": "float", "default": 0.0,
                         "description": "Most isolated island distance (0 = isolation held fixed)"},
        "area_exponent": {"type": "float", "default": 0.25,
                          "description": "Area-extinction exponent; sets the species-area slope z"},
        "interval": {"type": "float", "default": 1.0,
                     "description": "Generations per tick (discrete model: keep 1.0)"},
        "seed": {"type": "integer", "default": 0,
                 "description": "RNG seed; each island uses seed + its index"},
    },
    emitters=[{
        "address": "local:ParquetEmitter",
        "paths": ["richness", "colonizations", "extinctions", "presence", "island_meta"],
    }],
    core_extensions=[register_types, register_processes],
)
def mw_equilibrium(
    core=None,
    *,
    n_islands=24,
    n_pool=60,
    colonization_base=0.02,
    extinction_base=0.012,
    area_min=1.0,
    area_max=100.0,
    distance_min=0.0,
    distance_max=0.0,
    area_exponent=0.25,
    interval=1.0,
    seed=0,
):
    return build_mw_document(
        n_islands=n_islands,
        n_pool=n_pool,
        colonization_base=colonization_base,
        extinction_base=extinction_base,
        area_min=area_min,
        area_max=area_max,
        distance_min=distance_min,
        distance_max=distance_max,
        area_exponent=area_exponent,
        interval=interval,
        seed=seed,
    )
