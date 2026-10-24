"""Composite generator: a rescue-effect archipelago (Brown & Kodric-Brown 1977).

Independent islands at a **shared area and isolation**, differing only in their
``rescue_strength`` — linearly spaced from ``rescue_min`` to ``rescue_max``
across the islands. Each island is a
:class:`~viva_island_biogeography.models_rescue.RescueEffectProcess` drawing from
the same mainland pool of ``n_pool`` species. Because area, distance and the
base colonization/extinction rates are held fixed, the only thing that varies
along the archipelago is how strongly incoming propagules rescue the species
already present, so a single run sweeps the coupling between the two
MacArthur-Wilson curves:

* ``rescue_min == rescue_max`` makes every island an independent replicate at one
  rescue strength (use two such composites, at ``0`` and at ``0.6``-``0.8``, for
  a matched no-rescue / with-rescue comparison), and
* ``rescue_min != rescue_max`` lays out a rescue gradient, so one run shows the
  equilibrium richness *rising* and the turnover *falling* with rescue strength.

With ``rescue_max == rescue_min == 0`` the composite reduces exactly to the
independent-curves MacArthur-Wilson equilibrium (the baseline of the
``mw_equilibrium`` composite at matched rates), which the study uses as its
sanity check.
"""

from __future__ import annotations

import numpy as np
from process_bigraph.composite_generator import composite_generator

from ..models_rescue import register_rescue
from ..types import register_types


def _lin_space(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1:
        return [lo] * max(1, n)
    return list(np.linspace(lo, hi, n))


def build_rescue_document(
    *,
    n_islands: int,
    n_pool: int,
    colonization_base: float,
    extinction_base: float,
    area: float,
    distance: float,
    area_exponent: float,
    rescue_min: float,
    rescue_max: float,
    ref_area: float | None = None,
    interval: float = 1.0,
    seed: int = 0,
):
    """Assemble the process-bigraph document for a rescue-effect archipelago."""
    strengths = _lin_space(rescue_min, rescue_max, n_islands)
    if ref_area is None:
        ref_area = float(area)
    species_list = [f"Species_{i}" for i in range(n_pool)]

    presence_store: dict[str, dict] = {}
    richness_store: dict[str, float] = {}
    colonizations_store: dict[str, float] = {}
    extinctions_store: dict[str, float] = {}
    island_meta: dict[str, dict] = {}
    nodes: dict[str, dict] = {}

    for idx in range(n_islands):
        name = f"island_{idx:02d}"
        rescue_strength = float(strengths[idx])
        # Spread islands along a line so the spatial viz reads; geometry is
        # cosmetic here (area and distance are identical across islands).
        x = float(idx)
        y = 0.0

        presence_store[name] = {s: 0.0 for s in species_list}
        richness_store[name] = 0.0
        colonizations_store[name] = 0.0
        extinctions_store[name] = 0.0
        island_meta[name] = {
            "area": float(area), "distance": float(distance),
            "rescue_strength": rescue_strength, "x": x, "y": y,
        }

        nodes[name] = {
            "_type": "process",
            "address": "local:RescueEffectProcess",
            "config": {
                "name": name,
                "n_pool": n_pool,
                "colonization_base": colonization_base,
                "extinction_base": extinction_base,
                "area": float(area),
                "distance": float(distance),
                "area_exponent": area_exponent,
                "ref_area": ref_area,
                "rescue_strength": rescue_strength,
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
    name="rescue_effect",
    description=(
        "Brown & Kodric-Brown (1977) rescue-effect archipelago: identical "
        "islands (same area and isolation) differing only in how strongly "
        "incoming propagules rescue present species from extinction. Coupling "
        "the immigration and extinction curves raises equilibrium richness and "
        "damps turnover; rescue_strength=0 reduces to the independent-curves "
        "MacArthur-Wilson equilibrium."
    ),
    parameters={
        "n_islands": {"type": "integer", "default": 9,
                      "description": "Number of islands (rescue-strength levels, or replicates)"},
        "n_pool": {"type": "integer", "default": 60,
                   "description": "Mainland species-pool size P"},
        "colonization_base": {"type": "float", "default": 0.5,
                              "description": "Per-species colonization prob at distance 0 (also the rescue probability)"},
        "extinction_base": {"type": "float", "default": 0.5,
                            "description": "Per-species baseline extinction prob at the reference area"},
        "area": {"type": "float", "default": 100.0,
                 "description": "Island area (shared across all islands)"},
        "distance": {"type": "float", "default": 0.0,
                     "description": "Isolation distance (shared across all islands)"},
        "area_exponent": {"type": "float", "default": 0.25,
                          "description": "Area-extinction exponent"},
        "rescue_min": {"type": "float", "default": 0.0,
                       "description": "Rescue strength on the first island (0 = no rescue = baseline M-W)"},
        "rescue_max": {"type": "float", "default": 0.8,
                       "description": "Rescue strength on the last island (linear across islands)"},
        "interval": {"type": "float", "default": 1.0,
                     "description": "Generations per tick (discrete model: keep 1.0)"},
        "seed": {"type": "integer", "default": 0,
                 "description": "RNG seed; each island uses seed + its index"},
    },
    emitters=[{
        "address": "local:ParquetEmitter",
        "paths": ["richness", "colonizations", "extinctions", "presence", "island_meta"],
    }],
    core_extensions=[register_types, register_rescue],
)
def rescue_effect(
    core=None,
    *,
    n_islands=9,
    n_pool=60,
    colonization_base=0.5,
    extinction_base=0.5,
    area=100.0,
    distance=0.0,
    area_exponent=0.25,
    rescue_min=0.0,
    rescue_max=0.8,
    interval=1.0,
    seed=0,
):
    return build_rescue_document(
        n_islands=n_islands,
        n_pool=n_pool,
        colonization_base=colonization_base,
        extinction_base=extinction_base,
        area=area,
        distance=distance,
        area_exponent=area_exponent,
        rescue_min=rescue_min,
        rescue_max=rescue_max,
        interval=interval,
        seed=seed,
    )
