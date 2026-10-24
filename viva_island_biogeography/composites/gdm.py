"""Composite generator: the General Dynamic Model archipelago (Whittaker et al. 2008).

A small set of independent oceanic islands, each a
:class:`~viva_island_biogeography.models_gdm.GeneralDynamicModelProcess` drawing
from a shared mainland pool of ``n_pool`` species. Every island follows the same
colonization/extinction bookkeeping as :func:`mw_equilibrium`, but its area is
time-varying: it is born, grows to ``area_max`` at its own ``t_peak``, then
erodes away (island ontogeny). The islands here are staggered along a
``t_peak`` **age gradient** (``t_peak_min`` → ``t_peak_max``) so a single run
shows young, mid-life and old islands peaking at different times — the GDM
space-for-time logic made explicit.

Unlike the static MacArthur-Wilson composite, richness does NOT settle on a flat
plateau: because area (and therefore the instantaneous equilibrium
``S*(t)=P·c/(c+e(t))``) is humped, richness overshoots while the island is large
and young, then declines as it erodes. Mirrors
``viva_island_biogeography/composites/mw_equilibrium.py``.

Reference: Whittaker, Triantis & Ladle (2008), *J. Biogeography* 35:977-994
(citekey ``whittaker2008general``).
"""

from __future__ import annotations

import numpy as np
from process_bigraph.composite_generator import composite_generator

from ..models_gdm import register_gdm
from ..types import register_types


def _lin_space(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1:
        return [hi] * max(1, n)
    return list(np.linspace(lo, hi, n))


def build_gdm_document(
    *,
    n_islands: int,
    n_pool: int,
    colonization_base: float,
    extinction_base: float,
    area_max: float,
    t_peak_min: float,
    t_peak_max: float,
    ontogeny_shape: float,
    area_exponent: float,
    ref_area: float | None = None,
    interval: float = 1.0,
    seed: int = 0,
):
    """Assemble the process-bigraph document for a GDM archipelago (age gradient)."""
    t_peaks = _lin_space(t_peak_min, t_peak_max, n_islands)
    ref = float(ref_area) if ref_area else float(area_max)
    species_list = [f"Species_{i}" for i in range(n_pool)]

    presence_store: dict[str, dict] = {}
    richness_store: dict[str, float] = {}
    area_store: dict[str, float] = {}
    island_meta: dict[str, dict] = {}
    nodes: dict[str, dict] = {}

    for idx in range(n_islands):
        name = f"island_{idx:02d}"
        t_peak = float(t_peaks[idx])
        # Lay islands left-to-right by age for the spatial viz.
        x = float(idx)
        y = 0.0

        presence_store[name] = {s: 0.0 for s in species_list}
        richness_store[name] = 0.0
        area_store[name] = 0.0
        island_meta[name] = {
            "area_max": float(area_max),
            "t_peak": t_peak,
            "ontogeny_shape": float(ontogeny_shape),
            "area_exponent": float(area_exponent),
            "ref_area": ref,
            "x": x,
            "y": y,
        }

        nodes[name] = {
            "_type": "process",
            "address": "local:GeneralDynamicModelProcess",
            "config": {
                "name": name,
                "n_pool": n_pool,
                "colonization_base": colonization_base,
                "extinction_base": extinction_base,
                "area_max": area_max,
                "t_peak": t_peak,
                "ontogeny_shape": ontogeny_shape,
                "area_exponent": area_exponent,
                "ref_area": ref,
                "seed": seed + idx,
            },
            "interval": interval,
            "inputs": {"presence": ["presence", name]},
            "outputs": {
                "presence": ["presence", name],
                "richness": ["richness", name],
                "area": ["area", name],
            },
        }

    document = {
        "presence": presence_store,
        "richness": richness_store,
        "area": area_store,
        "island_meta": island_meta,
        **nodes,
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "richness": "map[string,float]",
                    "area": "map[string,float]",
                    "presence": "map[string,map[string,float]]",
                    "island_meta": "map[string,map[string,float]]",
                    "time": "float",
                }
            },
            "inputs": {
                "richness": ["richness"],
                "area": ["area"],
                "presence": ["presence"],
                "island_meta": ["island_meta"],
                "time": ["global_time"],
            },
        },
    }
    return document


@composite_generator(
    name="general_dynamic_model",
    description=(
        "General Dynamic Model of oceanic island biogeography (Whittaker, "
        "Triantis & Ladle 2008): a few islands with time-varying area (ontogeny) "
        "staggered along a t_peak age gradient. Each island grows then erodes, so "
        "species richness overshoots and then declines — a humped trajectory the "
        "static MacArthur-Wilson equilibrium cannot produce."
    ),
    parameters={
        "n_islands": {"type": "integer", "default": 3,
                      "description": "Number of independent islands (age gradient)"},
        "n_pool": {"type": "integer", "default": 60,
                   "description": "Mainland species-pool size P"},
        "colonization_base": {"type": "float", "default": 0.05,
                              "description": "Per-species colonization prob (isolation held at 0)"},
        "extinction_base": {"type": "float", "default": 0.01,
                            "description": "Per-species extinction prob at the reference (max) area"},
        "area_max": {"type": "float", "default": 100.0,
                     "description": "Peak island area (reached at t_peak)"},
        "t_peak_min": {"type": "float", "default": 120.0,
                       "description": "Youngest island's area-peak time"},
        "t_peak_max": {"type": "float", "default": 180.0,
                       "description": "Oldest island's area-peak time (age gradient)"},
        "ontogeny_shape": {"type": "float", "default": 2.0,
                           "description": "Sharpness of the area hump (>1 sharpens)"},
        "area_exponent": {"type": "float", "default": 1.0,
                          "description": "Area-extinction exponent (larger => steeper decline)"},
        "ref_area": {"type": "float", "default": 100.0,
                     "description": "Area at which extinction equals extinction_base"},
        "interval": {"type": "float", "default": 1.0,
                     "description": "Model-time units per tick (discrete model: keep 1.0)"},
        "seed": {"type": "integer", "default": 0,
                 "description": "RNG seed; each island uses seed + its index"},
    },
    emitters=[{
        "address": "local:ParquetEmitter",
        "paths": ["richness", "area", "presence", "island_meta"],
    }],
    core_extensions=[register_types, register_gdm],
)
def general_dynamic_model(
    core=None,
    *,
    n_islands=3,
    n_pool=60,
    colonization_base=0.05,
    extinction_base=0.01,
    area_max=100.0,
    t_peak_min=120.0,
    t_peak_max=180.0,
    ontogeny_shape=2.0,
    area_exponent=1.0,
    ref_area=100.0,
    interval=1.0,
    seed=0,
):
    return build_gdm_document(
        n_islands=n_islands,
        n_pool=n_pool,
        colonization_base=colonization_base,
        extinction_base=extinction_base,
        area_max=area_max,
        t_peak_min=t_peak_min,
        t_peak_max=t_peak_max,
        ontogeny_shape=ontogeny_shape,
        area_exponent=area_exponent,
        ref_area=ref_area,
        interval=interval,
        seed=seed,
    )
