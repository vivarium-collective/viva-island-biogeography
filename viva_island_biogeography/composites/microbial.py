"""Composite generator: the MacArthur-Wilson equilibrium archipelago in a
**microbial** parameter regime.

Microbial island biogeography (Martiny et al. 2006 *Nat. Rev. Microbiol.*;
Horner-Devine et al. 2004 *Nature*; Bell et al. 2005 tree-hole bacteria;
Hanson et al. 2012 *Nat. Rev. Microbiol.*) asks the same two questions the
macro-ecological theory does — does richness grow with area, and does it fall
with isolation — but the empirical answers differ in *degree*:

* the mainland species pool is enormous (``n_pool`` ~ hundreds+ rather than
  dozens), so islands rarely saturate it;
* per-taxon colonization **and** extinction are fast (high dispersal plus high
  local turnover), so richness turns over far more quickly at equilibrium; and
* the observed taxa-area slope is **flatter**: z ~ 0.02-0.1 for microbes versus
  the classic macro z ~ 0.15-0.35 (Horner-Devine 2004; Bell 2005).

No new process is introduced: this reuses the already-registered
:class:`MacArthurWilsonEquilibriumProcess` (via :func:`build_mw_document` from
:mod:`.mw_equilibrium`) with a microbial default regime — a large pool, higher
colonization/extinction, and a lower effective ``area_exponent`` that yields the
observed shallow microbial taxa-area slope. The same tunable parameters as
``mw_equilibrium`` are exposed, so a macro comparison regime is a parameter swap
rather than a different model.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from process_bigraph.composite_generator import composite_generator

from ..core import register_processes
from ..types import register_types
from ..viz import CATEGORICAL, MUTED, SURFACE, _apply_theme, write_html  # noqa: F401
from .mw_equilibrium import build_mw_document


@composite_generator(
    name="microbial_island",
    description=(
        "MacArthur-Wilson equilibrium model in a microbial regime: a large "
        "mainland taxa pool, fast colonization + extinction (high turnover), and "
        "a shallow area-extinction exponent giving the observed flat microbial "
        "taxa-area slope (z ~ 0.02-0.1; Horner-Devine 2004, Bell 2005). Reuses "
        "MacArthurWilsonEquilibriumProcess with no new process."
    ),
    parameters={
        "n_islands": {"type": "integer", "default": 12,
                      "description": "Number of independent islands (habitat patches)"},
        "n_pool": {"type": "integer", "default": 300,
                   "description": "Mainland taxa-pool size P (large for microbes)"},
        "colonization_base": {"type": "float", "default": 0.10,
                              "description": "Per-taxon colonization prob at distance 0 (fast dispersal)"},
        "extinction_base": {"type": "float", "default": 0.06,
                            "description": "Per-taxon extinction prob at the reference area (fast local turnover)"},
        "area_min": {"type": "float", "default": 1.0,
                     "description": "Smallest patch area (log-spaced across islands)"},
        "area_max": {"type": "float", "default": 100.0,
                     "description": "Largest patch area (= reference area for extinction)"},
        "distance_min": {"type": "float", "default": 0.0,
                         "description": "Least isolated patch distance (linear across islands)"},
        "distance_max": {"type": "float", "default": 0.0,
                         "description": "Most isolated patch distance (0 = isolation held fixed)"},
        "area_exponent": {"type": "float", "default": 0.10,
                          "description": "Area-extinction exponent; low -> flat microbial taxa-area slope z"},
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
def microbial_island(
    core=None,
    *,
    n_islands=12,
    n_pool=300,
    colonization_base=0.10,
    extinction_base=0.06,
    area_min=1.0,
    area_max=100.0,
    distance_min=0.0,
    distance_max=0.0,
    area_exponent=0.10,
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


# --------------------------------------------------------------------------
# Microbial-specific figures (macro-vs-microbe comparison)
# --------------------------------------------------------------------------
def _fit_z(A, S):
    """Log-log power-law slope z of a taxa-area relationship (positive S only)."""
    A = np.asarray(A, dtype=float)
    S = np.asarray(S, dtype=float)
    mask = S > 0
    if mask.sum() < 2 or np.ptp(np.log(A[mask])) == 0:
        return 0.0, 0.0
    z, logc = np.polyfit(np.log(A[mask]), np.log(S[mask]), 1)
    return float(z), float(logc)


def taxa_area_overlay(macro, micro, *, title="Taxa–area relationship: microbes vs macro-organisms"):
    """Overlay the macro and microbial taxa-area relationships on one log-log
    plot, each with its fitted power law S ∝ A^z.

    ``macro`` / ``micro`` are dicts ``{"A": areas, "S": richness, "label": str}``.
    Flatter microbial z appears directly as the shallower fitted line.
    """
    fig = go.Figure()
    regimes = [
        ("macro", macro, CATEGORICAL[1]),     # orange
        ("micro", micro, CATEGORICAL[0]),     # blue
    ]
    for _key, d, color in regimes:
        A = np.asarray(d["A"], dtype=float)
        S = np.asarray(d["S"], dtype=float)
        label = d["label"]
        z, logc = _fit_z(A, S)
        fig.add_trace(go.Scatter(
            x=A, y=S, mode="markers", name=f"{label} (data)",
            marker=dict(color=color, size=11, line=dict(color=SURFACE, width=1.5)),
            hovertemplate=f"<b>{label}</b><br>area=%{{x:.1f}}<br>S=%{{y:.1f}}<extra></extra>",
        ))
        mask = S > 0
        xfit = np.linspace(A[mask].min(), A[mask].max(), 60)
        yfit = np.exp(logc) * xfit ** z
        fig.add_trace(go.Scatter(
            x=xfit, y=yfit, mode="lines",
            name=f"{label}:  S ∝ A^z,  z = {z:.3f}",
            line=dict(color=color, dash="dash", width=2.5),
            hovertemplate="fit<extra></extra>",
        ))
    fig.update_xaxes(title="patch / island area  (log scale)", type="log")
    fig.update_yaxes(title="equilibrium taxon richness  S  (log scale)", type="log")
    return _apply_theme(fig, title)


def turnover_comparison(macro_turnover, micro_turnover, *,
                        title="Equilibrium turnover: microbes vs macro-organisms"):
    """Grouped stacked bars of the gross per-generation turnover rate
    (colonizations + extinctions per island per generation) at equilibrium.

    Each argument is a dict ``{"colonization": rate, "extinction": rate}``.
    The taller microbial bar is the fast-turnover signature.
    """
    regimes = ["macro-organisms", "microbes"]
    col = [macro_turnover["colonization"], micro_turnover["colonization"]]
    ext = [macro_turnover["extinction"], micro_turnover["extinction"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=regimes, y=col, name="colonizations", marker_color=CATEGORICAL[0],
        hovertemplate="<b>%{x}</b><br>colonizations=%{y:.3f} · gen⁻¹ · island⁻¹<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=regimes, y=ext, name="extinctions", marker_color=CATEGORICAL[7],
        hovertemplate="<b>%{x}</b><br>extinctions=%{y:.3f} · gen⁻¹ · island⁻¹<extra></extra>",
    ))
    totals = [c + e for c, e in zip(col, ext)]
    for xi, tot in zip(regimes, totals):
        fig.add_annotation(x=xi, y=tot, text=f"{tot:.2f}", showarrow=False,
                           yshift=12, font=dict(color=MUTED, size=12))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="gross events · generation⁻¹ · island⁻¹", rangemode="tozero")
    return _apply_theme(fig, title)


def richness_distribution(macro_S, micro_S, *,
                          title="Equilibrium richness across islands"):
    """Box + jittered points of per-island equilibrium richness for each regime —
    the microbial pool supports a far larger (and more variable) standing richness.
    """
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=list(macro_S), name="macro-organisms", boxpoints="all", jitter=0.5,
        marker_color=CATEGORICAL[1], line=dict(color=CATEGORICAL[1]),
        hovertemplate="macro S=%{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Box(
        y=list(micro_S), name="microbes", boxpoints="all", jitter=0.5,
        marker_color=CATEGORICAL[0], line=dict(color=CATEGORICAL[0]),
        hovertemplate="microbe S=%{y:.1f}<extra></extra>",
    ))
    fig.update_xaxes(title="")
    fig.update_yaxes(title="equilibrium taxon richness  S", rangemode="tozero")
    return _apply_theme(fig, title)
