"""The General Dynamic Model of oceanic island biogeography (Whittaker,
Triantis & Ladle 2008).

Where :class:`~viva_island_biogeography.processes.MacArthurWilsonEquilibriumProcess`
holds island area *fixed* and lets richness relax to a single static
equilibrium S*, the General Dynamic Model (GDM) adds **island ontogeny**: an
oceanic (volcanic) island is born, grows as it is built up, then subsides and
erodes back beneath the sea over millions of years. Area therefore traces a
humped trajectory ``area(t)``, and because extinction falls with area the
instantaneous equilibrium ``S*(t) = P·c / (c + e(t))`` is itself humped. Species
richness chases this moving target with a lag, so the GDM predicts an
**emergent humped richness trajectory** — richness rises to an overshoot while
the island is large and young, then *declines* as the island erodes — rather
than the flat MacArthur-Wilson plateau.

:class:`GeneralDynamicModelProcess` is deliberately identical to the canonical
MacArthur-Wilson process (a mainland pool of ``n_pool`` species, each present or
absent, colonizing absent species at ``c`` and removing present species at
``e``) EXCEPT that area is time-varying. Each tick the island advances its own
tick counter, evaluates ``area(t)`` from the ontogeny profile, recomputes
``e(t) = extinction_base·(ref_area/area(t))**area_exponent`` from the *current*
area, and emits the current area so the viz can plot ``area(t)`` against
richness. Larger area => lower extinction => higher instantaneous S*.

Ontogeny profile (a rescaled log-normal/Gamma-like hump)::

    area(t) = area_max · ( (t/t_peak)·exp(1 − t/t_peak) ) ** ontogeny_shape

which is 0 at ``t = 0``, rises to ``area_max`` exactly at ``t = t_peak``, and
erodes back toward 0 as ``t → ∞`` — the classic GDM island life-cycle.

Reference: Whittaker, R. J., Triantis, K. A. & Ladle, R. J. (2008). A general
dynamic theory of oceanic island biogeography. *Journal of Biogeography* 35(6),
977-994. (citekey ``whittaker2008general``).

All outputs follow this workspace's delta-emission convention (the mass/length
delta precedent): processes emit per-tick deltas so the stores accumulate to the
absolute values. ``presence`` emits ±1 occupancy changes, ``richness`` emits the
change in present-species count, and ``area`` emits the change in area versus the
previous tick so the plain-float store holds the absolute ``area(t)``.
"""

from __future__ import annotations

import math
import random

import numpy as np
import plotly.graph_objects as go
from process_bigraph import Process

from .processes import colonization_prob, extinction_prob
from .viz import (  # shared palette + theme helpers
    CATEGORICAL,
    GRID,
    INK_SECONDARY,
    MUTED,
    SEQUENTIAL_BLUE,
    _apply_theme,
    history_times,  # noqa: F401  (re-exported for study runners)
    write_html,  # noqa: F401  (re-exported for study runners)
)


# ----------------------------------------------------------------------
# Ontogeny area profile
# ----------------------------------------------------------------------
def gdm_area(t: float, area_max: float, t_peak: float, shape: float) -> float:
    """Humped island-ontogeny area at (model) time ``t``.

    ``area(t) = area_max * ((t/t_peak) * exp(1 - t/t_peak)) ** shape``.

    Zero at ``t=0``, equal to ``area_max`` at ``t=t_peak`` (the factor is 1
    there), and decaying back toward 0 as ``t`` grows past the peak. ``shape``
    sharpens (``>1``) or broadens (``<1``) the hump.
    """
    t = float(t)
    if t_peak <= 0.0:
        return float(area_max)
    r = t / float(t_peak)
    base = r * math.exp(1.0 - r)  # peaks at r=1 with value 1.0
    if base <= 0.0:
        return 0.0
    return float(area_max) * (base ** float(shape))


class GeneralDynamicModelProcess(Process):
    """One oceanic island under the General Dynamic Model (Whittaker et al. 2008).

    Identical colonization/extinction bookkeeping to
    :class:`~viva_island_biogeography.processes.MacArthurWilsonEquilibriumProcess`
    — a mainland pool of ``n_pool`` species, each present (1) or absent (0);
    every absent species colonizes with probability ``c = colonization_base`` and
    every present species goes extinct with probability ``e(t)`` — **except** the
    island's area is time-varying to model ontogeny. Each update advances the
    island's own tick counter, evaluates ``area(t)`` from
    :func:`gdm_area`, and recomputes
    ``e(t) = extinction_base·(ref_area/area(t))**area_exponent`` from the current
    area (larger area => lower extinction). The instantaneous equilibrium
    ``S*(t)=P·c/(c+e(t))`` is therefore humped, and richness chases it with a lag
    — producing the predicted overshoot-then-decline.

    Outputs (deltas, so the stores accumulate to absolute values):

    presence : map[string, float]
        Per-species occupancy delta (+1 colonization, -1 extinction).
    richness : float
        Change in present-species count; the float store accumulates to the
        absolute richness.
    area : float
        Change in island area versus the previous tick; the float store
        accumulates to the absolute ``area(t)`` so the viz can plot area directly.
    """

    config_schema = {
        "name": {"_type": "string", "_default": "island"},
        "n_pool": {"_type": "integer", "_default": 60},
        "colonization_base": {"_type": "float", "_default": 0.05},
        "extinction_base": {"_type": "float", "_default": 0.01},
        "area_max": {"_type": "float", "_default": 100.0},
        "t_peak": {"_type": "float", "_default": 150.0},
        "ontogeny_shape": {"_type": "float", "_default": 2.0},
        "area_exponent": {"_type": "float", "_default": 1.0},
        "ref_area": {"_type": "float", "_default": 100.0},
        "seed": {"_type": "integer", "_default": 0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.n_pool = int(self.config["n_pool"])
        self.species_list = [f"Species_{i}" for i in range(self.n_pool)]
        # Colonization is set by the mainland pool only (isolation held at 0 for
        # the ontogeny demonstration); constant through the island's life.
        self.c = colonization_prob(0.0, self.config["colonization_base"])
        self.area_max = float(self.config["area_max"])
        self.t_peak = float(self.config["t_peak"])
        self.shape = float(self.config["ontogeny_shape"])
        self.area_exponent = float(self.config["area_exponent"])
        self.ref_area = float(self.config["ref_area"])
        self.extinction_base = float(self.config["extinction_base"])
        self._rng = random.Random((self.config["seed"], self.config["name"]).__hash__())
        self._tick = 0
        self._prev_richness = 0
        self._prev_area = 0.0

    def inputs(self):
        return {"presence": "map[string,float]"}

    def outputs(self):
        return {
            "presence": "map[string,float]",
            "richness": "float",
            "area": "float",
        }

    def initial_state(self):
        return {"presence": {s: 0.0 for s in self.species_list}}

    def update(self, state, interval):
        # Advance the island's own geological clock (discrete model: one tick =
        # one model-time unit; first update is t=1 so area is strictly positive).
        self._tick += 1
        t = float(self._tick)
        area_t = gdm_area(t, self.area_max, self.t_peak, self.shape)
        e = extinction_prob(area_t, self.extinction_base,
                            self.area_exponent, self.ref_area)
        c = self.c

        presence = dict(state.get("presence", {}) or {})
        delta: dict[str, float] = {}
        for s in self.species_list:
            present = float(presence.get(s, 0.0)) > 0.0
            if present:
                if self._rng.random() < e:
                    delta[s] = -1.0
            else:
                if self._rng.random() < c:
                    delta[s] = 1.0

        richness_now = sum(
            1 for s in self.species_list
            if float(presence.get(s, 0.0)) + delta.get(s, 0.0) > 0.0)
        d_richness = float(richness_now - self._prev_richness)
        self._prev_richness = richness_now

        # Emit the AREA DELTA vs the previous tick so the plain-float store
        # accumulates to the absolute area(t) (the delta-accumulation convention).
        d_area = float(area_t - self._prev_area)
        self._prev_area = area_t

        return {"presence": delta, "richness": d_richness, "area": d_area}


def register_gdm(core):
    """Register :class:`GeneralDynamicModelProcess` into ``core``."""
    core.register_link("GeneralDynamicModelProcess", GeneralDynamicModelProcess)
    return core


# ----------------------------------------------------------------------
# Visualizations (GDM-specific; shared theme/palette from ..viz)
# ----------------------------------------------------------------------
def gdm_area_richness_overlay(times, area_series, richness_series, *,
                              title="Island ontogeny: area and species richness"):
    """Dual-axis overlay of ``area(t)`` (left) and richness ``S(t)`` (right).

    Both are humped, but richness peaks *after* area and decays more slowly —
    the lag that drives the hysteresis loop.
    """
    t = np.asarray(times, dtype=float)
    a = np.asarray(area_series, dtype=float)
    s = np.asarray(richness_series, dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=a, mode="lines", name="island area A(t)",
        line=dict(color=CATEGORICAL[1], width=2.5),
        hovertemplate="t=%{x:.0f}<br>area=%{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t, y=s, mode="lines", name="species richness S(t) (right axis)",
        line=dict(color=CATEGORICAL[0], width=2.5), yaxis="y2",
        hovertemplate="t=%{x:.0f}<br>richness=%{y:.1f}<extra></extra>"))
    i_area = int(np.argmax(a))
    i_rich = int(np.argmax(s))
    fig.add_vline(x=t[i_area], line=dict(color=CATEGORICAL[1], dash="dot", width=1),
                  annotation_text="area peak", annotation_position="top")
    fig.add_vline(x=t[i_rich], line=dict(color=CATEGORICAL[0], dash="dash", width=1),
                  annotation_text="richness peak", annotation_position="bottom")
    fig.update_layout(
        yaxis2=dict(title="species richness S", overlaying="y", side="right",
                    rangemode="tozero", showgrid=False,
                    title_font=dict(color=INK_SECONDARY), tickfont=dict(color=MUTED)))
    fig.update_xaxes(title="model time  t  (island age)")
    fig.update_yaxes(title="island area  A", rangemode="tozero")
    return _apply_theme(fig, title)


def gdm_hysteresis_loop(area_series, richness_series, times, *,
                        title="Richness–area hysteresis over island ontogeny"):
    """Richness vs area as a time-colored trajectory: richness lags area, so the
    rising (young, growing island) and falling (old, eroding island) limbs do
    not coincide — a hysteresis loop traversed once over the island's life."""
    a = np.asarray(area_series, dtype=float)
    s = np.asarray(richness_series, dtype=float)
    t = np.asarray(times, dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=a, y=s, mode="lines+markers",
        line=dict(color="rgba(42,120,214,0.35)", width=1.5),
        marker=dict(size=5, color=t, colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c]
                                                  for i, c in enumerate(SEQUENTIAL_BLUE)],
                    colorbar=dict(title="model<br>time t", tickfont=dict(color=MUTED)),
                    line=dict(width=0)),
        name="trajectory",
        hovertemplate="area=%{x:.1f}<br>richness=%{y:.1f}<br>t=%{marker.color:.0f}<extra></extra>"))
    # Mark growth vs erosion limbs by the area-peak split.
    i_area = int(np.argmax(a))
    fig.add_trace(go.Scatter(
        x=[a[i_area]], y=[s[i_area]], mode="markers",
        marker=dict(size=13, color=CATEGORICAL[1], symbol="star",
                    line=dict(color="#fcfcfb", width=1)),
        name="area peak"))
    fig.update_xaxes(title="island area  A(t)")
    fig.update_yaxes(title="species richness  S", rangemode="tozero")
    return _apply_theme(fig, title)


def gdm_ensemble_richness(times, richness_matrix, *,
                          title="Ensemble richness trajectory (overshoot then decline)"):
    """Ensemble-mean richness trajectory with a ±1σ band over seeds, plus the
    analytic mean peak marked. ``richness_matrix`` is ``(n_seeds, n_times)``."""
    t = np.asarray(times, dtype=float)
    M = np.asarray(richness_matrix, dtype=float)
    mean = M.mean(axis=0)
    sd = M.std(axis=0)
    fig = go.Figure()
    # variance band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([mean + sd, (mean - sd)[::-1]]),
        fill="toself", fillcolor="rgba(42,120,214,0.15)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name="±1σ across seeds", showlegend=True))
    for k in range(M.shape[0]):
        fig.add_trace(go.Scatter(
            x=t, y=M[k], mode="lines", line=dict(color=GRID, width=0.6),
            hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=t, y=mean, mode="lines", name="ensemble-mean richness",
        line=dict(color=CATEGORICAL[0], width=3),
        hovertemplate="t=%{x:.0f}<br>mean richness=%{y:.2f}<extra></extra>"))
    i_peak = int(np.argmax(mean))
    fig.add_vline(x=t[i_peak], line=dict(color=MUTED, dash="dash"),
                  annotation_text=f"peak S≈{mean[i_peak]:.0f} at t={t[i_peak]:.0f}",
                  annotation_position="top")
    fig.update_xaxes(title="model time  t  (island age)")
    fig.update_yaxes(title="species richness  S", rangemode="tozero")
    return _apply_theme(fig, title)
