"""The Brown & Kodric-Brown (1977) *rescue effect* as a process-bigraph process.

A follow-up to :class:`~viva_island_biogeography.processes.MacArthurWilsonEquilibriumProcess`
that couples the two MacArthur-Wilson curves. In the canonical equilibrium
model the immigration curve ``I(S) = (P - S)*c`` and the extinction curve
``E(S) = S*e`` are **independent**: colonization pressure only ever creates new
occupancies, never props up the ones already there. Brown & Kodric-Brown
(*Ecology* 58:445-449, 1977; citekey ``brown1977turnover``) pointed out that a
population receiving a steady trickle of immigrant propagules is far less likely
to blink out — immigration also *lowers extinction*. They called it the
**rescue effect**, and it is the mechanism behind the shallower-than-expected
species-area/isolation relationships they measured on the boulder-field
"islands" of spiny pocket mice.

Mechanism (how the two curves couple here)
-------------------------------------------
Same mainland pool of ``n_pool`` species, same present/absent occupancy on one
island, same colonization pressure ``c = colonization_base / (1 + distance)``
and baseline extinction ``e = extinction_base * (ref_area/area)**area_exponent``
as :class:`MacArthurWilsonEquilibriumProcess`. The one change: the same
colonization pressure ``c`` *also arrives at species that are already present*.
Each generation, every **present** species draws a rescue event with
probability ``c``; a species rescued this tick has its extinction probability
cut to ``e*(1 - rescue_strength)`` (``rescue_strength`` in ``[0, 1]``).

With ``rescue_strength = 0`` the rescue draw never changes anything and the
model reduces **exactly** to the baseline MacArthur-Wilson equilibrium.

Analytic consequence
---------------------
A present species goes extinct in a tick with probability
``e_eff = c*e*(1-rescue_strength) + (1-c)*e = e*(1 - c*rescue_strength)`` —
immigration shaves the effective extinction rate. The per-species two-state
chain then equilibrates at ``P(present) = c / (c + e_eff)``, so

    S*(rescue) = P * c / (c + e*(1 - c*rescue_strength))

which rises monotonically with ``rescue_strength`` (richness up) while the
equilibrium event flux ``S* * e_eff`` falls (turnover down) — exactly Brown &
Kodric-Brown's prediction, and what the ``rescue-effect`` study measures.
"""

from __future__ import annotations

import random

import numpy as np
from process_bigraph import Process

from .processes import colonization_prob, extinction_prob, mw_equilibrium_richness


def rescue_equilibrium_richness(n_pool, c, e, rescue_strength):
    """Analytic rescue-effect equilibrium S* = P*c / (c + e*(1 - c*rescue_strength)).

    Reduces to :func:`mw_equilibrium_richness` at ``rescue_strength == 0``.
    """
    e_eff = e * (1.0 - c * float(rescue_strength))
    return mw_equilibrium_richness(n_pool, c, e_eff)


class RescueEffectProcess(Process):
    """One island under the MacArthur-Wilson model *with* the rescue effect.

    Identical to :class:`MacArthurWilsonEquilibriumProcess` (presence/absence of
    ``n_pool`` pool species on a single island; absent species colonize at
    ``c = colonization_base/(1+distance)``; present species go extinct at the
    area-dependent base ``e``) **except** that the incoming colonization pressure
    also rescues present populations: each tick a present species is rescued with
    probability ``c`` and, if rescued, its extinction probability is reduced to
    ``e*(1 - rescue_strength)``.

    Outputs mirror :class:`MacArthurWilsonEquilibriumProcess` exactly (all
    deltas, so the stores accumulate — the mass/length delta precedent), which
    lets the study reuse :func:`viva_island_biogeography.viz.turnover_timeseries`
    and :func:`viva_island_biogeography.viz.mw_rate_crossing` unchanged.
    """

    config_schema = {
        "name": {"_type": "string", "_default": "island"},
        "n_pool": {"_type": "integer", "_default": 60},
        "colonization_base": {"_type": "float", "_default": 0.5},
        "extinction_base": {"_type": "float", "_default": 0.5},
        "area": {"_type": "float", "_default": 100.0},
        "distance": {"_type": "float", "_default": 0.0},
        "area_exponent": {"_type": "float", "_default": 0.25},
        "ref_area": {"_type": "float", "_default": 100.0},
        "rescue_strength": {"_type": "float", "_default": 0.0},
        "x": {"_type": "float", "_default": 0.0},
        "y": {"_type": "float", "_default": 0.0},
        "seed": {"_type": "integer", "_default": 0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.n_pool = int(self.config["n_pool"])
        self.species_list = [f"Species_{i}" for i in range(self.n_pool)]
        self.c = colonization_prob(self.config["distance"], self.config["colonization_base"])
        self.e = extinction_prob(
            self.config["area"], self.config["extinction_base"],
            self.config["area_exponent"], self.config["ref_area"])
        self.rescue_strength = float(self.config["rescue_strength"])
        # extinction probability for a present species that gets rescued this tick
        self.e_rescued = self.e * (1.0 - self.rescue_strength)
        self._rng = random.Random((self.config["seed"], self.config["name"]).__hash__())
        self._prev_richness = 0

    def inputs(self):
        return {"presence": "map[string,float]"}

    def outputs(self):
        return {
            "presence": "map[string,float]",
            "richness": "float",
            "cum_colonizations": "float",
            "cum_extinctions": "float",
        }

    def initial_state(self):
        return {"presence": {s: 0.0 for s in self.species_list}}

    def update(self, state, interval):
        presence = dict(state.get("presence", {}) or {})
        delta: dict[str, float] = {}
        colonizations = 0
        extinctions = 0
        for s in self.species_list:
            present = float(presence.get(s, 0.0)) > 0.0
            if present:
                # The same colonization pressure c also arrives for present
                # species; a rescued species has its extinction prob reduced.
                rescued = self._rng.random() < self.c
                e_eff = self.e_rescued if rescued else self.e
                if self._rng.random() < e_eff:
                    delta[s] = -1.0
                    extinctions += 1
            else:
                if self._rng.random() < self.c:
                    delta[s] = 1.0
                    colonizations += 1

        richness_now = sum(
            1 for s in self.species_list
            if float(presence.get(s, 0.0)) + delta.get(s, 0.0) > 0.0)
        d_richness = float(richness_now - self._prev_richness)
        self._prev_richness = richness_now

        return {
            "presence": delta,
            "richness": d_richness,
            "cum_colonizations": float(colonizations),
            "cum_extinctions": float(extinctions),
        }


def register_rescue(core):
    """Register :class:`RescueEffectProcess` into ``core`` (returns ``core``)."""
    core.register_link("RescueEffectProcess", RescueEffectProcess)
    return core


# ----------------------------------------------------------------------
# Rescue-effect figures (import the shared theme/palette from ..viz)
# ----------------------------------------------------------------------
import plotly.graph_objects as go  # noqa: E402

from .viz import CATEGORICAL, MUTED, SURFACE, _apply_theme, write_html  # noqa: E402,F401


def rescue_richness_vs_strength(strengths, s_obs, *, s_star=None,
                                title="Rescue effect raises equilibrium richness"):
    """Equilibrium richness rising with rescue strength: simulated ensemble mean
    (markers) over the analytic S*(rescue) curve (dashed)."""
    strengths = np.asarray(strengths, dtype=float)
    s_obs = np.asarray(s_obs, dtype=float)
    fig = go.Figure()
    if s_star is not None:
        s_star = np.asarray(s_star, dtype=float)
        order = np.argsort(strengths)
        fig.add_trace(go.Scatter(
            x=strengths[order], y=s_star[order], mode="lines",
            name="analytic  S* = P·c/(c + e(1−c·r))",
            line=dict(color=CATEGORICAL[7], width=2, dash="dash"),
            hovertemplate="rescue=%{x:.2f}<br>S*=%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=strengths, y=s_obs, mode="markers+lines", name="simulated (ensemble mean)",
        marker=dict(color=CATEGORICAL[0], size=12, line=dict(color=SURFACE, width=2)),
        line=dict(color=CATEGORICAL[0], width=1.5),
        hovertemplate="rescue=%{x:.2f}<br>S=%{y:.2f}<extra></extra>"))
    fig.update_xaxes(title="rescue strength  r")
    fig.update_yaxes(title="equilibrium richness  S", rangemode="tozero")
    return _apply_theme(fig, title)


def rescue_turnover_vs_strength(strengths, turnover, *,
                                title="Rescue effect damps turnover"):
    """Equilibrium turnover (gross colonization+extinction events per generation)
    falling with rescue strength."""
    strengths = np.asarray(strengths, dtype=float)
    turnover = np.asarray(turnover, dtype=float)
    order = np.argsort(strengths)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strengths[order], y=turnover[order], mode="markers+lines",
        name="gross events · generation⁻¹",
        marker=dict(color=CATEGORICAL[1], size=12, line=dict(color=SURFACE, width=2)),
        line=dict(color=CATEGORICAL[1], width=1.5),
        hovertemplate="rescue=%{x:.2f}<br>turnover=%{y:.2f} events/gen<extra></extra>"))
    fig.update_xaxes(title="rescue strength  r")
    fig.update_yaxes(title="turnover  (events · generation⁻¹)", rangemode="tozero")
    return _apply_theme(fig, title)


def rescue_rate_crossing(*, n_pool, c, e, rescue_strength,
                         title="Rescue effect lowers the extinction curve"):
    """The coupled-curve picture: immigration I(S)=(P−S)·c with the baseline
    extinction line E(S)=S·e and the rescue-lowered line E'(S)=S·e(1−c·r). The
    crossing S* slides right (richness up) as the extinction curve is pulled
    down."""
    s_axis = np.arange(0, n_pool + 1)
    i_theory = (n_pool - s_axis) * c
    e_no = s_axis * e
    e_rescue = s_axis * e * (1.0 - c * float(rescue_strength))
    sstar_no = mw_equilibrium_richness(n_pool, c, e)
    sstar_rescue = rescue_equilibrium_richness(n_pool, c, e, rescue_strength)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_axis, y=i_theory, mode="lines",
                             name="immigration  I(S) = (P−S)·c",
                             line=dict(color=CATEGORICAL[0], width=2.5)))
    fig.add_trace(go.Scatter(x=s_axis, y=e_no, mode="lines",
                             name="extinction  E(S) = S·e  (no rescue)",
                             line=dict(color=CATEGORICAL[7], width=2.5)))
    fig.add_trace(go.Scatter(x=s_axis, y=e_rescue, mode="lines",
                             name=f"extinction  E′(S) = S·e(1−c·r)  (r={rescue_strength:g})",
                             line=dict(color=CATEGORICAL[2], width=2.5, dash="dash")))
    fig.add_vline(x=sstar_no, line=dict(color=MUTED, dash="dot"),
                  annotation_text=f"S*₀ = {sstar_no:.1f}", annotation_position="top left")
    fig.add_vline(x=sstar_rescue, line=dict(color=CATEGORICAL[2], dash="dot"),
                  annotation_text=f"S*(r) = {sstar_rescue:.1f}", annotation_position="top right")
    fig.update_xaxes(title="species richness  S")
    fig.update_yaxes(title="rate  (events · generation⁻¹)", rangemode="tozero")
    return _apply_theme(fig, title)
