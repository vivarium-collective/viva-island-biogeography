"""Hubbell's (2001) zero-sum neutral metacommunity, as a process-bigraph Process.

A deliberately *neutral* counterpart to the phenomenological MacArthur-Wilson
equilibrium model in :mod:`viva_island_biogeography.processes`. Where
MacArthur & Wilson set per-species colonization/extinction probabilities from
isolation and area, the neutral theory of Hubbell (2001, *The Unified Neutral
Theory of Biodiversity and Biogeography*) assumes **ecological equivalence**:
every individual, of every species, has identical per-capita birth, death and
dispersal rates. Diversity on an island is then a dynamic balance between
**drift** (demographic stochasticity in a finite community) and **dispersal**
(immigration from a shared metacommunity / species pool).

The local community is **zero-sum**: it holds a fixed number of individuals
``J = max(1, round(area * density))`` (the island's "size" scales with area).
Each generation runs ``J`` Moran death/replacement events:

* a random resident dies — drawn uniformly over the ``J`` individuals, which is
  exactly "weighted by local abundance" (common species lose more individuals);
* the vacancy is filled either by an **immigrant** from the metacommunity, with
  probability ``m = immigration_base / (1 + distance)`` (isolation lowers
  immigration, the same distance kernel the rest of the workspace uses), or, with
  probability ``1 - m``, by the **offspring of a random local individual**.

The metacommunity is treated as a uniform pool of ``n_pool`` species (the
maximal-diversity limit); the skewed local species-abundance distribution that
emerges is produced by dispersal-drift alone — the headline neutral result.
Because ``J`` is fixed, larger islands sample and retain more species before
drift erodes them, so a positive species-area relationship emerges with **no**
colonization/extinction rate asymmetry at all. This is the counterpoint the
neutral-vs-equilibrium study tests against MacArthur-Wilson.

Delta-emission convention (shared with :class:`MacArthurWilsonEquilibriumProcess`):
the process emits per-species abundance **deltas** and a richness **delta**; the
stores accumulate them to absolute values (the mass/length delta precedent used
throughout this workspace).
"""

from __future__ import annotations

import math
import random

import numpy as np
import plotly.graph_objects as go
from process_bigraph import Process

from .viz import CATEGORICAL, MUTED, _apply_theme, write_html  # noqa: F401 (re-export)


def immigration_prob(distance: float, base: float) -> float:
    """Per-replacement immigration probability m = base / (1 + distance).

    The same isolation decay the Lotka-Volterra migration kernel and the
    MacArthur-Wilson colonization rate use, so distance lowers dispersal
    consistently across every model in this workspace. ``base`` is the
    immigration probability of a mainland (distance 0) island.
    """
    return float(base) / (1.0 + max(0.0, float(distance)))


def draw_community(n_individuals: int, species_list: list[str],
                   rng: random.Random) -> dict[str, float]:
    """Seed a zero-sum local community of ``n_individuals`` by drawing that many
    immigrants uniformly from the metacommunity ``species_list``.

    Returns a per-species count map (only present species), summing to
    ``n_individuals`` — used by the composite builder to seed each island.
    """
    counts: dict[str, float] = {}
    for _ in range(max(0, int(n_individuals))):
        s = rng.choice(species_list)
        counts[s] = counts.get(s, 0.0) + 1.0
    return counts


class NeutralMetacommunityProcess(Process):
    """One island under Hubbell's (2001) zero-sum neutral metacommunity.

    A fixed-size ``J = max(1, round(area * density))`` local community evolves by
    ``J`` Moran death/replacement events per generation (see module docstring).
    Species are ecologically equivalent; richness and the species-abundance
    distribution emerge from dispersal-drift balance, not from niche or rate
    differences.

    Inputs
    ------
    abundance : map[string, float]
        Current per-species counts on this island (read from the shared store;
        seeded with ``J`` immigrants by the composite builder). Sums to ``J``.

    Outputs (deltas, so the stores accumulate to absolute values)
    -------
    abundance : map[string, float]
        Net per-species count change over this generation's ``J`` events.
    richness : float
        Local richness delta (species present now minus the previously recorded
        count), emitted against the previous reading so the ``float`` store
        telescopes to the current number of species present.
    """

    config_schema = {
        "name": {"_type": "string", "_default": "island"},
        "n_pool": {"_type": "integer", "_default": 64},
        "density": {"_type": "float", "_default": 2.0},
        "area": {"_type": "float", "_default": 100.0},
        "distance": {"_type": "float", "_default": 0.0},
        "immigration_base": {"_type": "float", "_default": 0.2},
        "seed": {"_type": "integer", "_default": 0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.n_pool = int(self.config["n_pool"])
        self.species_list = [f"Species_{i}" for i in range(self.n_pool)]
        self.J = max(1, int(round(float(self.config["area"]) * float(self.config["density"]))))
        self.m = immigration_prob(self.config["distance"], self.config["immigration_base"])
        self._rng = random.Random((self.config["seed"], self.config["name"]).__hash__())
        self._prev_richness = 0

    def inputs(self):
        return {"abundance": "map[string,float]"}

    def outputs(self):
        return {
            "abundance": "map[string,float]",
            "richness": "float",
        }

    def initial_state(self):
        # The composite builder seeds the real J-individual community into the
        # shared store; return an empty default so this does not clobber it. If
        # the store ever arrives empty, update() seeds J immigrants defensively.
        return {"abundance": {}}

    def _draw_immigrant(self) -> str:
        return self._rng.choice(self.species_list)

    def update(self, state, interval):
        abundance = dict(state.get("abundance", {}) or {})

        # Rebuild the flat list of J individuals from the integer counts in the
        # store, so a random index is an individual drawn weighted by abundance.
        individuals: list[str] = []
        for s, c in abundance.items():
            n = int(round(float(c)))
            if n > 0:
                individuals.extend([s] * n)
        # Enforce the fixed zero-sum size J (pads from the metacommunity only if
        # the store under-seeded, e.g. an empty first tick; never over-counts).
        if len(individuals) < self.J:
            while len(individuals) < self.J:
                individuals.append(self._draw_immigrant())
        elif len(individuals) > self.J:
            individuals = individuals[:self.J]

        # J Moran death/replacement events (zero-sum: size stays J).
        rng = self._rng
        m = self.m
        n = len(individuals)
        for _ in range(self.J):
            dead = rng.randrange(n)
            if rng.random() < m:
                individuals[dead] = self._draw_immigrant()       # immigration
            else:
                individuals[dead] = individuals[rng.randrange(n)]  # local birth

        # New absolute counts and the net per-species delta vs the pre-tick store.
        new_counts: dict[str, float] = {}
        for s in individuals:
            new_counts[s] = new_counts.get(s, 0.0) + 1.0
        delta: dict[str, float] = {}
        for s in set(abundance) | set(new_counts):
            d = new_counts.get(s, 0.0) - float(abundance.get(s, 0.0))
            if d != 0.0:
                delta[s] = d

        richness_now = len(new_counts)  # every entry has count >= 1
        d_richness = float(richness_now - self._prev_richness)
        self._prev_richness = richness_now

        return {"abundance": delta, "richness": d_richness}


def register_neutral(core):
    """Register :class:`NeutralMetacommunityProcess` into ``core`` (return core)."""
    core.register_link("NeutralMetacommunityProcess", NeutralMetacommunityProcess)
    return core


# --------------------------------------------------------------------------
# Visualizations for the neutral-vs-equilibrium study
# (theme/palette/write_html imported from the shared viz module)
# --------------------------------------------------------------------------
def _power_fit(A, S):
    """Return (z, log_c, mask) for a log-log power-law fit S = c * A**z."""
    A = np.asarray(A, dtype=float)
    S = np.asarray(S, dtype=float)
    mask = S > 0
    if mask.sum() < 2 or np.ptp(np.log(A[mask])) <= 0:
        return 0.0, 0.0, mask
    z, logc = np.polyfit(np.log(A[mask]), np.log(S[mask]), 1)
    return float(z), float(logc), mask


def neutral_vs_equilibrium_sar(areas, S_neutral, S_equilibrium, z_neutral,
                               z_equilibrium, *,
                               title="Species–area relationship: neutral vs equilibrium"):
    """Overlay the emergent species-area relationships of the two models on
    log-log axes, each with its own fitted power-law slope ``z``.

    Two mechanisms, one macroecological law: dispersal-drift (neutral) and
    colonization-extinction (MacArthur-Wilson equilibrium) both produce a
    positive S ∝ A**z, letting the reader compare the slopes directly.
    """
    A = np.asarray(areas, dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=A, y=np.asarray(S_neutral, dtype=float), mode="markers",
        name="neutral (dispersal–drift)",
        marker=dict(color=CATEGORICAL[2], size=12, line=dict(color="#fcfcfb", width=2)),
        hovertemplate="area=%{x:.1f}<br>S=%{y:.2f}<extra>neutral</extra>"))
    fig.add_trace(go.Scatter(
        x=A, y=np.asarray(S_equilibrium, dtype=float), mode="markers",
        name="equilibrium (MacArthur–Wilson)",
        marker=dict(color=CATEGORICAL[0], size=11, symbol="square",
                    line=dict(color="#fcfcfb", width=2)),
        hovertemplate="area=%{x:.1f}<br>S=%{y:.2f}<extra>equilibrium</extra>"))

    for S, color, z, label in (
        (S_neutral, CATEGORICAL[2], z_neutral, "neutral fit"),
        (S_equilibrium, CATEGORICAL[0], z_equilibrium, "equilibrium fit"),
    ):
        _, logc, mask = _power_fit(A, S)
        if mask.sum() >= 2:
            xf = np.linspace(A[mask].min(), A[mask].max(), 60)
            yf = np.exp(logc) * xf ** z
            fig.add_trace(go.Scatter(
                x=xf, y=yf, mode="lines", name=f"{label}  z = {z:.3f}",
                line=dict(color=color, dash="dash", width=2),
                hovertemplate="fit<extra></extra>"))

    fig.update_xaxes(title="island area  (log scale)", type="log")
    fig.update_yaxes(title="species richness  S  (log scale)", type="log")
    return _apply_theme(fig, f"{title}  (neutral z = {z_neutral:.3f}, equilibrium z = {z_equilibrium:.3f})")


def neutral_rank_abundance(abundance, island_name, *, title=None):
    """Whittaker rank-abundance curve for one neutral island: species ranked
    from most to least abundant, abundance on a log y-axis.

    The characteristic concave neutral curve (a few common species, a long rare
    tail) emerges from drift alone under a uniform metacommunity.
    """
    counts = sorted((float(c) for c in abundance.values() if c > 0), reverse=True)
    ranks = list(range(1, len(counts) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ranks, y=counts, mode="lines+markers", name=island_name,
        line=dict(color=CATEGORICAL[2], width=2),
        marker=dict(color=CATEGORICAL[2], size=7, line=dict(color="#fcfcfb", width=1)),
        hovertemplate="rank=%{x}<br>abundance=%{y:.0f}<extra></extra>"))
    fig.update_xaxes(title="species rank (most → least abundant)")
    fig.update_yaxes(title="abundance  (log scale)", type="log")
    return _apply_theme(fig, title or f"Rank–abundance (Whittaker) — {island_name}")


def neutral_preston_octaves(abundance, island_name, *, title=None):
    """Preston-style species-abundance distribution: species binned into
    log2 abundance octaves. The internal mode + rare/common tails are the
    signature of Hubbell's zero-sum multinomial local community.
    """
    counts = [float(c) for c in abundance.values() if c > 0]
    octaves: dict[int, int] = {}
    for c in counts:
        o = int(math.floor(math.log2(c))) if c >= 1 else 0
        octaves[o] = octaves.get(o, 0) + 1
    xs = sorted(octaves)
    labels = [(f"{2 ** o}" if o == 0 else f"{2 ** o}–{2 ** (o + 1) - 1}") for o in xs]
    ys = [octaves[o] for o in xs]
    fig = go.Figure(go.Bar(
        x=labels, y=ys, marker=dict(color=CATEGORICAL[2], line=dict(color="#fcfcfb", width=1)),
        hovertemplate="octave=%{x} individuals<br>species=%{y}<extra></extra>"))
    fig.update_xaxes(title="abundance octave  (individuals per species)")
    fig.update_yaxes(title="number of species", rangemode="tozero")
    return _apply_theme(fig, title or f"Preston species-abundance octaves — {island_name}")
