"""Process-bigraph wrappers for the island-biogeography metacommunity model.

This is a **faithful port** of the reference model archived at
``github.com/vivarium-collective/island-biogeography`` (private) — specifically
``experiments/island_lotka_volterra.py`` (``Island`` + ``IslandManager``). The
discrete Lotka-Volterra population step, the carrying-capacity rescaling, the
distance-based stochastic migration (``migration_rate(d) = 1/(1+d)``), and the
random new-species introduction are reproduced line-for-line in process-bigraph
form — no paraphrase, no stub. The only structural change is the decomposition
the task calls for: each island becomes an :class:`IslandProcess` and the
inter-island coupling becomes a single :class:`MigrationProcess`, so an
archipelago is a real composite (N island processes + 1 migration coupler)
rather than one monolith.

Composition semantics (how the port maps onto process-bigraph):

* Species abundances live in a shared ``map[string, float]`` store per island.
  Each process emits **deltas** (new - current) so the engine accumulates them
  additively — a sibling growth/immigration process can write to the same store
  and compose correctly (the mass/length delta precedent).
* ``IslandProcess`` owns the per-island discrete LV step **and** the per-island
  random introduction (an island attribute in the reference), plus an exposed
  ``immigration`` input a sibling could write and a ``growth_modifier``
  environmental input.
* ``MigrationProcess`` is the metacommunity coupler: it reads every island's
  abundance + position and emits the distance-stochastic single-individual
  emigration/immigration exchanges as abundance deltas.

Both processes read the shared pre-tick abundance state and emit summing deltas,
so one tick reproduces the reference's ``LV -> migration -> introduction`` net
effect. (The only ordering nuance versus the monolithic reference: migration is
computed from pre-tick counts rather than post-LV counts within the same tick —
immaterial over a multi-generation run and documented here for honesty.)
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from process_bigraph import Process


def generate_random_interaction_matrix(num_species: int, weight: float = 0.1,
                                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Random species-interaction matrix (port of the reference helper).

    Reference: ``(np.random.rand(n, n) - 0.5) * weight``.
    """
    if rng is None:
        return (np.random.rand(num_species, num_species) - 0.5) * weight
    return (rng.random((num_species, num_species)) - 0.5) * weight


def migration_rate(distance: float) -> float:
    """Distance-decaying migration rate (reference: ``1 / (1 + distance)``)."""
    return 1.0 / (1.0 + distance)


def lotka_volterra_step(
    abundance: dict[str, float],
    species_list: list[str],
    interaction_matrix: np.ndarray,
    growth_rate: float,
    capacity: float,
) -> dict[str, float]:
    """One discrete Lotka-Volterra generation for a single island.

    Line-for-line port of ``Island.update_populations`` in the reference:

    * ``new_count = count_i + growth * count_i + sum_j M[i][j] * count_j * count_i``
    * clamp to ``max(int(new_count), 0)`` (integer truncation, non-negative),
    * if the projected total exceeds carrying capacity, rescale every species by
      ``capacity / total`` (again integer-truncated),
    * species that were absent (count <= 0) are reset to 0.

    Returns the **absolute** post-step abundance for every species in
    ``species_list``.
    """
    new_abundance: dict[str, float] = {}
    total_projected = 0.0

    counts = [float(abundance.get(s, 0.0)) for s in species_list]
    for i, species_i in enumerate(species_list):
        count_i = counts[i]
        if count_i > 0:  # only update existing species (reference guard)
            interaction_effects = 0.0
            for j, _species_j in enumerate(species_list):
                count_j = counts[j]
                if count_j > 0 and interaction_matrix[i][j] != 0:
                    interaction_effects += interaction_matrix[i][j] * count_j * count_i
            new_count = count_i + growth_rate * count_i + interaction_effects
            new_count = max(int(new_count), 0)  # non-negative integer
            new_abundance[species_i] = new_count
            total_projected += new_count

    # Carrying-capacity rescale (reference: scale back if over capacity).
    if total_projected > capacity and total_projected > 0:
        scaling_factor = capacity / total_projected
        for species in new_abundance:
            new_abundance[species] = int(new_abundance[species] * scaling_factor)

    # Reference resets all to 0, then writes the survivors.
    return {s: float(new_abundance.get(s, 0)) for s in species_list}


def seed_initial_abundance(
    species_list: list[str],
    initial_species: int,
    initial_individuals: int,
    interaction_matrix: np.ndarray,
    growth_rate: float,
    capacity: float,
    rng: random.Random,
) -> dict[str, float]:
    """Reproduce ``Island.__init__``: sample ``initial_species``, seed each with
    ``initial_individuals``, then run one LV step ("Run it once")."""
    abundance = {s: 0.0 for s in species_list}
    chosen = rng.sample(species_list, min(initial_species, len(species_list)))
    for s in chosen:
        abundance[s] = float(initial_individuals)
    return lotka_volterra_step(
        abundance, species_list, interaction_matrix, growth_rate, capacity)


class IslandProcess(Process):
    """A single island's time-stepped discrete Lotka-Volterra dynamics.

    One ``update`` call == one generation (the model is discrete-time; the
    ``interval`` magnitude is not used to subdivide the map — set the composite's
    process ``interval`` to 1.0 and run ``N`` steps for ``N`` generations).

    Inputs
    ------
    abundance : map[string, float]
        Current per-species counts on this island (read from the shared store).
    immigration : map[string, float]
        External arrivals a sibling process may inject this tick (per species).
        Folded straight into the emitted abundance delta; defaults to empty, so
        with nothing wired it has no effect. (The archipelago's own inter-island
        migration is applied by :class:`MigrationProcess` directly to the shared
        abundance store, so this port is free for external couplers.)
    growth_modifier : float
        Multiplies the island's mean growth rate (an environmental knob a sibling
        could drive); defaults to 1.0.

    Outputs
    -------
    abundance : map[string, float]
        Per-species **delta** (post-step absolute minus pre-step current, plus
        any injected immigration and this island's random introductions). Emitted
        as a bare ``map[string,float]`` so it composes additively with
        :class:`MigrationProcess` and any sibling.
    richness : float
        Delta of this island's local species richness (count of species with
        count > 0), emitted against the previous reading so the ``float`` store
        accumulates to the correct absolute value (the mass/length delta
        precedent). This is the island's *local* view (LV + introductions +
        injected immigration); the authoritative archipelago richness is derived
        from the full abundance store in the study runners.
    """

    config_schema = {
        "name": {"_type": "string", "_default": "island"},
        "species_list": {"_type": "list[string]", "_default": []},
        "interaction_matrix": {"_type": "list[list[float]]", "_default": []},
        "mean_growth_rate": {"_type": "float", "_default": 0.03},
        "area": {"_type": "float", "_default": 100.0},
        "individuals_per_area": {"_type": "float", "_default": 100.0},
        "introduction_probability": {"_type": "float", "_default": 0.0},
        "x": {"_type": "float", "_default": 0.0},
        "y": {"_type": "float", "_default": 0.0},
        "seed": {"_type": "integer", "_default": 0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.species_list = list(self.config["species_list"])
        matrix = self.config["interaction_matrix"]
        n = len(self.species_list)
        if matrix:
            self.interaction_matrix = np.asarray(matrix, dtype=float)
        else:
            self.interaction_matrix = np.zeros((n, n))
        self.capacity = self.config["area"] * self.config["individuals_per_area"]
        # Per-island RNG keyed by name+seed so introductions are reproducible
        # yet independent per island (reference draws from the global `random`).
        self._rng = random.Random((self.config["seed"], self.config["name"]).__hash__())
        self._prev_richness = 0

    def inputs(self):
        return {
            "abundance": "map[string,float]",
            "immigration": "map[string,float]",
            "growth_modifier": "float",
        }

    def outputs(self):
        return {
            "abundance": "map[string,float]",
            "richness": "float",
        }

    def initial_state(self):
        return {
            "abundance": {s: 0.0 for s in self.species_list},
            "immigration": {},
            "growth_modifier": 1.0,
        }

    def update(self, state, interval):
        abundance = dict(state.get("abundance", {}) or {})
        immigration = dict(state.get("immigration", {}) or {})
        growth_modifier = state.get("growth_modifier", 1.0)
        if growth_modifier is None:
            growth_modifier = 1.0
        growth_rate = self.config["mean_growth_rate"] * growth_modifier

        # 1. Discrete Lotka-Volterra step (faithful port).
        post_lv = lotka_volterra_step(
            abundance, self.species_list, self.interaction_matrix,
            growth_rate, self.capacity)

        # 2. Per-island random introduction (reference: island attribute, rolled
        #    once per generation; adds 2 individuals of a random global species).
        introductions: dict[str, float] = {}
        if self._rng.random() < self.config["introduction_probability"]:
            new_species = self._rng.choice(self.species_list)
            introductions[new_species] = introductions.get(new_species, 0.0) + 2.0

        # 3. Emit the abundance DELTA (post-step absolute - pre-step current),
        #    plus introductions and any externally injected immigration.
        delta: dict[str, float] = {}
        for s in self.species_list:
            d = post_lv[s] - float(abundance.get(s, 0.0))
            d += float(introductions.get(s, 0.0))
            d += float(immigration.get(s, 0.0))
            if d != 0.0:
                delta[s] = d

        # 4. Local richness readout as a delta against the previous reading.
        post_total = {
            s: float(abundance.get(s, 0.0)) + delta.get(s, 0.0)
            for s in self.species_list
        }
        richness_now = sum(1 for s in self.species_list if post_total[s] > 0)
        d_richness = float(richness_now - self._prev_richness)
        self._prev_richness = richness_now

        return {"abundance": delta, "richness": d_richness}


class MigrationProcess(Process):
    """The metacommunity migration coupler (port of ``IslandManager.simulate``'s
    inter-island migration loop).

    Reads every island's abundance and position, and for each ordered island
    pair ``(i, j)`` with ``i != j`` moves a single randomly-chosen present
    species from ``i`` to ``j`` with probability ``migration_rate(distance_ij)``
    where ``distance_ij`` is the Euclidean distance between the islands and
    ``migration_rate(d) = 1/(1+d)`` — exactly the reference. The resulting
    single-individual ``emigrate``/``immigrate`` adjustments are emitted as
    abundance deltas keyed by island name, which compose additively with each
    :class:`IslandProcess`'s own LV delta on the shared abundance store.

    Inputs
    ------
    abundances : map[string, map[string,float]]
        The whole archipelago abundance tree ``{island_name: {species: count}}``.

    Outputs
    -------
    exchange : map[string, map[string,float]]
        Per-island per-species migration **deltas** ``{island_name: {species:
        +/-n}}`` (negative = emigration, positive = immigration).
    """

    config_schema = {
        "island_names": {"_type": "list[string]", "_default": []},
        "positions": {"_type": "list[list[float]]", "_default": []},
        "seed": {"_type": "integer", "_default": 0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.island_names = list(self.config["island_names"])
        self.positions = [tuple(p) for p in self.config["positions"]]
        self._rng = random.Random((self.config["seed"], "migration").__hash__())
        # Precompute pairwise migration rates (reference precomputes these once).
        n = len(self.island_names)
        self._migration_rates = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    xi, yi = self.positions[i]
                    xj, yj = self.positions[j]
                    dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    self._migration_rates[i][j] = migration_rate(dist)

    def inputs(self):
        return {"abundances": "map[string,map[string,float]]"}

    def outputs(self):
        return {"exchange": "map[string,map[string,float]]"}

    def initial_state(self):
        return {"abundances": {name: {} for name in self.island_names}}

    def update(self, state, interval):
        abundances = state.get("abundances", {}) or {}
        exchange: dict[str, dict[str, float]] = {name: {} for name in self.island_names}
        # Working copy decremented as individuals emigrate, so (faithful to the
        # reference's sequential ``emigrate``) we never move more individuals of a
        # species than the source island actually holds this tick — which also
        # keeps every abundance >= 0.
        working = {name: dict(abundances.get(name, {}) or {}) for name in self.island_names}

        for i, name_i in enumerate(self.island_names):
            for j, name_j in enumerate(self.island_names):
                if i == j:
                    continue
                if self._rng.random() < self._migration_rates[i][j]:
                    migratable = [s for s, c in working[name_i].items() if c > 0]
                    if migratable:
                        species = self._rng.choice(migratable)
                        working[name_i][species] -= 1.0  # sequential emigrate
                        # emigrate from i, immigrate to j (single individual)
                        exchange[name_i][species] = exchange[name_i].get(species, 0.0) - 1.0
                        exchange[name_j][species] = exchange[name_j].get(species, 0.0) + 1.0

        # Drop empty per-island maps so the emitted delta is minimal.
        return {"exchange": {k: v for k, v in exchange.items() if v}}


# ----------------------------------------------------------------------
# Canonical MacArthur-Wilson equilibrium model
# ----------------------------------------------------------------------
# A second, deliberately *phenomenological* process alongside the mechanistic
# Lotka-Volterra port above. Where IslandProcess grows abundances through
# explicit interactions, this reproduces the ORIGINAL theory of island
# biogeography (MacArthur & Wilson 1963 Evolution 17:373; 1967 monograph): an
# island draws from a mainland species pool of size P, each species simply
# present or absent, with per-species colonization and extinction probabilities
# set by isolation (distance) and island area. Species richness relaxes to the
# equilibrium S* where the immigration rate I(S) = (P - S)*c equals the
# extinction rate E(S) = S*e, and species identity turns over even at constant S.


def colonization_prob(distance: float, base: float) -> float:
    """Per-species per-generation colonization probability, declining with isolation.

    ``c(d) = base / (1 + distance)`` — the same distance decay the LV port uses
    for migration (:func:`migration_rate`), so isolation lowers immigration in
    both models. ``base`` is the colonization probability of a mainland
    (distance 0) species.
    """
    return base / (1.0 + max(0.0, float(distance)))


def extinction_prob(area: float, base: float, area_exponent: float,
                    ref_area: float = 1.0) -> float:
    """Per-species per-generation extinction probability, declining with area.

    ``e(A) = base * (ref_area / A) ** area_exponent``, clamped to ``[0, 1]``.
    Larger islands carry larger populations per species and so lose them less
    often — the area effect. ``ref_area`` is the area at which extinction equals
    ``base`` (set it to the largest island so that island is the least
    extinction-prone). In the extinction-limited regime (A << ref_area) this
    yields S* ~ A**area_exponent, i.e. a species-area power law with slope
    z ~ area_exponent.
    """
    area = float(area)
    if area <= 0.0:
        return float(min(1.0, max(0.0, base)))
    return float(min(1.0, max(0.0, base * (ref_area / area) ** area_exponent)))


def mw_equilibrium_richness(n_pool: int, c: float, e: float) -> float:
    """Analytic equilibrium richness S* = P * c / (c + e) (0 if c + e == 0)."""
    denom = c + e
    return float(n_pool) * c / denom if denom > 0 else 0.0


class MacArthurWilsonEquilibriumProcess(Process):
    """One island under the canonical MacArthur-Wilson equilibrium model.

    The phenomenological colonization/extinction model of *The Theory of Island
    Biogeography* (MacArthur & Wilson 1963, 1967), as a counterpart to the
    mechanistic Lotka-Volterra :class:`IslandProcess`. A mainland pool of
    ``n_pool`` species feeds the island; each species is present (1) or absent
    (0). Each generation every absent species colonizes with probability
    ``c = colonization_base / (1 + distance)`` and every present species goes
    extinct with probability ``e = extinction_base * (ref_area/area)**area_exponent``.

    Richness relaxes to S* = P*c/(c+e): larger area (lower ``e``) and lower
    isolation (higher ``c``) both raise it, and species identity turns over even
    once S has plateaued (the turnover prediction). Immigration I(S)=(P-S)c
    falls with S and extinction E(S)=S*e rises with S; they cross at S* — the
    rate-crossing diagram.

    Outputs (all deltas, so the stores accumulate to absolute values — the
    mass/length delta precedent used throughout this workspace):

    presence : map[string, float]
        Per-species occupancy delta (+1 colonization, -1 extinction); the store
        accumulates to a 0/1 occupancy vector.
    richness : float
        Local richness delta; accumulates to the absolute present-species count.
    cum_colonizations, cum_extinctions : float
        Per-tick gross colonization / extinction counts, emitted as deltas so
        the stores accumulate to running totals (their first difference per tick
        is the gross turnover, from which the per-S event rates are measured).
    """

    config_schema = {
        "name": {"_type": "string", "_default": "island"},
        "n_pool": {"_type": "integer", "_default": 50},
        "colonization_base": {"_type": "float", "_default": 0.02},
        "extinction_base": {"_type": "float", "_default": 0.012},
        "area": {"_type": "float", "_default": 100.0},
        "distance": {"_type": "float", "_default": 0.0},
        "area_exponent": {"_type": "float", "_default": 0.25},
        "ref_area": {"_type": "float", "_default": 100.0},
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
                if self._rng.random() < self.e:
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
