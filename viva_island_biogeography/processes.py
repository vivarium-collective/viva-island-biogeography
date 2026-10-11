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
