#!/usr/bin/env python3
"""Canonical run for the ``island-archipelago`` study.

Runs the Lotka-Volterra ``island_archipelago`` composite to equilibrium and
measures, across islands, (a) corr(equilibrium total abundance, area) and
(b) corr(equilibrium richness, log area) — showing area sets abundance (exactly)
but not richness (random competitive exclusion). Renders an animated spatial
map, a richness-per-island timeseries, and the mainland's species abundance
stacked-area dynamics. Records the run to ``.pbg/runs.jsonl``.

Run from the workspace root::

    python workspace/studies/island-archipelago/sims/run.py
"""
from __future__ import annotations

import math
import random
import time
import uuid
from pathlib import Path

import numpy as np
from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]  # repo root (holds .pbg/)

from viva_island_biogeography.core import build_core
from viva_island_biogeography.composites.archipelago import island_archipelago
from viva_island_biogeography.composites._common import island_specs
from viva_island_biogeography import viz
from vivarium_workbench.lib.run_log import append_run_event
from vivarium_workbench.lib.run_viz import mirror_viz_dir

SPEC_ID = "viva_island_biogeography.composites.archipelago.island_archipelago"
STUDY_SLUG = "island-archipelago"
INVESTIGATION_SLUG = "island-biogeography"

N_ISLANDS = 6
N_SPECIES = 30
STEPS = 150
SEED = 0
TAIL = 25  # equilibrium window (last N generations)

CAPACITY_CORR_MIN = 0.95
RICHNESS_CORR_MAX = 0.35


def main() -> int:
    run_id = uuid.uuid4().hex
    params = dict(n_islands=N_ISLANDS, n_species=N_SPECIES, steps=STEPS, seed=SEED)
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID,
        "label": STUDY_SLUG, "started_at": time.time(), "status": "running",
        "n_steps": STEPS, "emitter": "ram", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": params,
    })
    try:
        core = build_core()
        doc = island_archipelago(
            core, n_islands=N_ISLANDS, n_species=N_SPECIES, growth_rate=0.03,
            interaction_weight=0.1, introduction_probability=0.05, seed=SEED)
        sim = Composite({"state": doc}, core=core)
        sim.run(float(STEPS))
        rows = gather_emitter_results(sim)[("emitter",)]

        specs = island_specs(N_ISLANDS, N_SPECIES, random.Random(SEED))
        meta = {s["name"]: {"x": s["x"], "y": s["y"], "area": s["area"]} for s in specs}
        names = [s["name"] for s in specs]
        area = np.array([meta[n]["area"] for n in names])

        tail = rows[-TAIL:]
        eq_total = np.array([
            np.mean([sum(r["islands"][n].values()) for r in tail]) for n in names])
        eq_rich = np.array([
            np.mean([sum(1 for c in r["islands"][n].values() if c > 0) for r in tail])
            for n in names])
        capacity = area * 100.0

        capacity_area_corr = float(np.corrcoef(eq_total, area)[0, 1])
        richness_area_corr = float(np.corrcoef(eq_rich, np.log(area))[0, 1])
        max_capacity_ratio = float((eq_total / capacity).max())

        # --- visualizations ---------------------------------------------------
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz.write_html(
            viz.spatial_map_animation(rows, meta, color_by="richness",
                                      title="Archipelago species richness over time"),
            viz_dir / "spatial_map.html")
        viz.write_html(
            viz.richness_timeseries(rows, names,
                                    title="Species richness per island"),
            viz_dir / "richness_timeseries.html")
        viz.write_html(
            viz.abundance_stacked_area(rows, "mainland", top_k=7),
            viz_dir / "mainland_abundance.html")
        viz_paths = [
            viz_dir / "spatial_map.html",
            viz_dir / "richness_timeseries.html",
            viz_dir / "mainland_abundance.html",
        ]

        t1 = CAPACITY_CORR_MIN < capacity_area_corr
        t2 = richness_area_corr < RICHNESS_CORR_MAX
        outcomes = {
            "capacity_area_corr": round(capacity_area_corr, 4),
            "richness_area_corr": round(richness_area_corr, 4),
            "max_capacity_ratio": round(max_capacity_ratio, 4),
            "abundance-scales-with-area": "PASS" if t1 else "FAIL",
            "richness-not-set-by-area": "PASS" if t2 else "FAIL",
        }
        print("=== island-archipelago verdicts ===")
        for n, a, r in zip(names, area, eq_rich):
            print(f"  {n:10s} area={a:7.0f}  eq_total={eq_total[names.index(n)]:9.0f}  eq_rich={r:.2f}")
        print(f"  capacity_area_corr = {capacity_area_corr:.4f}  (> {CAPACITY_CORR_MIN}? {t1})")
        print(f"  richness_area_corr = {richness_area_corr:.4f}  (< {RICHNESS_CORR_MAX}? {t2})")
    except Exception:
        append_run_event(WORKSPACE_ROOT, {
            "run_id": run_id, "event": "completed", "completed_at": time.time(),
            "n_steps": 0, "status": "failed"})
        raise

    # Carry the figures into the Runs-tab "📊 Viz" action (.pbg/runs/<id>/viz.json).
    mirror_viz_dir(WORKSPACE_ROOT, run_id, viz_paths)
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "completed", "completed_at": time.time(),
        "n_steps": STEPS, "status": "completed", "outcomes": outcomes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
