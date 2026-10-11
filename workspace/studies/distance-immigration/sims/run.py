#!/usr/bin/env python3
"""Canonical run for the ``distance-immigration`` study.

Isolates the ``MigrationProcess`` coupler. With islands held at full occupancy,
tallies per-island immigration received over a seed ensemble and measures its
correlation with mean distance to the other islands (the colonization gradient),
and sums the net migration flux to confirm individuals are conserved exactly.
Renders the migration-kernel network over the spatial map and an
immigration-vs-distance scatter. Records the run to ``.pbg/runs.jsonl``.

Run from the workspace root::

    python workspace/studies/distance-immigration/sims/run.py
"""
from __future__ import annotations

import math
import random
import time
import uuid
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]

from viva_island_biogeography.core import build_core
from viva_island_biogeography.composites._common import island_specs
from viva_island_biogeography.processes import MigrationProcess
from viva_island_biogeography import viz
from vivarium_workbench.lib.run_log import append_run_event
from vivarium_workbench.lib.run_viz import mirror_viz_dir

SPEC_ID = "viva_island_biogeography.composites.archipelago.island_archipelago"
STUDY_SLUG = "distance-immigration"
INVESTIGATION_SLUG = "island-biogeography"

N_ISLANDS = 8
N_SPECIES = 30
SEEDS = range(6)
TICKS = 400

CORR_MAX = -0.3
NET_TOL = 1e-9


def main() -> int:
    run_id = uuid.uuid4().hex
    params = dict(n_islands=N_ISLANDS, n_species=N_SPECIES, ticks=TICKS,
                  seeds=list(SEEDS))
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID,
        "label": STUDY_SLUG, "started_at": time.time(), "status": "running",
        "n_steps": TICKS, "emitter": "ram", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": params,
    })
    try:
        specs = island_specs(N_ISLANDS, N_SPECIES, random.Random(0))
        meta = {s["name"]: {"x": s["x"], "y": s["y"], "area": s["area"]} for s in specs}
        names = [s["name"] for s in specs]
        positions = [[meta[n]["x"], meta[n]["y"]] for n in names]

        mean_dist = np.array([
            np.mean([math.hypot(meta[a]["x"] - meta[b]["x"], meta[a]["y"] - meta[b]["y"])
                     for b in names if b != a])
            for a in names])

        full = {n: {f"Species_{i}": 100.0 for i in range(N_SPECIES)} for n in names}
        received = np.zeros(N_ISLANDS)
        total_net = 0.0
        for seed in SEEDS:
            mp = MigrationProcess(config={
                "island_names": names, "positions": positions, "seed": seed},
                core=build_core())
            for _ in range(TICKS):
                out = mp.update({"abundances": full}, 1.0)
                for ni, ex in out["exchange"].items():
                    i = names.index(ni)
                    for _s, v in ex.items():
                        total_net += v
                        if v > 0:
                            received[i] += v
        received /= len(list(SEEDS))

        immigration_distance_corr = float(np.corrcoef(received, mean_dist)[0, 1])

        # --- visualizations ---------------------------------------------------
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz.write_html(viz.migration_network(meta), viz_dir / "migration_network.html")
        viz.write_html(
            viz.metric_vs_predictor(
                mean_dist, received, names,
                xtitle="mean distance to other islands",
                ytitle="immigration received per ensemble",
                title="Isolation reduces colonization", logx=False, trend=True),
            viz_dir / "immigration_vs_distance.html")
        viz_paths = [
            viz_dir / "migration_network.html",
            viz_dir / "immigration_vs_distance.html",
        ]

        t1 = immigration_distance_corr < CORR_MAX
        t2 = abs(total_net) < NET_TOL
        outcomes = {
            "immigration_distance_corr": round(immigration_distance_corr, 4),
            "total_net_migration": total_net,
            "isolation-reduces-immigration": "PASS" if t1 else "FAIL",
            "migration-conserves-individuals": "PASS" if t2 else "FAIL",
        }
        print("=== distance-immigration verdicts ===")
        for n, d, r in zip(names, mean_dist, received):
            print(f"  {n:10s} mean_dist={d:7.1f}  immigration={r:8.1f}")
        print(f"  immigration_distance_corr = {immigration_distance_corr:.4f}  (< {CORR_MAX}? {t1})")
        print(f"  total_net_migration = {total_net}  (~0? {t2})")
    except Exception:
        append_run_event(WORKSPACE_ROOT, {
            "run_id": run_id, "event": "completed", "completed_at": time.time(),
            "n_steps": 0, "status": "failed"})
        raise

    # Carry the figures into the Runs-tab "📊 Viz" action (.pbg/runs/<id>/viz.json).
    mirror_viz_dir(WORKSPACE_ROOT, run_id, viz_paths)
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "completed", "completed_at": time.time(),
        "n_steps": TICKS, "status": "completed", "outcomes": outcomes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
