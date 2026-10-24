#!/usr/bin/env python3
"""Canonical run for the ``species-area-and-isolation`` study.

The two central predictions of MacArthur-Wilson theory, as laws, from the
``mw_equilibrium`` composite:

* **Species-area law** — an area gradient (fixed isolation) yields S ~ A^z
  (Arrhenius 1921; Preston 1962), measured as the log-log slope z.
* **Isolation gradient** — a distance gradient (fixed area) yields richness
  declining with isolation, because colonization c = c0/(1+distance) falls.

Each law is measured on an ensemble mean over seeds. Run from the workspace root::

    python workspace/studies/species-area-and-isolation/sims/run.py
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]

from viva_island_biogeography.core import build_core
from viva_island_biogeography.composites.mw_equilibrium import build_mw_document
from viva_island_biogeography import viz
from vivarium_workbench.lib.run_log import append_run_event
from vivarium_workbench.lib.run_viz import mirror_viz_dir

SPEC_ID = "viva_island_biogeography.composites.mw_equilibrium.mw_equilibrium"
STUDY_SLUG = "species-area-and-isolation"
INVESTIGATION_SLUG = "island-biogeography"

N_ISLANDS = 12
N_POOL = 60
COLONIZATION_BASE = 0.02
EXTINCTION_BASE = 0.012
AREA_EXPONENT = 0.25
STEPS = 400
SEED = 1
N_SEEDS = 8
TAIL_FRAC = 0.25

FIXED_AREA_FOR_ISOLATION = 30.0
DISTANCE_MAX = 30.0

SAR_Z_MIN = 0.08           # a real, positive species-area slope
ISOLATION_CORR_MAX = -0.9  # richness declines monotonically with isolation (Spearman rank)


def _spearman(x, y):
    """Rank (Spearman) correlation — monotonicity, robust to the hyperbolic
    c=c0/(1+d) isolation decline that Pearson would understate."""
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    return float(np.corrcoef(rx, ry)[0, 1])


def _run(core, *, area_min, area_max, distance_min, distance_max, seed):
    doc = build_mw_document(
        n_islands=N_ISLANDS, n_pool=N_POOL,
        colonization_base=COLONIZATION_BASE, extinction_base=EXTINCTION_BASE,
        area_min=area_min, area_max=area_max,
        distance_min=distance_min, distance_max=distance_max,
        area_exponent=AREA_EXPONENT, seed=seed)
    sim = Composite({"state": doc}, core=core)
    sim.run(float(STEPS))
    return gather_emitter_results(sim)[("emitter",)]


def _tail_mean_S(rows, names):
    t = rows[max(0, int(len(rows) * (1 - TAIL_FRAC))):]
    return np.array([float(np.mean([r["richness"][n] for r in t])) for n in names])


def _ensemble_S(core, names, **cfg):
    runs0 = _run(core, seed=SEED, **cfg)
    ens = [_tail_mean_S(runs0, names)]
    for s in range(SEED + 1, SEED + N_SEEDS):
        ens.append(_tail_mean_S(_run(core, seed=s, **cfg), names))
    return runs0, np.mean(np.vstack(ens), axis=0)


def main() -> int:
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID,
        "label": STUDY_SLUG, "started_at": time.time(), "status": "running",
        "n_steps": STEPS, "emitter": "ram", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": dict(n_islands=N_ISLANDS, n_pool=N_POOL, n_seeds=N_SEEDS, steps=STEPS),
    })
    try:
        core = build_core()
        names = [f"island_{i:02d}" for i in range(N_ISLANDS)]

        # --- species-area: area gradient, isolation fixed at 0 ----------------
        area_cfg = dict(area_min=1.0, area_max=100.0, distance_min=0.0, distance_max=0.0)
        area_rows, S_area = _ensemble_S(core, names, **area_cfg)
        A = np.array([area_rows[-1]["island_meta"][n]["area"] for n in names])
        mask = S_area > 0
        sar_z = float(np.polyfit(np.log(A[mask]), np.log(S_area[mask]), 1)[0])

        # --- isolation: distance gradient, area fixed -------------------------
        dist_cfg = dict(area_min=FIXED_AREA_FOR_ISOLATION, area_max=FIXED_AREA_FOR_ISOLATION,
                        distance_min=0.0, distance_max=DISTANCE_MAX)
        dist_rows, S_dist = _ensemble_S(core, names, **dist_cfg)
        D = np.array([dist_rows[-1]["island_meta"][n]["distance"] for n in names])
        isolation_corr = _spearman(D, S_dist)            # monotonic decline
        isolation_corr_pearson = float(np.corrcoef(D, S_dist)[0, 1])

        # --- visualizations: swap ensemble-mean S into the last row for charts
        for r in [area_rows[-1]]:
            for i, n in enumerate(names):
                r["richness"][n] = float(S_area[i])
        for r in [dist_rows[-1]]:
            for i, n in enumerate(names):
                r["richness"][n] = float(S_dist[i])

        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz.write_html(
            viz.species_area_curve(area_rows[-1:], area_rows[-1]["island_meta"], tail_frac=1.0),
            viz_dir / "species_area.html")
        viz.write_html(
            viz.species_distance_curve(dist_rows[-1:], dist_rows[-1]["island_meta"], tail_frac=1.0),
            viz_dir / "species_distance.html")
        viz_paths = [viz_dir / "species_area.html", viz_dir / "species_distance.html"]

        t1 = sar_z > SAR_Z_MIN
        t2 = isolation_corr < ISOLATION_CORR_MAX
        outcomes = {
            "sar_z": round(sar_z, 4),
            "isolation_corr_spearman": round(isolation_corr, 4),
            "isolation_corr_pearson": round(isolation_corr_pearson, 4),
            "species-area-law": "PASS" if t1 else "FAIL",
            "isolation-declines": "PASS" if t2 else "FAIL",
        }
        print("=== species-area-and-isolation verdicts ===")
        print(f"  SAR z = {sar_z:.4f}  (> {SAR_Z_MIN}? {t1})")
        print(f"  isolation corr(distance, S) = {isolation_corr:.4f}  (< {ISOLATION_CORR_MAX}? {t2})")
        print("  area gradient S:", np.round(S_area, 1).tolist())
        print("  isolation S:    ", np.round(S_dist, 1).tolist())
    except Exception:
        append_run_event(WORKSPACE_ROOT, {
            "run_id": run_id, "event": "completed", "completed_at": time.time(),
            "n_steps": 0, "status": "failed"})
        raise

    mirror_viz_dir(WORKSPACE_ROOT, run_id, viz_paths)
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "completed", "completed_at": time.time(),
        "n_steps": STEPS, "status": "completed", "outcomes": outcomes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
