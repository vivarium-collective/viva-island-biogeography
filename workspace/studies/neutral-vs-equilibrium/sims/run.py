#!/usr/bin/env python3
"""Canonical run for the ``neutral-vs-equilibrium`` study.

Do dispersal-drift (Hubbell 2001 neutral) and colonization-extinction
(MacArthur & Wilson 1967 equilibrium) predict the same macroecology? This run
drives BOTH composites over the SAME log-spaced area gradient, measures the
emergent species-area slope ``z`` for each (ensemble-averaged over seeds), and
inspects the neutral local species-abundance distribution. The neutral model has
no colonization/extinction rate asymmetry at all, so any positive ``z`` it
produces comes purely from drift + dispersal in a fixed-J local community.

Run from the workspace root::

    python workspace/studies/neutral-vs-equilibrium/sims/run.py
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
from viva_island_biogeography.models_neutral import (
    register_neutral,
    neutral_vs_equilibrium_sar,
    neutral_rank_abundance,
    neutral_preston_octaves,
    write_html,
)
from viva_island_biogeography.composites.neutral import build_neutral_document
from viva_island_biogeography.composites.mw_equilibrium import build_mw_document
from vivarium_workbench.lib.run_log import append_run_event
from vivarium_workbench.lib.run_viz import mirror_viz_dir

SPEC_ID = "viva_island_biogeography.composites.neutral.neutral_metacommunity"
STUDY_SLUG = "neutral-vs-equilibrium"
INVESTIGATION_SLUG = "island-biogeography"

# Shared area gradient (both models span the SAME islands).
N_ISLANDS = 12
AREA_MIN, AREA_MAX = 1.0, 100.0
STEPS = 300
SEED = 1
N_SEEDS = 8          # ensemble: average equilibrium richness across seeds
TAIL_FRAC = 0.25

# Neutral (Hubbell 2001) parameters.
N_POOL = 64
DENSITY = 2.0
IMMIGRATION_BASE = 0.2

# MacArthur-Wilson equilibrium parameters (the already-built counterpart).
MW_N_POOL = 60
COLONIZATION_BASE = 0.02
EXTINCTION_BASE = 0.012
AREA_EXPONENT = 0.25

SAR_Z_MIN = 0.05     # a clear positive species-area slope


def _run_neutral(core, seed):
    doc = build_neutral_document(
        n_islands=N_ISLANDS, n_pool=N_POOL, density=DENSITY,
        immigration_base=IMMIGRATION_BASE, area_min=AREA_MIN, area_max=AREA_MAX,
        distance_min=0.0, distance_max=0.0, seed=seed)
    sim = Composite({"state": doc}, core=core)
    sim.run(float(STEPS))
    return gather_emitter_results(sim)[("emitter",)]


def _run_mw(core, seed):
    doc = build_mw_document(
        n_islands=N_ISLANDS, n_pool=MW_N_POOL,
        colonization_base=COLONIZATION_BASE, extinction_base=EXTINCTION_BASE,
        area_min=AREA_MIN, area_max=AREA_MAX, distance_min=0.0, distance_max=0.0,
        area_exponent=AREA_EXPONENT, seed=seed)
    sim = Composite({"state": doc}, core=core)
    sim.run(float(STEPS))
    return gather_emitter_results(sim)[("emitter",)]


def _tail_mean_S(rows, names):
    tail = rows[max(0, int(len(rows) * (1 - TAIL_FRAC))):]
    return np.array([float(np.mean([r["richness"][n] for r in tail])) for n in names])


def _ensemble_S(run_fn, core, names):
    ens = [_tail_mean_S(run_fn(core, SEED), names)]
    for s in range(SEED + 1, SEED + N_SEEDS):
        ens.append(_tail_mean_S(run_fn(core, s), names))
    return np.mean(np.vstack(ens), axis=0)


def _sar_z(S_obs, A):
    S_obs = np.asarray(S_obs, dtype=float)
    A = np.asarray(A, dtype=float)
    mask = S_obs > 0
    if mask.sum() < 2 or np.ptp(np.log(A[mask])) <= 0:
        return 0.0
    return float(np.polyfit(np.log(A[mask]), np.log(S_obs[mask]), 1)[0])


def main() -> int:
    run_id = uuid.uuid4().hex
    params = dict(n_islands=N_ISLANDS, n_pool=N_POOL, density=DENSITY,
                  immigration_base=IMMIGRATION_BASE, area_min=AREA_MIN,
                  area_max=AREA_MAX, steps=STEPS, seed=SEED, n_seeds=N_SEEDS)
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID,
        "label": STUDY_SLUG, "started_at": time.time(), "status": "running",
        "n_steps": STEPS, "emitter": "ram", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": params,
    })
    try:
        core = build_core()
        register_neutral(core)

        # --- neutral (dispersal-drift) ---------------------------------------
        neutral_rows = _run_neutral(core, SEED)   # primary seed backs the figures
        meta = neutral_rows[-1]["island_meta"]
        names = sorted(meta)
        A = np.array([meta[n]["area"] for n in names], dtype=float)
        S_neutral = _ensemble_S(_run_neutral, core, names)
        z_neutral = _sar_z(S_neutral, A)

        # --- equilibrium (MacArthur-Wilson) over the SAME area gradient -------
        mw_rows = _run_mw(core, SEED)
        mw_names = sorted(mw_rows[-1]["island_meta"])
        S_equilibrium = _ensemble_S(_run_mw, core, mw_names)
        z_equilibrium = _sar_z(S_equilibrium, A)

        # Representative neutral island for the species-abundance distribution:
        # the largest (richest, longest-tailed) island of the primary seed.
        rep = names[-1]
        rep_abundance = neutral_rows[-1]["abundance"][rep]

        # --- visualizations ---------------------------------------------------
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        write_html(
            neutral_vs_equilibrium_sar(A, S_neutral, S_equilibrium,
                                       z_neutral, z_equilibrium),
            viz_dir / "sar_overlay.html")
        write_html(neutral_rank_abundance(rep_abundance, rep),
                   viz_dir / "rank_abundance.html")
        write_html(neutral_preston_octaves(rep_abundance, rep),
                   viz_dir / "preston_octaves.html")
        viz_paths = [viz_dir / "sar_overlay.html",
                     viz_dir / "rank_abundance.html",
                     viz_dir / "preston_octaves.html"]

        t1 = z_neutral > SAR_Z_MIN
        t2 = (z_neutral > SAR_Z_MIN) and (z_equilibrium > SAR_Z_MIN)
        outcomes = {
            "z_neutral": round(z_neutral, 4),
            "z_equilibrium": round(z_equilibrium, 4),
            "both_sar": 1.0 if t2 else 0.0,
            "representative_island": rep,
            "rep_richness": int(sum(1 for c in rep_abundance.values() if c > 0)),
            "neutral-produces-sar": "PASS" if t1 else "FAIL",
            "both-models-show-sar": "PASS" if t2 else "FAIL",
        }
        print("=== neutral-vs-equilibrium verdicts ===")
        for i, n in enumerate(names):
            print(f"  {n}  area={meta[n]['area']:7.1f}  J={meta[n]['J']:5.0f}  "
                  f"S_neutral={S_neutral[i]:5.2f}  S_equil={S_equilibrium[i]:5.2f}")
        print(f"  z_neutral     = {z_neutral:.4f}  (> {SAR_Z_MIN}? {t1})")
        print(f"  z_equilibrium = {z_equilibrium:.4f}")
        print(f"  neutral-produces-sar  : {'PASS' if t1 else 'FAIL'}")
        print(f"  both-models-show-sar  : {'PASS' if t2 else 'FAIL'}")
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
