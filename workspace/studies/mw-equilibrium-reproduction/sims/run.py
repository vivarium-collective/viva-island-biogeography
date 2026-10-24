#!/usr/bin/env python3
"""Canonical run for the ``mw-equilibrium-reproduction`` study.

Reproduces the ORIGINAL theory of island biogeography (MacArthur & Wilson 1963,
1967) with the phenomenological ``mw_equilibrium`` composite: independent
islands over a log-spaced area gradient, each relaxing to the equilibrium S*
where immigration I(S)=(P−S)c meets extinction E(S)=S·e. Shows (a) the
rate-crossing diagram sits on the simulated events, (b) richness relaxes to the
analytic S*, and (c) species identity turns over while S stays flat.

Run from the workspace root::

    python workspace/studies/mw-equilibrium-reproduction/sims/run.py
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
from viva_island_biogeography.processes import (
    colonization_prob, extinction_prob, mw_equilibrium_richness)
from viva_island_biogeography import viz
from vivarium_workbench.lib.run_log import append_run_event
from vivarium_workbench.lib.run_viz import mirror_viz_dir

SPEC_ID = "viva_island_biogeography.composites.mw_equilibrium.mw_equilibrium"
STUDY_SLUG = "mw-equilibrium-reproduction"
INVESTIGATION_SLUG = "island-biogeography"

N_ISLANDS = 12
N_POOL = 60
COLONIZATION_BASE = 0.02
EXTINCTION_BASE = 0.012
AREA_MIN, AREA_MAX = 1.0, 100.0
AREA_EXPONENT = 0.25
STEPS = 400
SEED = 1
N_SEEDS = 8          # ensemble: average equilibrium S across seeds (noise down)
TAIL_FRAC = 0.25

EQ_TRACK_CORR_MIN = 0.9   # ensemble-mean observed S must track analytic S* across islands
TURNOVER_RATIO_MIN = 3.0  # gross equilibrium events >> net richness change


def _run_once(core, seed):
    doc = build_mw_document(
        n_islands=N_ISLANDS, n_pool=N_POOL,
        colonization_base=COLONIZATION_BASE, extinction_base=EXTINCTION_BASE,
        area_min=AREA_MIN, area_max=AREA_MAX,
        distance_min=0.0, distance_max=0.0,
        area_exponent=AREA_EXPONENT, seed=seed)
    sim = Composite({"state": doc}, core=core)
    sim.run(float(STEPS))
    return gather_emitter_results(sim)[("emitter",)]


def main() -> int:
    run_id = uuid.uuid4().hex
    params = dict(n_islands=N_ISLANDS, n_pool=N_POOL, area_min=AREA_MIN,
                  area_max=AREA_MAX, area_exponent=AREA_EXPONENT, steps=STEPS, seed=SEED)
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID,
        "label": STUDY_SLUG, "started_at": time.time(), "status": "running",
        "n_steps": STEPS, "emitter": "ram", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": params,
    })
    try:
        core = build_core()
        # Primary run (seed=SEED) backs the figures; the ensemble backs the metrics.
        rows = _run_once(core, SEED)
        meta = rows[-1]["island_meta"]
        names = sorted(meta)
        ref_area = max(AREA_MAX, AREA_MIN, 1.0)

        def tail_mean_S(rws):
            t = rws[max(0, int(len(rws) * (1 - TAIL_FRAC))):]
            return np.array([float(np.mean([r["richness"][n] for r in t])) for n in names])

        ens = [tail_mean_S(rows)]
        for s in range(SEED + 1, SEED + N_SEEDS):
            ens.append(tail_mean_S(_run_once(core, s)))
        S_obs = np.mean(np.vstack(ens), axis=0)   # ensemble-mean equilibrium richness
        tail = rows[max(0, int(len(rows) * (1 - TAIL_FRAC))):]
        S_star = np.array([
            mw_equilibrium_richness(
                N_POOL,
                colonization_prob(meta[n]["distance"], COLONIZATION_BASE),
                extinction_prob(meta[n]["area"], EXTINCTION_BASE, AREA_EXPONENT, ref_area))
            for n in names])
        eq_track_corr = float(np.corrcoef(S_obs, S_star)[0, 1])

        # turnover on a representative mid-gradient island
        rep = names[N_ISLANDS // 2]
        gross = (rows[-1]["colonizations"][rep] - tail[0]["colonizations"][rep]
                 + rows[-1]["extinctions"][rep] - tail[0]["extinctions"][rep])
        net = abs(rows[-1]["richness"][rep] - tail[0]["richness"][rep])
        turnover_ratio = float(gross / max(1.0, net))

        A = np.array([meta[n]["area"] for n in names])
        mask = S_obs > 0
        sar_z = float(np.polyfit(np.log(A[mask]), np.log(S_obs[mask]), 1)[0])

        # --- visualizations ---------------------------------------------------
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        mw_kw = dict(n_pool=N_POOL, colonization_base=COLONIZATION_BASE,
                     extinction_base=EXTINCTION_BASE, area_exponent=AREA_EXPONENT,
                     ref_area=ref_area)
        viz.write_html(viz.mw_rate_crossing(rows, meta, rep, **mw_kw),
                       viz_dir / "rate_crossing.html")
        viz.write_html(viz.mw_richness_to_equilibrium(rows, meta, **mw_kw),
                       viz_dir / "approach_to_equilibrium.html")
        viz.write_html(viz.turnover_timeseries(rows, rep),
                       viz_dir / "turnover.html")
        viz_paths = [viz_dir / "rate_crossing.html",
                     viz_dir / "approach_to_equilibrium.html",
                     viz_dir / "turnover.html"]

        t1 = eq_track_corr > EQ_TRACK_CORR_MIN
        t2 = turnover_ratio > TURNOVER_RATIO_MIN
        outcomes = {
            "eq_track_corr": round(eq_track_corr, 4),
            "sar_z": round(sar_z, 4),
            "turnover_ratio": round(turnover_ratio, 2),
            "representative_island": rep,
            "equilibrium-tracks-theory": "PASS" if t1 else "FAIL",
            "turnover-at-equilibrium": "PASS" if t2 else "FAIL",
        }
        print("=== mw-equilibrium-reproduction verdicts ===")
        for n in names:
            i = names.index(n)
            print(f"  {n}  area={meta[n]['area']:7.1f}  S_obs={S_obs[i]:5.2f}  S*={S_star[i]:5.2f}")
        print(f"  eq_track_corr = {eq_track_corr:.4f}  (> {EQ_TRACK_CORR_MIN}? {t1})")
        print(f"  turnover_ratio({rep}) = {turnover_ratio:.2f}  (> {TURNOVER_RATIO_MIN}? {t2})")
        print(f"  SAR z = {sar_z:.4f}")
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
