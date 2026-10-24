#!/usr/bin/env python3
"""Canonical run for the ``rescue-effect`` study.

Brown & Kodric-Brown's (1977) *rescue effect* — immigration also lowers
extinction — implemented by coupling the two MacArthur-Wilson curves in the
``rescue_effect`` composite. Each present species is rescued with probability
``c`` and, if rescued, its extinction probability drops to ``e*(1-rescue_strength)``.

This run (a) sweeps ``rescue_strength`` across islands at matched area/isolation
and measures the ensemble-mean equilibrium richness S* and the equilibrium
turnover (gross colonization+extinction events per generation) at each level,
and (b) reports the matched no-rescue (r=0) vs with-rescue (r=0.8) comparison.
Sanity check: at r=0 the simulated S* must match the analytic MacArthur-Wilson
S* = P*c/(c+e).

Run from the workspace root::

    python workspace/studies/rescue-effect/sims/run.py
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
from viva_island_biogeography.composites.rescue import build_rescue_document
from viva_island_biogeography.models_rescue import (
    register_rescue, rescue_equilibrium_richness,
    rescue_richness_vs_strength, rescue_turnover_vs_strength, rescue_rate_crossing)
from viva_island_biogeography.processes import (
    colonization_prob, extinction_prob, mw_equilibrium_richness)
from viva_island_biogeography import viz
from vivarium_workbench.lib.run_log import append_run_event
from vivarium_workbench.lib.run_viz import mirror_viz_dir

SPEC_ID = "viva_island_biogeography.composites.rescue.rescue_effect"
STUDY_SLUG = "rescue-effect"
INVESTIGATION_SLUG = "island-biogeography"

N_ISLANDS = 9                 # rescue-strength levels 0.0 .. 0.8 in steps of 0.1
N_POOL = 60
COLONIZATION_BASE = 0.5       # c at distance 0 (also the per-tick rescue probability)
EXTINCTION_BASE = 0.5         # baseline e at the reference area
AREA = 100.0
DISTANCE = 0.0
AREA_EXPONENT = 0.25
RESCUE_MIN, RESCUE_MAX = 0.0, 0.8
STEPS = 400
SEED = 1
N_SEEDS = 8                   # ensemble: average equilibrium across seeds
TAIL_FRAC = 0.25

RESCUE_WITH = 0.8             # the headline "with rescue" level (last island)
RICHNESS_MARGIN_MIN = 2.0     # S_with must exceed S_no by at least this many species
TURNOVER_DROP_MIN = 2.0       # turnover must fall by at least this many events/gen
SANITY_TOL = 1.5              # |S_obs(r=0) - analytic M-W S*| must be within this


def _run_once(core, seed):
    doc = build_rescue_document(
        n_islands=N_ISLANDS, n_pool=N_POOL,
        colonization_base=COLONIZATION_BASE, extinction_base=EXTINCTION_BASE,
        area=AREA, distance=DISTANCE, area_exponent=AREA_EXPONENT,
        rescue_min=RESCUE_MIN, rescue_max=RESCUE_MAX, seed=seed)
    sim = Composite({"state": doc}, core=core)
    sim.run(float(STEPS))
    return gather_emitter_results(sim)[("emitter",)]


def main() -> int:
    run_id = uuid.uuid4().hex
    params = dict(n_islands=N_ISLANDS, n_pool=N_POOL,
                  colonization_base=COLONIZATION_BASE, extinction_base=EXTINCTION_BASE,
                  area=AREA, distance=DISTANCE, rescue_min=RESCUE_MIN,
                  rescue_max=RESCUE_MAX, steps=STEPS, seed=SEED)
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID,
        "label": STUDY_SLUG, "started_at": time.time(), "status": "running",
        "n_steps": STEPS, "emitter": "ram", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": params,
    })
    try:
        core = build_core()
        register_rescue(core)

        # Primary run (seed=SEED) backs the figures; the ensemble backs the metrics.
        rows = _run_once(core, SEED)
        meta = rows[-1]["island_meta"]
        names = sorted(meta)
        strengths = np.array([meta[n]["rescue_strength"] for n in names])

        def tail(rws):
            return rws[max(0, int(len(rws) * (1 - TAIL_FRAC))):]

        def tail_mean_S(rws):
            t = tail(rws)
            return np.array([float(np.mean([r["richness"][n] for r in t])) for n in names])

        def turnover_per_gen(rws):
            """Gross colonization+extinction events per generation over the tail
            window, per island."""
            t = tail(rws)
            n_gen = max(1, len(t) - 1)
            out = []
            for n in names:
                gross = ((t[-1]["colonizations"][n] - t[0]["colonizations"][n])
                         + (t[-1]["extinctions"][n] - t[0]["extinctions"][n]))
                out.append(gross / n_gen)
            return np.array(out)

        # Ensemble over seeds.
        S_ens = [tail_mean_S(rows)]
        T_ens = [turnover_per_gen(rows)]
        for s in range(SEED + 1, SEED + N_SEEDS):
            r = _run_once(core, s)
            S_ens.append(tail_mean_S(r))
            T_ens.append(turnover_per_gen(r))
        S_obs = np.mean(np.vstack(S_ens), axis=0)
        T_obs = np.mean(np.vstack(T_ens), axis=0)

        # Analytic rescue curve + baseline M-W rate at r=0.
        c = colonization_prob(DISTANCE, COLONIZATION_BASE)
        e = extinction_prob(AREA, EXTINCTION_BASE, AREA_EXPONENT, max(AREA, 1.0))
        S_star = np.array([rescue_equilibrium_richness(N_POOL, c, e, r) for r in strengths])
        mw_sstar = mw_equilibrium_richness(N_POOL, c, e)   # independent-curves baseline

        # No-rescue (island r=0) vs with-rescue (island r=RESCUE_WITH).
        i_no = int(np.argmin(np.abs(strengths - 0.0)))
        i_with = int(np.argmin(np.abs(strengths - RESCUE_WITH)))
        S_no, S_with = float(S_obs[i_no]), float(S_obs[i_with])
        T_no, T_with = float(T_obs[i_no]), float(T_obs[i_with])

        richness_margin = S_with - S_no
        turnover_drop = T_no - T_with
        turnover_ratio = T_with / T_no if T_no > 0 else float("nan")
        sanity_err = abs(S_no - mw_sstar)

        g_rich = richness_margin > RICHNESS_MARGIN_MIN
        g_turn = turnover_drop > TURNOVER_DROP_MIN
        g_sanity = sanity_err < SANITY_TOL

        # --- visualizations ---------------------------------------------------
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz.write_html(
            rescue_richness_vs_strength(strengths, S_obs, s_star=S_star),
            viz_dir / "richness_vs_rescue.html")
        viz.write_html(
            rescue_turnover_vs_strength(strengths, T_obs),
            viz_dir / "turnover_vs_rescue.html")
        viz.write_html(
            rescue_rate_crossing(n_pool=N_POOL, c=c, e=e, rescue_strength=RESCUE_WITH),
            viz_dir / "rate_crossing.html")
        viz_paths = [viz_dir / "richness_vs_rescue.html",
                     viz_dir / "turnover_vs_rescue.html",
                     viz_dir / "rate_crossing.html"]

        outcomes = {
            "S_no_rescue": round(S_no, 3),
            "S_with_rescue": round(S_with, 3),
            "richness_margin": round(richness_margin, 3),
            "turnover_no_rescue": round(T_no, 3),
            "turnover_with_rescue": round(T_with, 3),
            "turnover_drop": round(turnover_drop, 3),
            "turnover_ratio": round(turnover_ratio, 3),
            "mw_baseline_sstar": round(float(mw_sstar), 3),
            "sanity_err_r0": round(sanity_err, 3),
            "rescue-raises-richness": "PASS" if g_rich else "FAIL",
            "rescue-lowers-turnover": "PASS" if g_turn else "FAIL",
            "r0-matches-baseline": "PASS" if g_sanity else "FAIL",
        }
        print("=== rescue-effect verdicts ===")
        print(f"  c = {c:.3f}   e = {e:.3f}   P = {N_POOL}")
        for i, n in enumerate(names):
            print(f"  {n}  r={strengths[i]:.2f}  S_obs={S_obs[i]:5.2f}  "
                  f"S*={S_star[i]:5.2f}  turnover={T_obs[i]:5.2f} ev/gen")
        print(f"  no-rescue (r=0):   S={S_no:.2f}  turnover={T_no:.2f} ev/gen")
        print(f"  with-rescue (r={RESCUE_WITH}): S={S_with:.2f}  turnover={T_with:.2f} ev/gen")
        print(f"  richness margin = {richness_margin:+.2f}  (> {RICHNESS_MARGIN_MIN}? {g_rich})")
        print(f"  turnover drop   = {turnover_drop:+.2f} ev/gen  "
              f"(> {TURNOVER_DROP_MIN}? {g_turn})   ratio={turnover_ratio:.3f}")
        print(f"  SANITY r=0: S_obs={S_no:.2f}  analytic M-W S*={mw_sstar:.2f}  "
              f"|err|={sanity_err:.2f}  (< {SANITY_TOL}? {g_sanity})")
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
