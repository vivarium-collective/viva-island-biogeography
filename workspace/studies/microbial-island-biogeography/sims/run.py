#!/usr/bin/env python3
"""Canonical run for the ``microbial-island-biogeography`` study.

Does the taxa-area relationship hold for **microbes**, and is its slope ``z``
flatter than for macro-organisms?

Microbial island biogeography (Martiny et al. 2006; Horner-Devine et al. 2004;
Bell et al. 2005) reports a positive but **shallow** taxa-area slope
(z ~ 0.02-0.1) against the classic macro z ~ 0.15-0.35, driven by a huge taxa
pool, strong dispersal and fast local turnover. This run reuses the MacArthur-
Wilson equilibrium process (``build_mw_document`` — no new process) to run two
regimes over the **same** area gradient:

* **MACRO** — small pool (n_pool~60), classic area-extinction exponent (~0.25),
  slow turnover; the regime the ``species-area-and-isolation`` study calibrated.
* **MICROBIAL** — large pool (n_pool~300), shallow area-extinction exponent
  (~0.10), fast colonization + extinction (high turnover).

It measures the log-log taxa-area slope ``z`` for each, plus the gross
equilibrium turnover rate, and renders the macro-vs-microbe overlay. Run from
the workspace root::

    python workspace/studies/microbial-island-biogeography/sims/run.py
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
from viva_island_biogeography.composites import microbial as mviz
from viva_island_biogeography import viz
from vivarium_workbench.lib.run_log import append_run_event
from vivarium_workbench.lib.run_viz import mirror_viz_dir

SPEC_ID = "viva_island_biogeography.composites.microbial.microbial_island"
STUDY_SLUG = "microbial-island-biogeography"
INVESTIGATION_SLUG = "island-biogeography"

# Same area gradient + isolation (fixed at 0) for both regimes, so the only
# difference between the measured slopes is the regime's parameters.
N_ISLANDS = 12
AREA_MIN = 1.0
AREA_MAX = 100.0
STEPS = 400
SEED = 1
N_SEEDS = 6
TAIL_FRAC = 0.25

# --- MACRO regime (classic macro-ecological island biogeography) -----------
MACRO = dict(n_pool=60, colonization_base=0.02, extinction_base=0.012,
             area_exponent=0.25)
# --- MICROBIAL regime (large pool, fast turnover, shallow area exponent) ----
MICRO = dict(n_pool=300, colonization_base=0.10, extinction_base=0.06,
             area_exponent=0.10)

# Outcome gates
SAR_Z_MIN = 0.0  # microbial taxa-area relationship holds (positive slope)


def _run(core, *, n_pool, colonization_base, extinction_base, area_exponent, seed):
    doc = build_mw_document(
        n_islands=N_ISLANDS, n_pool=n_pool,
        colonization_base=colonization_base, extinction_base=extinction_base,
        area_min=AREA_MIN, area_max=AREA_MAX,
        distance_min=0.0, distance_max=0.0,
        area_exponent=area_exponent, seed=seed)
    sim = Composite({"state": doc}, core=core)
    sim.run(float(STEPS))
    return gather_emitter_results(sim)[("emitter",)]


def _tail_mean_S(rows, names):
    t = rows[max(0, int(len(rows) * (1 - TAIL_FRAC))):]
    return np.array([float(np.mean([r["richness"][n] for r in t])) for n in names])


def _tail_turnover(rows, names):
    """Gross per-generation turnover (colonizations + extinctions per island
    per generation) over the equilibrium tail, from the cumulative stores."""
    t0 = max(0, int(len(rows) * (1 - TAIL_FRAC)))
    span = max(1, (len(rows) - 1) - t0)
    col_rate, ext_rate = [], []
    for n in names:
        col = np.array([r["colonizations"][n] for r in rows], dtype=float)
        ext = np.array([r["extinctions"][n] for r in rows], dtype=float)
        col_rate.append((col[-1] - col[t0]) / span)
        ext_rate.append((ext[-1] - ext[t0]) / span)
    return {"colonization": float(np.mean(col_rate)),
            "extinction": float(np.mean(ext_rate))}


def _ensemble(core, names, cfg):
    """Return (first-run rows, ensemble-mean tail richness, mean turnover)."""
    rows0 = _run(core, seed=SEED, **cfg)
    S_ens = [_tail_mean_S(rows0, names)]
    turn = [_tail_turnover(rows0, names)]
    for s in range(SEED + 1, SEED + N_SEEDS):
        rows = _run(core, seed=s, **cfg)
        S_ens.append(_tail_mean_S(rows, names))
        turn.append(_tail_turnover(rows, names))
    S = np.mean(np.vstack(S_ens), axis=0)
    turnover = {k: float(np.mean([t[k] for t in turn])) for k in ("colonization", "extinction")}
    return rows0, S, turnover


def _fit_z(A, S):
    mask = S > 0
    return float(np.polyfit(np.log(A[mask]), np.log(S[mask]), 1)[0])


def main() -> int:
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID,
        "label": STUDY_SLUG, "started_at": time.time(), "status": "running",
        "n_steps": STEPS, "emitter": "ram", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": dict(n_islands=N_ISLANDS, n_seeds=N_SEEDS, steps=STEPS,
                       macro=MACRO, micro=MICRO),
    })
    try:
        core = build_core()
        names = [f"island_{i:02d}" for i in range(N_ISLANDS)]

        macro_rows, S_macro, turn_macro = _ensemble(core, names, MACRO)
        micro_rows, S_micro, turn_micro = _ensemble(core, names, MICRO)

        A = np.array([macro_rows[-1]["island_meta"][n]["area"] for n in names])
        z_macro = _fit_z(A, S_macro)
        z_micro = _fit_z(A, S_micro)

        gross_macro = turn_macro["colonization"] + turn_macro["extinction"]
        gross_micro = turn_micro["colonization"] + turn_micro["extinction"]

        # --- figures ------------------------------------------------------
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        write_html = viz.write_html
        write_html(
            mviz.taxa_area_overlay(
                {"A": A, "S": S_macro, "label": "macro-organisms"},
                {"A": A, "S": S_micro, "label": "microbes"}),
            viz_dir / "taxa_area_overlay.html")
        write_html(
            mviz.turnover_comparison(turn_macro, turn_micro),
            viz_dir / "turnover_comparison.html")
        write_html(
            mviz.richness_distribution(S_macro, S_micro),
            viz_dir / "richness_distribution.html")
        viz_paths = [viz_dir / "taxa_area_overlay.html",
                     viz_dir / "turnover_comparison.html",
                     viz_dir / "richness_distribution.html"]

        # --- gates --------------------------------------------------------
        t_holds = z_micro > SAR_Z_MIN                 # microbial-sar-holds
        t_flatter = z_micro < z_macro                 # microbial-z-flatter
        outcomes = {
            "z_macro": round(z_macro, 4),
            "z_micro": round(z_micro, 4),
            "turnover_macro": round(gross_macro, 4),
            "turnover_micro": round(gross_micro, 4),
            "microbial-sar-holds": "PASS" if t_holds else "FAIL",
            "microbial-z-flatter": "PASS" if t_flatter else "FAIL",
        }
        print("=== microbial-island-biogeography verdicts ===")
        print(f"  z_macro = {z_macro:.4f}")
        print(f"  z_micro = {z_micro:.4f}")
        print(f"  microbial SAR holds (z_micro > 0)? {t_holds}")
        print(f"  microbial z flatter than macro (z_micro < z_macro)? {t_flatter}")
        print(f"  gross turnover  macro = {gross_macro:.4f}   micro = {gross_micro:.4f}"
              f"   (events · gen⁻¹ · island⁻¹)")
        print("  macro S:", np.round(S_macro, 1).tolist())
        print("  micro S:", np.round(S_micro, 1).tolist())
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
