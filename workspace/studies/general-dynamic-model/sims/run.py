#!/usr/bin/env python3
"""Canonical run for the ``general-dynamic-model`` study.

Implements the General Dynamic Model of oceanic island biogeography (Whittaker,
Triantis & Ladle 2008) with the ``general_dynamic_model`` composite: a few
islands whose area rises then erodes over geological time (island ontogeny).
The question: does species richness OVERSHOOT and then DECLINE — a humped
trajectory — unlike the flat static-area MacArthur-Wilson equilibrium?

Renders (a) area(t) and richness(t) overlaid on a dual axis showing the hump,
(b) the richness-vs-area hysteresis loop (richness lags area), and (c) the
ensemble richness trajectory with a ±1σ band.

Run from the workspace root::

    python workspace/studies/general-dynamic-model/sims/run.py
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
from viva_island_biogeography.models_gdm import (
    register_gdm, gdm_area_richness_overlay, gdm_hysteresis_loop,
    gdm_ensemble_richness)
from viva_island_biogeography.composites.gdm import build_gdm_document
from viva_island_biogeography import viz
from vivarium_workbench.lib.run_log import append_run_event
from vivarium_workbench.lib.run_viz import mirror_viz_dir

SPEC_ID = "viva_island_biogeography.composites.gdm.general_dynamic_model"
STUDY_SLUG = "general-dynamic-model"
INVESTIGATION_SLUG = "island-biogeography"

N_ISLANDS = 3
N_POOL = 60
COLONIZATION_BASE = 0.05
EXTINCTION_BASE = 0.01
AREA_MAX = 100.0
T_PEAK_MIN, T_PEAK_MAX = 120.0, 180.0
ONTOGENY_SHAPE = 2.0
AREA_EXPONENT = 1.0
REF_AREA = 100.0
STEPS = 500          # ~3.3 * focal t_peak: well past the peak, into erosion
SEED = 1
N_SEEDS = 10         # ensemble for the richness trajectory + variance band

# Focal island = the mid-age island (its t_peak lands near STEPS/3 so its
# richness peak falls inside the middle third of the run).
FOCAL_IDX = N_ISLANDS // 2
FOCAL = f"island_{FOCAL_IDX:02d}"

# Hump gate margins: peak must sit in the middle third and clear both the early
# and late richness by a margin (early/late each below 0.75 * peak).
MARGIN_FRAC = 0.75
EDGE_FRAC = 0.10     # first/last 10% of the run define "early"/"late"


def _run_once(core, seed):
    doc = build_gdm_document(
        n_islands=N_ISLANDS, n_pool=N_POOL,
        colonization_base=COLONIZATION_BASE, extinction_base=EXTINCTION_BASE,
        area_max=AREA_MAX, t_peak_min=T_PEAK_MIN, t_peak_max=T_PEAK_MAX,
        ontogeny_shape=ONTOGENY_SHAPE, area_exponent=AREA_EXPONENT,
        ref_area=REF_AREA, seed=seed)
    sim = Composite({"state": doc}, core=core)
    sim.run(float(STEPS))
    return gather_emitter_results(sim)[("emitter",)]


def _smooth(y, w=15):
    """Centered moving average so the peak index is robust to per-tick noise."""
    y = np.asarray(y, dtype=float)
    if len(y) < w or w < 2:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def main() -> int:
    run_id = uuid.uuid4().hex
    params = dict(n_islands=N_ISLANDS, n_pool=N_POOL, area_max=AREA_MAX,
                  t_peak_min=T_PEAK_MIN, t_peak_max=T_PEAK_MAX,
                  ontogeny_shape=ONTOGENY_SHAPE, area_exponent=AREA_EXPONENT,
                  steps=STEPS, seed=SEED, n_seeds=N_SEEDS)
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID,
        "label": STUDY_SLUG, "started_at": time.time(), "status": "running",
        "n_steps": STEPS, "emitter": "ram", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": params,
    })
    try:
        core = build_core()
        register_gdm(core)

        # Primary run (seed=SEED) backs the overlay + hysteresis figures.
        rows = _run_once(core, SEED)
        times = viz.history_times(rows)
        meta = rows[-1]["island_meta"]

        area_focal = np.array([r["area"][FOCAL] for r in rows], dtype=float)
        rich_focal = np.array([r["richness"][FOCAL] for r in rows], dtype=float)

        # Ensemble richness trajectory on the focal island.
        ens = [rich_focal]
        for s in range(SEED + 1, SEED + N_SEEDS):
            r2 = _run_once(core, s)
            ens.append(np.array([r["richness"][FOCAL] for r in r2], dtype=float))
        L = min(len(e) for e in ens)
        M = np.vstack([e[:L] for e in ens])      # (n_seeds, n_times)
        t_ens = times[:L]
        mean_rich = M.mean(axis=0)

        # --- hump detection on the smoothed ensemble mean --------------------
        smooth = _smooth(mean_rich, 15)
        peak_idx = int(np.argmax(smooth))
        peak_rich = float(mean_rich[peak_idx])
        peak_time = float(t_ens[peak_idx])
        edge = max(1, int(L * EDGE_FRAC))
        early_rich = float(np.mean(mean_rich[:edge]))
        late_rich = float(np.mean(mean_rich[-edge:]))

        in_middle_third = (L / 3.0) < peak_idx < (2.0 * L / 3.0)
        clears_early = early_rich < MARGIN_FRAC * peak_rich
        clears_late = late_rich < MARGIN_FRAC * peak_rich
        hump = bool(in_middle_third and clears_early and clears_late)

        # --- visualizations --------------------------------------------------
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz.write_html(
            gdm_area_richness_overlay(times, area_focal, rich_focal,
                                      title=(f"Island ontogeny — {FOCAL} "
                                             f"(t_peak={meta[FOCAL]['t_peak']:.0f})")),
            viz_dir / "area_richness_overlay.html")
        viz.write_html(
            gdm_hysteresis_loop(area_focal, rich_focal, times),
            viz_dir / "richness_area_hysteresis.html")
        viz.write_html(
            gdm_ensemble_richness(t_ens, M),
            viz_dir / "ensemble_richness.html")
        viz_paths = [viz_dir / "area_richness_overlay.html",
                     viz_dir / "richness_area_hysteresis.html",
                     viz_dir / "ensemble_richness.html"]

        outcomes = {
            "peak_time": round(peak_time, 1),
            "peak_richness": round(peak_rich, 2),
            "early_richness": round(early_rich, 2),
            "late_richness": round(late_rich, 2),
            "peak_over_late": round(peak_rich / max(1e-9, late_rich), 2),
            "focal_island": FOCAL,
            "peak_in_middle_third": bool(in_middle_third),
            "richness-is-humped": "PASS" if hump else "FAIL",
        }
        print("=== general-dynamic-model verdicts ===")
        print(f"  focal island = {FOCAL}  t_peak={meta[FOCAL]['t_peak']:.0f}  "
              f"area_max={meta[FOCAL]['area_max']:.0f}")
        print(f"  run length L = {L} ticks; middle third = "
              f"({L / 3.0:.0f}, {2.0 * L / 3.0:.0f})")
        print(f"  peak richness = {peak_rich:.2f} at t = {peak_time:.0f} "
              f"(index {peak_idx}, in middle third? {in_middle_third})")
        print(f"  early richness (first {int(EDGE_FRAC*100)}%) = {early_rich:.2f}")
        print(f"  late  richness (last  {int(EDGE_FRAC*100)}%) = {late_rich:.2f}")
        print(f"  peak / late = {peak_rich / max(1e-9, late_rich):.2f}")
        print(f"  richness-is-humped? {hump}  "
              f"(middle-third={in_middle_third}, clears_early={clears_early}, "
              f"clears_late={clears_late})")
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
