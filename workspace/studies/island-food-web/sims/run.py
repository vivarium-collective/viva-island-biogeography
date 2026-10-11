#!/usr/bin/env python3
"""Canonical run for the ``island-food-web`` study.

Runs the trophic ``island_food_web`` composite across a seed ensemble and a
matched-parameter random-matrix ``island_archipelago`` ensemble, and measures
(a) the across-island corr(richness, log area) for the food web and (b) the
mean-richness gain of the food web over random competition. Renders a
richness-vs-area scatter (food web vs random), a richness-per-island timeseries,
and an animated spatial map. Records the run to ``.pbg/runs.jsonl``.

Run from the workspace root::

    python workspace/studies/island-food-web/sims/run.py
"""
from __future__ import annotations

import random
import time
import uuid
from pathlib import Path

import numpy as np
from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]

from viva_island_biogeography.core import build_core
from viva_island_biogeography.composites.archipelago import island_archipelago
from viva_island_biogeography.composites.food_web import island_food_web
from viva_island_biogeography.composites._common import island_specs
from viva_island_biogeography import viz
from vivarium_workbench.lib.run_log import append_run_event
from vivarium_workbench.lib.run_viz import mirror_viz_dir

SPEC_ID = "viva_island_biogeography.composites.food_web.island_food_web"
STUDY_SLUG = "island-food-web"
INVESTIGATION_SLUG = "island-biogeography"

N_ISLANDS = 8
N_SPECIES = 30
STEPS = 150
SEEDS = range(6)
TAIL = 25

RICHNESS_CORR_MIN = 0.2
RICHNESS_GAIN_MIN = 0.5


def _ensemble_richness(gen, names, **kw):
    rich_runs = []
    last_rows = None
    for seed in SEEDS:
        core = build_core()
        doc = gen(core, n_islands=N_ISLANDS, n_species=N_SPECIES, seed=seed, **kw)
        sim = Composite({"state": doc}, core=core)
        sim.run(float(STEPS))
        rows = gather_emitter_results(sim)[("emitter",)]
        last_rows = rows
        tail = rows[-TAIL:]
        rich_runs.append(np.array([
            np.mean([sum(1 for c in r["islands"][n].values() if c > 0) for r in tail])
            for n in names]))
    return np.mean(rich_runs, axis=0), last_rows


def main() -> int:
    run_id = uuid.uuid4().hex
    params = dict(n_islands=N_ISLANDS, n_species=N_SPECIES, steps=STEPS,
                  seeds=list(SEEDS))
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID,
        "label": STUDY_SLUG, "started_at": time.time(), "status": "running",
        "n_steps": STEPS, "emitter": "ram", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": params,
    })
    try:
        specs = island_specs(N_ISLANDS, N_SPECIES, random.Random(0))
        meta = {s["name"]: {"x": s["x"], "y": s["y"], "area": s["area"]} for s in specs}
        names = [s["name"] for s in specs]
        area = np.array([meta[n]["area"] for n in names])

        fw_rich, fw_rows = _ensemble_richness(
            island_food_web, names, growth_rate=0.03,
            predator_prey_interaction=0.1, introduction_probability=0.05)
        rand_rich, _ = _ensemble_richness(
            island_archipelago, names, growth_rate=0.03,
            interaction_weight=0.1, introduction_probability=0.05)

        richness_area_corr = float(np.corrcoef(fw_rich, np.log(area))[0, 1])
        richness_gain = float(fw_rich.mean() - rand_rich.mean())

        # --- visualizations ---------------------------------------------------
        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        fig = viz.metric_vs_predictor(
            area, fw_rich, names, xtitle="island area", ytitle="equilibrium richness",
            title="Species-area relationship: food web vs random competition",
            logx=True, trend=True)
        # overlay the random-competition points for direct contrast
        import plotly.graph_objects as go
        fig.add_trace(go.Scatter(
            x=area, y=rand_rich, mode="markers", name="random competition",
            marker=dict(size=11, color=viz.CATEGORICAL[7], symbol="x",
                        line=dict(color=viz.SURFACE, width=1)),
            hovertemplate="random<br>area=%{x:.0f}<br>richness=%{y:.2f}<extra></extra>"))
        viz.write_html(fig, viz_dir / "richness_vs_area.html")

        viz.write_html(
            viz.richness_timeseries(fw_rows, names,
                                    title="Food-web species richness per island"),
            viz_dir / "richness_timeseries.html")
        viz.write_html(
            viz.spatial_map_animation(fw_rows, meta, color_by="richness",
                                      title="Food-web archipelago richness over time"),
            viz_dir / "spatial_map.html")
        viz_paths = [
            viz_dir / "richness_vs_area.html",
            viz_dir / "richness_timeseries.html",
            viz_dir / "spatial_map.html",
        ]

        t1 = richness_area_corr > RICHNESS_CORR_MIN
        t2 = richness_gain > RICHNESS_GAIN_MIN
        outcomes = {
            "richness_area_corr": round(richness_area_corr, 4),
            "richness_gain_over_random": round(richness_gain, 4),
            "food_web_mean_richness": round(float(fw_rich.mean()), 3),
            "random_mean_richness": round(float(rand_rich.mean()), 3),
            "food-web-recovers-species-area": "PASS" if t1 else "FAIL",
            "trophic-coexistence-exceeds-random": "PASS" if t2 else "FAIL",
        }
        print("=== island-food-web verdicts ===")
        print(f"  food-web mean richness = {fw_rich.mean():.3f}")
        print(f"  random   mean richness = {rand_rich.mean():.3f}")
        print(f"  richness_area_corr = {richness_area_corr:.4f}  (> {RICHNESS_CORR_MIN}? {t1})")
        print(f"  richness_gain      = {richness_gain:.4f}  (> {RICHNESS_GAIN_MIN}? {t2})")
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
