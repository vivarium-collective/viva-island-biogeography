"""Workspace derived-scalar evaluators.

The discovery-engine grader resolves a behavior test whose ``measure`` is
``{kind: derived_scalar, field: <name>}`` by first looking for ``<name>`` as an
emitted observable and, failing that, calling a workspace-registered computer.
``load_workspace_derived_scalars`` imports THIS module and calls
``register_derived_scalars(reg)``; each per-study submodule contributes its
field computers so the engine grades those tests by CODE against the real
persisted run store, instead of SKIP-ing them (agent bucket) as unresolvable.

Each computer recomputes its study's published metric from emitted observables
only — the same math the study's ``sims/run.py`` runs — so a PASS is earned by a
run, never asserted. No metric here introduces a mechanism or parameter the
study did not already use.
"""
from __future__ import annotations


# Per-study computer modules. Each defines ``register(reg)``; each is imported
# independently so a not-yet-created sibling (during a parallel rollout) never
# blocks the modules that DO exist.
_STUDY_EVALUATOR_MODULES = (
    "evaluators_archipelago",
    "evaluators_distance_immigration",
    "evaluators_food_web",
    "evaluators_gdm",
    "evaluators_microbial",
    "evaluators_mw_equilibrium",
    "evaluators_neutral",
    "evaluators_rescue",
    "evaluators_species_area",
)


def register_derived_scalars(reg: dict) -> None:
    import importlib
    for name in _STUDY_EVALUATOR_MODULES:
        try:
            mod = importlib.import_module(f"{__package__}.{name}")
        except ModuleNotFoundError:
            continue  # sibling study module not present yet
        mod.register(reg)
