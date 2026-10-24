"""Workspace core builder.

The dashboard (and this repo's own composite generators / study runners) run
each composite in a subprocess that calls this ``build_core()`` to obtain a
core with viva-island-biogeography's own process classes registered. Editable
installs are not always visible to ``process_bigraph.allocate_core()``'s
distribution-keyed auto-discovery (uv-build / pixi editable installs list no
file manifest), so ``local:IslandProcess`` / ``local:MigrationProcess`` can
fail to resolve with "no link found at address". Register them explicitly here
instead — see ``docs/conventions/discovery.md`` and the viva-fenics precedent.
"""

from __future__ import annotations

from process_bigraph import allocate_core

from .processes import (
    IslandProcess,
    MigrationProcess,
    MacArthurWilsonEquilibriumProcess,
)
from .models_neutral import NeutralMetacommunityProcess
from .models_gdm import GeneralDynamicModelProcess
from .models_rescue import RescueEffectProcess
from .types import register_types

_PROCESSES = (
    ("IslandProcess", IslandProcess),
    ("MigrationProcess", MigrationProcess),
    ("MacArthurWilsonEquilibriumProcess", MacArthurWilsonEquilibriumProcess),
    ("NeutralMetacommunityProcess", NeutralMetacommunityProcess),
    ("GeneralDynamicModelProcess", GeneralDynamicModelProcess),
    ("RescueEffectProcess", RescueEffectProcess),
)


def register_processes(core):
    """Register this workspace's Process classes into ``core``."""
    for name, cls in _PROCESSES:
        core.register_link(name, cls)
    return core


def build_core(core=None):
    """Return a core with viva-island-biogeography's types + processes registered."""
    if core is None:
        core = allocate_core()
    register_types(core)
    register_processes(core)
    return core
