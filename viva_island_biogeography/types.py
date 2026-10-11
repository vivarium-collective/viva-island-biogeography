"""Custom bigraph-schema types for viva-island-biogeography.

The island model composes entirely out of built-in schema types
(``map[string,float]`` abundances, ``float`` richness/time), so there are no
bespoke types to register. ``register_types`` is kept as the conventional
module-level hook (``recursive_dynamic_import`` / ``build_core`` call it) and
simply returns ``core`` unchanged.
"""

from __future__ import annotations


def register_types(core):
    """No custom types needed; hand ``core`` back unchanged (required)."""
    return core
