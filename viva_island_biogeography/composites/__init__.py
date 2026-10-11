"""Composite generators for viva-island-biogeography.

Importing the submodules fires the ``@composite_generator`` decorators so the
generators register in the shared registry on package import.
"""

from . import archipelago  # noqa: F401
from . import food_web  # noqa: F401

from .archipelago import island_archipelago
from .food_web import island_food_web

__all__ = ["island_archipelago", "island_food_web"]
