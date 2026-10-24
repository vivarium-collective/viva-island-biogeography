"""Composite generators for viva-island-biogeography.

Importing the submodules fires the ``@composite_generator`` decorators so the
generators register in the shared registry on package import.
"""

from . import archipelago  # noqa: F401
from . import food_web  # noqa: F401
from . import mw_equilibrium  # noqa: F401
from . import microbial  # noqa: F401
from . import gdm  # noqa: F401
from . import rescue  # noqa: F401
from . import neutral  # noqa: F401

from .archipelago import island_archipelago
from .food_web import island_food_web
from .mw_equilibrium import mw_equilibrium
from .microbial import microbial_island
from .gdm import general_dynamic_model
from .rescue import rescue_effect
from .neutral import neutral_metacommunity

__all__ = [
    "island_archipelago", "island_food_web", "mw_equilibrium",
    "microbial_island", "general_dynamic_model", "rescue_effect",
    "neutral_metacommunity",
]
