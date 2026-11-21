"""Python source texts, each named by the statements it holds, for the transformations in `mapFolding.kitAST`.

Each text is formatted as `ast.unparse` formats it, so a transformation that changes nothing returns the text unchanged.
Consumers parse a text into a new tree before transforming it, and do not mutate the texts.
"""
from __future__ import annotations

sourceFunctionsChicken: str = """def chicken(egg):
    return chicken.feathers + rooster.chicken + egg

def coop():

    def chicken():
        return 13
    return chicken"""
"""A module-level function and a nested function share the identifier `chicken`, which is also a name and an attribute."""

sourceImportsTopLevel: str = """from __future__ import annotations
from collections import deque
from typing import TYPE_CHECKING
import numpy
if TYPE_CHECKING:
    from pathlib import Path
if sys.version_info >= (3, 14):
    import tomllib

def chicken(egg: Path):
    import math
    return deque((math.pi, egg))"""
"""Top-level imports, an import-only `TYPE_CHECKING` block, a conditional import, and an import inside a function."""
