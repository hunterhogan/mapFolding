from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mapFolding._e.theTypes import Leaf, Pile

leafOrigin: Leaf = 0
"""The `leaf` at the origin of all dimensions, with `0` in every `DimensionIndex`."""
pileOrigin: Pile = 0
"""The `pile` at the origin of all dimensions, with `0` in every `DimensionIndex`."""
