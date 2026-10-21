from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mapFolding.theTypes import Leaf, Pile
    from typing import Final

leafOrigin: Final[Leaf] = 0
"""The `leaf` at the origin of all dimensions, with `0` in every `DimensionIndex`."""

pileOrigin: Final[Pile] = 0
"""The `pile` at the origin of all dimensions, with `0` in every `DimensionIndex`."""
