from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mapFolding.theTypes import Endpoint, EndpointPosition, Leaf, Pile
    from typing import Final

endpointOrigin: Final[Endpoint] = 0
"""The endpoint label fixed at the origin of a normalized arch diagram."""

positionOrigin: Final[EndpointPosition] = 0
"""The first endpoint position along the diagram axis."""

leafOrigin: Final[Leaf] = 0
"""The `leaf` at the origin of all dimensions, with `0` in every `DimensionIndex`."""

pileOrigin: Final[Pile] = 0
"""The `pile` at the origin of all dimensions, with `0` in every `DimensionIndex`."""
