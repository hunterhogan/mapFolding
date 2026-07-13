from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import EliminationState
	from mapFolding._e.theTypes import Leaf

def getLeafDomain(state: EliminationState, leaf: Leaf) -> range:
	from mapFolding._e._2上nDimensional.leafDomains import _getLeafDomain  # ruff:ignore[import-outside-top-level]
	return _getLeafDomain(leaf, state.dimensionsTotal, state.mapShape, state.leavesTotal)
