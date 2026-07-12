from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import EliminationState
	from mapFolding._e.theTypes import LeafOptions, Pile

def getLeafOptions(state: EliminationState, pile: Pile) -> LeafOptions:
	from mapFolding._e._2上nDimensional.pileOptions import _getLeafOptions  # ruff:ignore[import-outside-top-level]
	return _getLeafOptions(pile, state.dimensionsTotal, state.mapShape, state.leavesTotal)
