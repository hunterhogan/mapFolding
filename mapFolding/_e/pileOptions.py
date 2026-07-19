from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import EliminationState
	from mapFolding._e.theTypes import LeafOptions, Pile, UndeterminedPiles

def getLeafOptions(state: EliminationState, pile: Pile) -> LeafOptions:
	from mapFolding._e._2上nDimensional.pileOptions import _getLeafOptions  # ruff:ignore[import-outside-top-level]
	return _getLeafOptions(pile, state.dimensionsTotal, state.mapShape, state.leavesTotal)

def getDictionaryLeafOptions(state: EliminationState) -> UndeterminedPiles:
	"""At `pile`, which `leaf` values may be found in a `folding`: the mathematical range, not a Python `range` object.

	Returns
	-------
	pilesUndetermined: UndeterminedPiles
		`pile: leafOptions` for each `pile` in the `folding`, where `leafOptions` is a bitset of all
		`leaf` values that may be found at that `pile`.
	"""
	return {pile: getLeafOptions(state, pile) for pile in range(state.leavesTotal)}
