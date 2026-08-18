from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import StateElimination
	from mapFolding._e.theTypes import ChoicesLeaf, Pile, UndeterminedPiles

def getChoicesLeaf(state: StateElimination, pile: Pile) -> ChoicesLeaf:
	from mapFolding._e._2上nDimensional.pileOptions import _getChoicesLeaf  # ruff: ignore[import-outside-top-level]
	return _getChoicesLeaf(pile, state.totalDimensions, state.mapShape, state.totalLeaves)

def getDictionaryChoicesLeaf(state: StateElimination) -> UndeterminedPiles:
	"""At `pile`, which `leaf` values may be found in a `folding`: the mathematical range, not a Python `range` object.

	Returns
	-------
	pilesUndetermined: UndeterminedPiles
		`pile: choicesLeaf` for each `pile` in the `folding`, where `choicesLeaf` is a bitset of all
		`leaf` values that may be found at that `pile`.
	"""
	return {pile: getChoicesLeaf(state, pile) for pile in range(state.totalLeaves)}
