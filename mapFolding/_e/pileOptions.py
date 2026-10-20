"""Compute `ChoicesLeaf` values for map piles.

(AI generated docstring)

You can obtain shape-based leaf choices for one pile or every pile. You receive `ChoicesLeaf` values independent of
the current pinning or reductions in an elimination state.

Contents
--------
Functions
	getChoicesLeaf
		Compute possible leaves for a pile in the state's map.
	getLookupChoicesLeaf
		Build `ChoicesLeaf` values keyed by pile index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding._e.dataBaskets import StateElimination
	from mapFolding._e.theTypes import ChoicesLeaf, UndeterminedPiles
	from mapFolding.theTypes import Pile

def getChoicesLeaf(state: StateElimination, pile: Pile) -> ChoicesLeaf:
	"""Compute possible leaves for a pile in the state's map.

	(AI generated docstring)

	You can use the map shape in `state` to compute possible leaves at `pile`. You receive bounds and
	parity restrictions from the specialized pile-options implementation [1] when every dimension length
	is two, or every leaf for other shapes.

	Parameters
	----------
	state : StateElimination
		Elimination state providing the map shape and leaf count.
	pile : Pile
		Pile index whose leaf choices are requested.

	Returns
	-------
	choicesLeaf : ChoicesLeaf
		`ChoicesLeaf` including the sentinel at the total leaf count.

	See Also
	--------
	`getLookupChoicesLeaf`
		Compute `ChoicesLeaf` values for every pile.

	References
	----------
	[1] `mapFolding._e.p2上nDimensional.pileOptions._getChoicesLeaf`
	"""
	from mapFolding._e.p2上nDimensional.pileOptions import _getChoicesLeaf  # ruff: ignore[import-outside-top-level]
	return _getChoicesLeaf(pile, state.totalDimensions, state.mapShape, state.totalLeaves)

def getLookupChoicesLeaf(state: StateElimination) -> UndeterminedPiles:
	"""At `pile`, which `leaf` values may be found in a `folding`: the mathematical range, not a Python `range` object.

	Returns
	-------
	pilesUndetermined: UndeterminedPiles
		`pile: choicesLeaf` for each `pile` in the `folding`, where `choicesLeaf` is a `ChoicesLeaf` of all
		`leaf` values that may be found at that `pile`.
	"""
	return {pile: getChoicesLeaf(state, pile) for pile in range(state.totalLeaves)}
