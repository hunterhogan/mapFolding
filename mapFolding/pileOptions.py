"""Compute possible leaf choices for piles and test pile-domain restrictions.

(AI generated docstring)

You can compute a `ChoicesLeaf` for one pile or a lookup for every pile. For supported maps whose
axis lengths are `2`, bounds, parity, and extra spacing restrict the leaf choices. Other shapes
retain every leaf. You can also apply the individual restrictions separately.

Contents
--------
Functions
	Boolean filters
		filterCeiling
			Test whether `pile` is below the upper domain bound for `leaf`.
		filterDoubleParity
			Test the second pile bit for the special head-plus-one leaf.
		filterFloor
			Test whether `pile` reaches the lower domain bound for `leaf`.
		filterParity
			Test whether `pile` has the parity required by the domain of `leaf`.
	Leaf choices
		getChoicesLeaf
			Compute possible leaves for one pile in the state's map.
		getLookupChoicesLeaf
			Map every pile to its possible leaf choices.
"""
from __future__ import annotations

from functools import cache, partial
from gmpy2 import bit_flip, bit_mask
from mapFolding import leafOrigin, 零, 首零
from mapFolding.beDRY import mapShapeIs2上nDimensions
from mapFolding.beDRYElimination import makeChoicesLeaf
from mapFolding.measure import 工dimensionTail, 工dimension首零, 工totalDimensionsOdd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding.dataBaskets import StateElimination
	from mapFolding.theTypes import ChoicesLeaf, Leaf, Pile, UndeterminedPiles

# IMPROVEMENT formula for pile ranges instead of deconstructing leaf domains. Second best, DRYer code: this
# module just rearranges the code for leaf domains. The formula would likely lead to new functions in
# "reduce permutation space."

#======== Boolean filters ======================================

def filterCeiling(pile: Pile, totalDimensions: int, leaf: Leaf) -> bool:
	"""Test whether `pile` is below the upper domain bound for `leaf`.

	(AI generated docstring)

	You can reject a pile at or above the exclusive upper bound for `leaf`.

	Parameters
	----------
	pile : Pile
		Possible pile index to test.
	totalDimensions : int
		Number of axes in a map whose axis lengths are all `2`.
	leaf : Leaf
		Leaf index whose pile domain supplies the restriction.

	Returns
	-------
	pileBelowCeiling : bool
		`True` when `pile` is below the exclusive upper bound.

	Domain Bounds
	-------------
	Supply a map whose axis lengths are `2`. You can obtain the pile domain from
	`mapFolding.leafDomains.getDomainLeaf` [1]. Combine the predicates to check membership in the
	complete domain.

	References
	----------
	[1] `mapFolding.leafDomains.getDomainLeaf`
	"""
	return pile < int(bit_mask(totalDimensions) ^ bit_mask(totalDimensions - 工dimension首零(leaf))) - 工totalDimensionsOdd(leaf) + 2 - (leaf == leafOrigin)

def filterFloor(pile: Pile, leaf: Leaf) -> bool:
	"""Test whether `pile` reaches the lower domain bound for `leaf`.

	(AI generated docstring)

	You can reject a pile below the inclusive lower bound for `leaf`.

	Parameters
	----------
	pile : Pile
		Possible pile index to test.
	leaf : Leaf
		Leaf index whose pile domain supplies the restriction.

	Returns
	-------
	pileAboveFloor : bool
		`True` when `pile` is at or above the inclusive lower bound.

	Domain Bounds
	-------------
	Supply a map whose axis lengths are `2`. You can obtain the pile domain from
	`mapFolding.leafDomains.getDomainLeaf` [1]. Combine the predicates to check membership in the
	complete domain.

	References
	----------
	[1] `mapFolding.leafDomains.getDomainLeaf`
	"""
	return int(bit_flip(0, 工dimensionTail(leaf) + 1)) + 工totalDimensionsOdd(leaf) - 1 - (leaf == leafOrigin) <= pile

def filterParity(pile: Pile, leaf: Leaf) -> bool:
	"""Test whether `pile` has the parity required by the domain of `leaf`.

	(AI generated docstring)

	You can check whether `pile` and the first possible pile of `leaf` are both even or both odd. Check
	the domain bounds separately.

	Parameters
	----------
	pile : Pile
		Possible pile index to test.
	leaf : Leaf
		Leaf index whose pile domain supplies the restriction.

	Returns
	-------
	parityMatches : bool
		`True` when the pile parity matches the domain start.

	Domain Bounds
	-------------
	Supply a map whose axis lengths are `2`. You can obtain the pile domain from
	`mapFolding.leafDomains.getDomainLeaf` [1]. Combine the predicates to check membership in the
	complete domain.

	References
	----------
	[1] `mapFolding.leafDomains.getDomainLeaf`
	"""
	return (pile & 1) == ((int(bit_flip(0, 工dimensionTail(leaf) + 1)) + 工totalDimensionsOdd(leaf) - 1 - (leaf == leafOrigin)) & 1)

def filterDoubleParity(pile: Pile, totalDimensions: int, leaf: Leaf) -> bool:
	"""Test the second pile bit for the special head-plus-one leaf.

	(AI generated docstring)

	You can enforce the extra spacing of the special head-plus-one leaf. For every other `leaf`, you
	receive `True`.

	Parameters
	----------
	pile : Pile
		Possible pile index to test.
	totalDimensions : int
		Number of axes in a map whose axis lengths are all `2`.
	leaf : Leaf
		Leaf index whose pile domain supplies the restriction.

	Returns
	-------
	secondBitMatches : bool
		`True` when the extra spacing test passes or does not apply.

	Domain Bounds
	-------------
	Supply a map whose axis lengths are `2`. You can obtain the pile domain from
	`mapFolding.leafDomains.getDomainLeaf` [1]. Combine the predicates to check membership in the
	complete domain.

	Second Bit
	----------
	You can identify the special leaf as `首零(totalDimensions) + 零` with the positional-numeral system [2].
	You compare bit `1` of `pile` with bit `1` of the domain start. Combine this predicate with
	`filterParity` to enforce spacing of four piles.

	References
	----------
	[1] `mapFolding.leafDomains.getDomainLeaf`

	[2] `mapFolding._semiotics`
	"""
	if leaf != 首零(totalDimensions) + 零:
		return True
	return (pile >> 1 & 1) == ((int(bit_flip(0, 工dimensionTail(leaf) + 1)) + 工totalDimensionsOdd(leaf) - 1 - (leaf == leafOrigin)) >> 1 & 1)

#======== getChoicesLeaf ======================================

@cache
def _getChoicesLeaf(pile: Pile, totalDimensions: int, mapShape: tuple[int, ...], totalLeaves: int) -> ChoicesLeaf:
	choicesLeaf: Iterable[Leaf] = range(totalLeaves)
	if mapShapeIs2上nDimensions(mapShape):
		parityMatch: Callable[[Leaf], bool] = partial(filterParity, pile)
		pileAboveFloor: Callable[[Leaf], bool] = partial(filterFloor, pile)
		pileBelowCeiling: Callable[[Leaf], bool] = partial(filterCeiling, pile, totalDimensions)
		matchLargerStep: Callable[[Leaf], bool] = partial(filterDoubleParity, pile, totalDimensions)

		choicesLeaf = filter(parityMatch, choicesLeaf)
		choicesLeaf = filter(pileAboveFloor, choicesLeaf)
		choicesLeaf = filter(pileBelowCeiling, choicesLeaf)
		choicesLeaf = filter(matchLargerStep, choicesLeaf)

	return makeChoicesLeaf(totalLeaves, choicesLeaf)


def getChoicesLeaf(state: StateElimination, pile: Pile) -> ChoicesLeaf:
	"""Compute possible leaves for a pile in the state's map.

	(AI generated docstring)

	You can use the map shape in `state` to compute possible leaves at `pile`. You receive bounds and
	parity restrictions from the cached shape-based calculation [1] for supported maps whose dimension
	lengths are all two, or every leaf for other shapes.

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
	[1] `mapFolding.pileOptions._getChoicesLeaf`
	"""
	from mapFolding.pileOptions import _getChoicesLeaf  # ruff: ignore[import-outside-top-level]
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
