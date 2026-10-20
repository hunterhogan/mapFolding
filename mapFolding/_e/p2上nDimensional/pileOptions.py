"""Test possible pile positions against coordinate domain restrictions.

(AI generated docstring)

You can check the bounds, parity, and extra spacing of pile positions for maps whose axis lengths are
`2`. Combine these separate restrictions to test the complete individual domain.

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
"""
from __future__ import annotations

from functools import cache, partial
from gmpy2 import bit_flip, bit_mask
from mapFolding import leafOrigin
from mapFolding._e import makeChoicesLeaf
from mapFolding._e.p2上nDimensional import 工dimensionTail, 工dimension首零, 工totalDimensionsOdd, 零, 首零
from mapFolding.beDRY import mapShapeIs2上nDimensions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding._e.theTypes import ChoicesLeaf
	from mapFolding.theTypes import Leaf, Pile

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
	`mapFolding._e.leafDomains.getDomainLeaf` [1]. Combine the predicates to check membership in the
	complete domain.

	References
	----------
	[1] `mapFolding._e.leafDomains.getDomainLeaf`
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
	`mapFolding._e.leafDomains.getDomainLeaf` [1]. Combine the predicates to check membership in the
	complete domain.

	References
	----------
	[1] `mapFolding._e.leafDomains.getDomainLeaf`
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
	`mapFolding._e.leafDomains.getDomainLeaf` [1]. Combine the predicates to check membership in the
	complete domain.

	References
	----------
	[1] `mapFolding._e.leafDomains.getDomainLeaf`
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
	`mapFolding._e.leafDomains.getDomainLeaf` [1]. Combine the predicates to check membership in the
	complete domain.

	Second Bit
	----------
	You can identify the special leaf as `首零(totalDimensions) + 零` with the positional-numeral system [2].
	You compare bit `1` of `pile` with bit `1` of the domain start. Combine this predicate with
	`filterParity` to enforce spacing of four piles.

	References
	----------
	[1] `mapFolding._e.leafDomains.getDomainLeaf`

	[2] `mapFolding._e.p2上nDimensional.semiotics`
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
