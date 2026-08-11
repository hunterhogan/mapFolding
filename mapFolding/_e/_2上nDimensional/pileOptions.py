from __future__ import annotations

from functools import cache, partial
from gmpy2 import bit_flip, bit_mask
from mapFolding._e import leafOrigin, makeChoicesLeaf
from mapFolding._e._2上nDimensional import dimensionNearestTail, dimensionNearest首, howManyDimensionsHaveOddParity, 零, 首零
from mapFolding.beDRY import mapShapeIs2上nDimensions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding._e.theTypes import ChoicesLeaf, Leaf, Pile

# TODO formula for pile ranges instead of deconstructing leaf domains. Second best, DRYer code.

#======== Boolean filters ======================================

def filterCeiling(pile: Pile, dimensionsTotal: int, leaf: Leaf) -> bool:
	return pile < int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(leaf))) - howManyDimensionsHaveOddParity(leaf) + 2 - (leaf == leafOrigin)

def filterFloor(pile: Pile, leaf: Leaf) -> bool:
	return int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin) <= pile

def filterParity(pile: Pile, leaf: Leaf) -> bool:
	return (pile & 1) == ((int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin)) & 1)

def filterDoubleParity(pile: Pile, dimensionsTotal: int, leaf: Leaf) -> bool:
	if leaf != 首零(dimensionsTotal) + 零:
		return True
	return (pile >> 1 & 1) == ((int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin)) >> 1 & 1)

#======== getChoicesLeaf ======================================

@cache
def _getChoicesLeaf(pile: Pile, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> ChoicesLeaf:
	choicesLeaf: Iterable[Leaf] = range(leavesTotal)
	if mapShapeIs2上nDimensions(mapShape):
		parityMatch: Callable[[Leaf], bool] = partial(filterParity, pile)
		pileAboveFloor: Callable[[Leaf], bool] = partial(filterFloor, pile)
		pileBelowCeiling: Callable[[Leaf], bool] = partial(filterCeiling, pile, dimensionsTotal)
		matchLargerStep: Callable[[Leaf], bool] = partial(filterDoubleParity, pile, dimensionsTotal)

		choicesLeaf = filter(parityMatch, choicesLeaf)
		choicesLeaf = filter(pileAboveFloor, choicesLeaf)
		choicesLeaf = filter(pileBelowCeiling, choicesLeaf)
		choicesLeaf = filter(matchLargerStep, choicesLeaf)

	return makeChoicesLeaf(leavesTotal, choicesLeaf)
