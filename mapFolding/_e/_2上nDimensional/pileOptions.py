from __future__ import annotations

from functools import cache
from gmpy2 import bit_flip, bit_mask
from humpy_cytoolz import curry as syntacticCurry
from mapFolding._e import leafOrigin, makeLeafOptions
from mapFolding._e._2上nDimensional import (
	dimensionNearestTail, dimensionNearest首, howManyDimensionsHaveOddParity, mapShapeIs2上nDimensions, 零, 首零)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding._e.theTypes import Leaf, LeafOptions, Pile

#======== Boolean filters ======================================

@syntacticCurry
def filterCeiling(pile: Pile, dimensionsTotal: int, leaf: Leaf) -> bool:
	return pile < int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(leaf))) - howManyDimensionsHaveOddParity(leaf) + 2 - (leaf == leafOrigin)

@syntacticCurry
def filterFloor(pile: Pile, leaf: Leaf) -> bool:
	return int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin) <= pile

@syntacticCurry
def filterParity(pile: Pile, leaf: Leaf) -> bool:
	return (pile & 1) == ((int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin)) & 1)

@syntacticCurry
def filterDoubleParity(pile: Pile, dimensionsTotal: int, leaf: Leaf) -> bool:
	if leaf != 首零(dimensionsTotal) + 零:
		return True
	return (pile >> 1 & 1) == ((int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin)) >> 1 & 1)

#======== getLeafOptions ======================================

@cache
def _getLeafOptions(pile: Pile, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> LeafOptions:
	leafOptions: Iterable[Leaf] = range(leavesTotal)
	if mapShapeIs2上nDimensions(mapShape):
		parityMatch: Callable[[Leaf], bool] = filterParity(pile)
		pileAboveFloor: Callable[[Leaf], bool] = filterFloor(pile)
		pileBelowCeiling: Callable[[Leaf], bool] = filterCeiling(pile, dimensionsTotal)
		matchLargerStep: Callable[[Leaf], bool] = filterDoubleParity(pile, dimensionsTotal)

		leafOptions = filter(parityMatch, leafOptions)
		leafOptions = filter(pileAboveFloor, leafOptions)
		leafOptions = filter(pileBelowCeiling, leafOptions)
		leafOptions = filter(matchLargerStep, leafOptions)

	return makeLeafOptions(leavesTotal, leafOptions)
