from __future__ import annotations

from functools import cache, partial
from gmpy2 import bit_flip, bit_mask
from mapFolding._e import leafOrigin, makeChoicesLeaf
from mapFolding._e._2上nDimensional import 工dimensionTail, 工dimension首零, 工totalDimensionsOdd, 零, 首零
from mapFolding.beDRY import mapShapeIs2上nDimensions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding._e.theTypes import ChoicesLeaf, Leaf, Pile

# IMPROVEMENT formula for pile ranges instead of deconstructing leaf domains. Second best, DRYer code: this
# module just rearranges the code for leaf domains. The formula would likely lead to new functions in
# "reduce permutation space."

#======== Boolean filters ======================================

def filterCeiling(pile: Pile, totalDimensions: int, leaf: Leaf) -> bool:
	return pile < int(bit_mask(totalDimensions) ^ bit_mask(totalDimensions - 工dimension首零(leaf))) - 工totalDimensionsOdd(leaf) + 2 - (leaf == leafOrigin)

def filterFloor(pile: Pile, leaf: Leaf) -> bool:
	return int(bit_flip(0, 工dimensionTail(leaf) + 1)) + 工totalDimensionsOdd(leaf) - 1 - (leaf == leafOrigin) <= pile

def filterParity(pile: Pile, leaf: Leaf) -> bool:
	return (pile & 1) == ((int(bit_flip(0, 工dimensionTail(leaf) + 1)) + 工totalDimensionsOdd(leaf) - 1 - (leaf == leafOrigin)) & 1)

def filterDoubleParity(pile: Pile, totalDimensions: int, leaf: Leaf) -> bool:
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
