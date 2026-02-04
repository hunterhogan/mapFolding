#======== (mathematical) ranges of piles ====================
from bisect import bisect_left
from collections.abc import Callable, Iterable, Iterator
from cytoolz.functoolz import curry as syntacticCurry
from functools import cache, partial
from gmpy2 import bit_flip, bit_mask, is_even, is_odd
from hunterMakesPy import raiseIfNone
from mapFolding._e import (
	dimensionNearestTail, dimensionNearest首, howManyDimensionsHaveOddParity, invertLeafIn2上nDimensions, Leaf, leafOrigin,
	mapShapeIs2上nDimensions, Pile, 零, 首一, 首二, 首零, 首零一)
from mapFolding._e.dataBaskets import EliminationState
from more_itertools import flatten
from operator import add, iadd, isub, mul
from pprint import pprint
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import pandas

# TODO formula for pile ranges instead of deconstructing leaf domains. Second best, DRYer code.

@syntacticCurry
def filterCeiling(pile: Pile, dimensionsTotal: int, leaf: Leaf) -> bool:
	return pile <  int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(leaf))) - howManyDimensionsHaveOddParity(leaf) + 2 - (leaf == leafOrigin)

@syntacticCurry
def filterFloor(pile: Pile, leaf: Leaf) -> bool:
	return int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin) <= pile

@syntacticCurry
def filterParity(pile: Pile, leaf: Leaf) -> bool:
	return (pile & 1) == ((int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin)) & 1)

@syntacticCurry
def filterDoubleParity(pile: Pile, dimensionsTotal: int, leaf: Leaf) -> bool:
	if leaf != 首零(dimensionsTotal)+零:
		return True
	return (pile >> 1 & 1) == ((int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin)) >> 1 & 1)

def getPileRange(state: EliminationState, pile: Pile) -> Iterator[Leaf]:
	return iter(_getPileRange(pile, state.dimensionsTotal, state.mapShape, state.leavesTotal))
@cache
def _getPileRange(pile: Pile, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> tuple[Leaf, ...]:
	if mapShapeIs2上nDimensions(mapShape):
		parityMatch: Callable[[Leaf], bool] = filterParity(pile)
		pileAboveFloor: Callable[[Leaf], bool] = filterFloor(pile)
		pileBelowCeiling: Callable[[Leaf], bool] = filterCeiling(pile, dimensionsTotal)
		matchLargerStep: Callable[[Leaf], bool] = filterDoubleParity(pile, dimensionsTotal)

		pileRange: Iterable[Leaf] = range(leavesTotal)
		pileRange = filter(parityMatch, pileRange)
		pileRange = filter(pileAboveFloor, pileRange)
		pileRange = filter(pileBelowCeiling, pileRange)
		return tuple(filter(matchLargerStep, pileRange))

	return tuple(range(leavesTotal))

def getDictionaryPileRanges(state: EliminationState) -> dict[Pile, tuple[Leaf, ...]]:
	"""At `pile`, which `leaf` values may be found in a `folding`: the mathematical range, not a Python `range` object."""
	return {pile: tuple(getPileRange(state, pile)) for pile in range(state.leavesTotal)}

# ruff: noqa: ERA001 T201 T203  # noqa: RUF100

def _getGroupedBy(state: EliminationState, pileTarget: Pile, groupByLeavesAtPiles: tuple[Pile, ...]) -> dict[Leaf | tuple[Leaf, ...], list[Leaf]]:
	from mapFolding._e.dataDynamic import getDataFrameFoldings  # noqa: PLC0415

	dataframeFoldings: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
	groupedBy: dict[Leaf | tuple[Leaf, ...], list[Leaf]] = dataframeFoldings.groupby(list(groupByLeavesAtPiles))[pileTarget].apply(list).to_dict() # pyright: ignore[reportAssignmentType]
	return {leaves: sorted(set(listLeaves)) for leaves, listLeaves in groupedBy.items()}

def getExcludedLeaves(state: EliminationState, pileTarget: Pile, groupByLeavesAtPiles: tuple[Pile, ...]) -> dict[Leaf | tuple[Leaf, ...], list[Leaf]]:
	return {leaves: sorted(set(getDictionaryPileRanges(state)[pileTarget]).difference(set(listLeaves))) for leaves, listLeaves in _getGroupedBy(state, pileTarget, groupByLeavesAtPiles).items()}

if __name__ == '__main__':

	state = EliminationState((2,) * 6)
	"""
000011	3
		5	(5, 6, 10, 18, 34)
		9	(9, 10, 12, 20, 36)
001111	15
		17	17	(17, 18, 20, 24, 40)
010111	(23, 24, 40)
011011	(27, 29, 45)
		33	33	(33, 34, 36, 40)
realRange=(3, 5, 6, 9, 10, 15, 17, 18, 23, 27, 29, 34, 39, 43, 45, 51, 53, 57)
100111	39		(39, 40)
101011		43
			45	(45, 46, 54)
110011	51
		53	53	(53, 54, 58)
		57		(57, 58, 60)

111111	63

even bit count
0	0	00	11	its creases: crease+1
0	0	11	11	its creases: crease+1
0	1	01	11	crease+1
0	1	10	11	its creases: crease+1

odd bit count
1	0	01	11	crease+1
1	0	10	11	its creases: crease+1
1	1	00	11	its creases: crease+1
1	1	11	11	n/a

tt = (3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30, 33, 34, 36, 39, 40, 43, 45, 46, 51, 53, 54, 57, 58, 60, 63)
pp = (1, 2, 4, 8, 16, 32)

pp63 = (63,)
pp60 = (60,)
pp58 = (58, 60)
pp57 = (57, 58, 60)
pp54 = (54, 58)
pp53 = (53, 54, 58)
pp51 = (51, 53, 57)
pp46 = (46, 54)
pp45 = (45, 46, 54)
pp43 = (43, 45, 53)
pp40 = (40,)
pp39 = (39, 40)
pp36 = (36, 40)
pp34 = (34, 36, 40)
pp33 = (33, 34, 36, 40)
pp30 = (30, 34)
pp29 = (29, 30, 34)
pp27 = (27, 29, 45)
pp24 = (24, 40)
pp23 = (23, 24, 40)
pp20 = (20, 24, 40)
pp18 = (18, 20, 24, 40)
pp17 = (17, 18, 20, 24, 40)
pp15 = (15, 17, 33)
pp12 = (12, 20, 36)
pp10 = (10, 12, 20, 36)
pp9  = (9, 10, 12, 20, 36)
pp6  = (6, 10, 18, 34)
pp5  = (5, 6, 10, 18, 34)
pp3  = (3, 5, 9, 17, 33)

	"""

	pile: Pile = 4
	pileDimension = bisect_left(state.sumsOfProductsOfDimensionsNearest首, pile>>1<<1)
	leafMinimum = is_even(pile) + state.productsOfDimensions[pileDimension]
	pileRange: list[Leaf] = []

	pileRange.append(leafMinimum)

	if is_even(pile):
		dd = pileDimension

		ss = state.sumsOfProductsOfDimensions[dd]
		pileRange.extend(map(partial(iadd, leafMinimum - ss), state.sumsOfProductsOfDimensions[1:dd]))
		pileRange.extend(map(partial(iadd, leafMinimum - ss), state.sumsOfProductsOfDimensions[dd + 1: state.dimensionsTotal]))

		if dd < dimensionNearest首(pile):
			dd += 1

			ss = state.productsOfDimensions[dd]
			# pileRange.extend(map(partial(isub, leafMinimum + ss), state.sumsOfProductsOfDimensions[1:dd]))
			# pileRange.extend(map(partial(iadd, leafMinimum + ss), state.productsOfDimensions[dd + 1: state.dimensionsTotal]))


		if (pile % 4 == 0) and ((零)+首零(state.dimensionsTotal) in pileRange):
			pileRange.remove((零)+首零(state.dimensionsTotal))
			"""33 has step = 4"""


	if is_odd(pile):
		dd = pileDimension

		ss = state.sumsOfProductsOfDimensions[dd]
		pileRange.extend(map(partial(iadd, leafMinimum + ss), state.productsOfDimensions[1:dd]))
		pileRange.extend(map(partial(iadd, leafMinimum + ss), state.productsOfDimensions[dd + 1: state.dimensionsTotal]))

		dd += 1

		ss = state.sumsOfProductsOfDimensions[dd]
		pileRange.extend(map(partial(iadd, leafMinimum + ss), state.productsOfDimensions[1:dd]))
		pileRange.extend(map(partial(iadd, leafMinimum + ss), state.productsOfDimensions[dd + 1: state.dimensionsTotal]))

		dd += 1

		ss = state.sumsOfProductsOfDimensions[dd]
		pileRange.extend(map(partial(iadd, leafMinimum + ss), state.productsOfDimensions[1:dd]))
		pileRange.extend(map(partial(iadd, leafMinimum + ss), state.productsOfDimensions[dd + 1: state.dimensionsTotal]))

		dd += 1

		ss = state.sumsOfProductsOfDimensions[dd]
		pileRange.extend(map(partial(iadd, leafMinimum + ss), state.productsOfDimensions[1:dd]))
		pileRange.extend(map(partial(iadd, leafMinimum + ss), state.productsOfDimensions[dd + 1: state.dimensionsTotal]))

	print(f"{pile=}\t{pileDimension=}")
	print("computed=", sorted(set(pileRange)))
	realRange = tuple(getPileRange(state, pile))
	print(f"{realRange=}")
	pileAnte = tuple(getPileRange(state, pile - 1))
	print(f"{pileAnte=}")


	pileRangeByFormula: bool = False
	if pileRangeByFormula:
		state = EliminationState((2,) * 6)

		# NOTE works for 9 <= odd piles <= 47
		# I _think_ I need to be able to pass start/stop to intraDimensionalLeaves
		# Yes, sort of. `Z0Z_alphaBeta` and `intraDimensionalLeaves` need to be the same function: and I need to be able to tweak all of the parameters.

		@syntacticCurry
		def intraDimensionalLeaves(state: EliminationState, dimensionOrigin: int) -> list[int]:
			return list(map(partial(add, dimensionOrigin+2), state.sumsOfProductsOfDimensions[1: dimensionNearest首(dimensionOrigin)]))

		@syntacticCurry
		def Z0Z_alphaBeta(state: EliminationState, alphaStart: int = 0, betaStop: int = 0, charlieStep: int = 1) -> list[int]:
			return list(flatten(map(intraDimensionalLeaves(state), state.productsOfDimensions[2 + alphaStart: (state.dimensionsTotal - 1) + betaStop: charlieStep])))

		def Z0Z_getPileRange(state: EliminationState, pile: Pile) -> Iterable[Leaf]:
			pileRange: list[Leaf] = []

			# odd leaves < 32.
			# ? 12 < even leaves < 32.
			# ? 24 < even leaves < 32.
			# piles 49, 51, 53, 55 need a higher start on yy=0.
			for yy in range(3):
				pileRange.extend(map(partial(mul, state.productsOfDimensions[yy]), Z0Z_alphaBeta(state, betaStop=-(yy))))

			# 32 < even leaves
			for yy in range(1):
				pileRange.extend(map(partial(invertLeafIn2上nDimensions, state.dimensionsTotal), map(partial(mul, state.productsOfDimensions[yy])
					, Z0Z_alphaBeta(state
						, alphaStart=yy+(state.dimensionsTotal - 2 - dimensionNearest首(pile))
						, betaStop=-(yy)
					))))
			# ? 32 < odd leaves < 52
			# ? 32 < odd leaves < 36
			for yy in range(1,3):
				pileRange.extend(map(partial(invertLeafIn2上nDimensions, state.dimensionsTotal), map(partial(mul, state.productsOfDimensions[yy]), Z0Z_alphaBeta(state, betaStop=-(yy)))))

			# dimension origins
			# piles 51, 53, 55 need a higher start.
			pileRange.extend(state.productsOfDimensions[1 + ((零)+首零(state.dimensionsTotal) < pile):dimensionNearest首(pile+1)])
			# inverse dimension origins: 62, 61, 59, 55, 47, 31
			# pile5 needs a higher start.
			pileRange.extend(map(partial(invertLeafIn2上nDimensions, state.dimensionsTotal), state.productsOfDimensions[0:state.dimensionsTotal]))

			return tuple(sorted(pileRange))

		def Z0Z_getPileRangeEven(state: EliminationState, pile: Pile) -> Iterable[Leaf]:
			pileRange: list[Leaf] = []

			for yy in range(3):
				pileRange.extend(map(
					partial(add, 1)
					, (map(
						partial(mul, state.productsOfDimensions[yy])
						, Z0Z_alphaBeta(state, alphaStart = 0, betaStop=-(yy))
				)
			)
		)
	)

			# for yy in range(1):
			# 	pileRange.extend(map(partial(Z0Z_invert, state.dimensionsTotal), map(partial(mul, state.productsOfDimensions[yy])
			# 		, Z0Z_alphaBeta(state
			# 			, alphaStart=yy+(state.dimensionsTotal - 2 - dimensionNearest首(pile))
			# 			, betaStop=-(yy)
			# 		))))
			# for yy in range(1,3):
			# 	pileRange.extend(map(partial(Z0Z_invert, state.dimensionsTotal), map(partial(mul, state.productsOfDimensions[yy]), Z0Z_alphaBeta(state, betaStop=-(yy)))))

			# dimension origins
			pileRange.extend(map(partial(add, 1), state.productsOfDimensions[1 + ((零)+首零(state.dimensionsTotal) < pile):dimensionNearest首(pile+1)]))
			# inverse dimension origins: 62, 61, 59, 55, 47, 31
			pileRange.extend(map(partial(invertLeafIn2上nDimensions, state.dimensionsTotal), map(partial(add, 1), state.productsOfDimensions[1:state.dimensionsTotal])))

			return tuple(sorted(pileRange))

		for pile in range(首一(state.dimensionsTotal), 首零一(state.dimensionsTotal), 2):
			print(pile, (real:=tuple(getPileRange(state, pile))) == (computed:=Z0Z_getPileRangeEven(state, pile)), end=': ')
			# print(f"{ansiColors.Green}surplus: {set(computed).difference(real)}", f"{ansiColors.Magenta}missing: {set(real).difference(computed)}{ansiColorReset}", sep='\n')
			pprint(f"{computed=}", width=180)

		for pile in range((零)+首二(state.dimensionsTotal), 首零一(state.dimensionsTotal), 2):
			print(pile, (real:=tuple(getPileRange(state, pile))) == (computed:=Z0Z_getPileRange(state, pile)), end=': ')
			# print(f"surplus: {set(computed).difference(real)}", f"missing: {set(real).difference(computed)}", sep='\n')
			pprint(f"{computed=}", width=180)

			# > 32: matches most tail0s != 1
			# if pile > 32:
			# 	pile-=1
			# else:
			# 	pile+=1
			# zz = tuple(map(partial(xor, 1), zz))
			# print(pile, (ll:=getPileRange(state, pile)) == (zz), end=': ')
			# # print(set(zz).difference(ll), set(ll).difference(zz), sep='\t')
			# pprint(zz, width=180)

