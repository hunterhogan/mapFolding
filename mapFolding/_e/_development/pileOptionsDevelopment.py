# DEVELOPMENT module.
# pyright: reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
# ruff: file-ignore[import-outside-top-level, commented-out-code, print, p-print]
# ty: ignore[invalid-argument-type]
from __future__ import annotations

from bisect import bisect_left
from functools import cache, partial
from gmpy2 import bit_flip, bit_mask, is_even as isEven吗, is_odd as isOdd吗
from humpy_toolz.curried.operator import add, iadd, mul
from hunterMakesPy import raiseIfNone
from itertools import filterfalse
from mapFolding._e import getIteratorOfLeaves, leafOrigin, makeChoicesLeaf
from mapFolding._e._2上nDimensional import (
	dimensionNearestTail, dimensionNearest首, howManyDimensionsHaveOddParity, invertLeafIn2上nDimensions, 零, 首一, 首二, 首零, 首零一)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pileOptions import getChoicesLeaf
from mapFolding.beDRY import mapShapeIs2上nDimensions
from more_itertools import flatten
from pprint import pprint
from typing import TYPE_CHECKING
from Z0Z_tools import DOTitems

if TYPE_CHECKING:
	from collections.abc import Callable, Iterable
	from mapFolding._e.theTypes import ChoicesLeaf, Leaf, Pile
	import pandas

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

#======== Functions to help find a formula ======================================

def _getGroupedBy(state: EliminationState, pileTarget: Pile, groupByLeavesAtPiles: tuple[Pile, ...]) -> dict[Leaf | tuple[Leaf, ...], list[Leaf]]:
	from mapFolding.kitFilesystem import getDataFrameFoldings

	dataframeFoldings: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
	groupedBy: dict[Leaf | tuple[Leaf, ...], list[Leaf]] = dataframeFoldings.groupby(list(groupByLeavesAtPiles))[pileTarget].apply(list).to_dict()
	return {leaves: sorted(set(boxOfLeaves)) for leaves, boxOfLeaves in groupedBy.items()}

def getExcludedLeaves(state: EliminationState, pile: Pile, groupByLeavesAtPiles: tuple[Pile, ...]) -> dict[Leaf | tuple[Leaf, ...], list[Leaf]]:
	from mapFolding._e.pileOptions import getDictionaryChoicesLeaf
	return {leaves: sorted(filterfalse(boxOfLeaves.__contains__, (getIteratorOfLeaves(getDictionaryChoicesLeaf(state)[pile]))))
		for leaves, boxOfLeaves in DOTitems(_getGroupedBy(state, pile, groupByLeavesAtPiles))}

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
	pileDimension = bisect_left(state.mapShape首ProductsSums, pile >> 1 << 1)
	leafMinimum = isEven吗(pile) + state.mapShapeProducts[pileDimension]
	pileRange: list[Leaf] = []

	pileRange.append(leafMinimum)

	if isEven吗(pile):
		dd = pileDimension

		ss = state.mapShapeProductsSums[dd]
		pileRange.extend(map(iadd(leafMinimum - ss), state.mapShapeProductsSums[1:dd]))
		pileRange.extend(map(iadd(leafMinimum - ss), state.mapShapeProductsSums[dd + 1: state.dimensionsTotal]))

		if dd < dimensionNearest首(pile):
			dd += 1

			ss = state.mapShapeProducts[dd]
			# pileRange.extend(map(partial(isub, leafMinimum + ss), state.mapShapeProductsSums[1:dd]))
			# pileRange.extend(map(iadd(leafMinimum + ss), state.mapShapeProducts[dd + 1: state.dimensionsTotal]))

		if (pile % 4 == 0) and ((零) + 首零(state.dimensionsTotal) in pileRange):
			pileRange.remove((零) + 首零(state.dimensionsTotal))
			"""33 has step = 4"""

	if isOdd吗(pile):
		dd = pileDimension

		ss = state.mapShapeProductsSums[dd]
		pileRange.extend(map(iadd(leafMinimum + ss), state.mapShapeProducts[1:dd]))
		pileRange.extend(map(iadd(leafMinimum + ss), state.mapShapeProducts[dd + 1: state.dimensionsTotal]))

		dd += 1

		ss = state.mapShapeProductsSums[dd]
		pileRange.extend(map(iadd(leafMinimum + ss), state.mapShapeProducts[1:dd]))
		pileRange.extend(map(iadd(leafMinimum + ss), state.mapShapeProducts[dd + 1: state.dimensionsTotal]))

		dd += 1

		ss = state.mapShapeProductsSums[dd]
		pileRange.extend(map(iadd(leafMinimum + ss), state.mapShapeProducts[1:dd]))
		pileRange.extend(map(iadd(leafMinimum + ss), state.mapShapeProducts[dd + 1: state.dimensionsTotal]))

		dd += 1

		ss = state.mapShapeProductsSums[dd]
		pileRange.extend(map(iadd(leafMinimum + ss), state.mapShapeProducts[1:dd]))
		pileRange.extend(map(iadd(leafMinimum + ss), state.mapShapeProducts[dd + 1: state.dimensionsTotal]))

	print(f"{pile=}\t{pileDimension=}")
	print("computed=", sorted(set(pileRange)))
	realRange = tuple(getIteratorOfLeaves(getChoicesLeaf(state, pile)))
	print(f"{realRange=}")
	pileAnte = tuple(getIteratorOfLeaves(getChoicesLeaf(state, pile - 1)))
	print(f"{pileAnte=}")

	pileRangeByFormula: bool = False
	if pileRangeByFormula:
		state = EliminationState((2,) * 6)

		# DEVELOPMENT works for 9 <= odd piles <= 47
		# I _think_ I need to be able to pass start/stop to intraDimensionalLeaves
		# Yes, sort of. `Z0Z_alfaBeta` and `intraDimensionalLeaves` need to be the same function: and I need to be able to tweak all of the parameters.

		def intraDimensionalLeaves(state: EliminationState, dimensionOrigin: int) -> list[int]:
			return list(map(add(dimensionOrigin + 2), state.mapShapeProductsSums[1: dimensionNearest首(dimensionOrigin)]))

		def Z0Z_alfaBeta(state: EliminationState, alfaStart: int = 0, betaStop: int = 0, charlieStep: int = 1) -> list[int]:
			return list(flatten(map(partial(intraDimensionalLeaves, state), state.mapShapeProducts[2 + alfaStart: (state.dimensionsTotal - 1) + betaStop: charlieStep])))

		def Z0Z_getPileRange(state: EliminationState, pile: Pile) -> Iterable[Leaf]:
			pileRange: list[Leaf] = []

			# odd leaves < 32.
			# ? 12 < even leaves < 32.
			# ? 24 < even leaves < 32.
			# piles 49, 51, 53, 55 need a higher start on yy=0.
			for yy in range(3):
				pileRange.extend(map(mul(state.mapShapeProducts[yy]), Z0Z_alfaBeta(state, betaStop=-(yy))))

			# 32 < even leaves
			for yy in range(1):
				pileRange.extend(map(partial(invertLeafIn2上nDimensions, state.dimensionsTotal), map(mul(state.mapShapeProducts[yy])
					, Z0Z_alfaBeta(state
						, alfaStart=yy + (state.dimensionsTotal - 2 - dimensionNearest首(pile))
						, betaStop=-(yy)
					))))
			# ? 32 < odd leaves < 52
			# ? 32 < odd leaves < 36
			for yy in range(1, 3):
				pileRange.extend(map(partial(invertLeafIn2上nDimensions, state.dimensionsTotal), map(mul(state.mapShapeProducts[yy]), Z0Z_alfaBeta(state, betaStop=-(yy)))))

			# dimension origins
			# piles 51, 53, 55 need a higher start.
			pileRange.extend(state.mapShapeProducts[1 + ((零) + 首零(state.dimensionsTotal) < pile):dimensionNearest首(pile + 1)])
			# inverse dimension origins: 62, 61, 59, 55, 47, 31
			# pile5 needs a higher start.
			pileRange.extend(map(partial(invertLeafIn2上nDimensions, state.dimensionsTotal), state.mapShapeProducts[0:state.dimensionsTotal]))

			return tuple(sorted(pileRange))

		def Z0Z_getPileRangeEven(state: EliminationState, pile: Pile) -> Iterable[Leaf]:
			pileRange: list[Leaf] = []

			for yy in range(3):
				pileRange.extend(map(
					add(1)
					, (map(
						mul(state.mapShapeProducts[yy])
						, Z0Z_alfaBeta(state, alfaStart=0, betaStop=-(yy))
				)
			)
		)
	)

			# for yy in range(1):
			# 	pileRange.extend(map(partial(Z0Z_invert, state.dimensionsTotal), map(mul(state.mapShapeProducts[yy])
			# 		, Z0Z_alfaBeta(state
			# 			, alfaStart=yy+(state.dimensionsTotal - 2 - dimensionNearest首(pile))
			# 			, betaStop=-(yy)
			# 		))))
			# for yy in range(1,3):
			# 	pileRange.extend(map(partial(Z0Z_invert, state.dimensionsTotal), map(mul(state.mapShapeProducts[yy]), Z0Z_alfaBeta(state, betaStop=-(yy)))))

			# dimension origins
			pileRange.extend(map(add(1), state.mapShapeProducts[1 + ((零) + 首零(state.dimensionsTotal) < pile):dimensionNearest首(pile + 1)]))
			# inverse dimension origins: 62, 61, 59, 55, 47, 31
			pileRange.extend(map(partial(invertLeafIn2上nDimensions, state.dimensionsTotal), map(add(1), state.mapShapeProducts[1:state.dimensionsTotal])))

			return tuple(sorted(pileRange))

		for pile in range(首一(state.dimensionsTotal), 首零一(state.dimensionsTotal), 2):
			print(pile, (real := tuple(getIteratorOfLeaves(getChoicesLeaf(state, pile)))) == (computed := Z0Z_getPileRangeEven(state, pile)), end=': ')
			# print(f"{ansiColors.Green}surplus: {set(computed).difference(real)}", f"{ansiColors.Magenta}missing: {set(real).difference(computed)}{ansiColorReset}", sep='\n')
			pprint(f"{computed=}", width=180)

		for pile in range((零) + 首二(state.dimensionsTotal), 首零一(state.dimensionsTotal), 2):
			print(pile, (real := tuple(getIteratorOfLeaves(getChoicesLeaf(state, pile)))) == (computed := Z0Z_getPileRange(state, pile)), end=': ')
			# print(f"surplus: {set(computed).difference(real)}", f"missing: {set(real).difference(computed)}", sep='\n')
			pprint(f"{computed=}", width=180)

			# > 32: matches most tail0s != 1
			# if pile > 32:
			# 	pile-=1
			# else:
			# 	pile+=1
			# zz = tuple(map(partial(xor, 1), zz))
			# print(pile, (ll:=getChoicesLeaf(state, pile)) == (zz), end=': ')
			# # print(set(zz).difference(ll), set(ll).difference(zz), sep='\t')
			# pprint(zz, width=180)
