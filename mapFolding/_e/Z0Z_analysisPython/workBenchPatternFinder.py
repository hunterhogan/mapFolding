# ruff: noqa: ERA001 T201 T203  # noqa: RUF100
from collections.abc import Iterable
from cytoolz.functoolz import curry as syntacticCurry
from functools import partial
from hunterMakesPy import raiseIfNone
from mapFolding._e import (
	dimensionNearest首, getDictionaryLeafDomains, getDictionaryPileRanges, getLeavesCreaseBack, getLeavesCreaseNext,
	getPileRange, getSumsOfProductsOfDimensionsNearest首, leafInSubHyperplane, ptount, Z0Z_invert, 零, 首一, 首二, 首零, 首零一)
from mapFolding._e._dataDynamic import getDataFrameFoldings
from mapFolding._e.dataBaskets import EliminationState
from more_itertools import flatten
from operator import add, mul
from pprint import pprint
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import pandas

def _getGroupedBy(state: EliminationState, pileTarget: int, groupByLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	dataframeFoldings: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))
	groupedBy: dict[int | tuple[int, ...], list[int]] = dataframeFoldings.groupby(list(groupByLeavesAtPiles))[pileTarget].apply(list).to_dict() # pyright: ignore[reportAssignmentType]
	return {leaves: sorted(set(listLeaves)) for leaves, listLeaves in groupedBy.items()}

def getExcludedLeaves(state: EliminationState, pileTarget: int, groupByLeavesAtPiles: tuple[int, ...]) -> dict[int | tuple[int, ...], list[int]]:
	return {leaves: sorted(set(getDictionaryPileRanges(state)[pileTarget]).difference(set(listLeaves))) for leaves, listLeaves in _getGroupedBy(state, pileTarget, groupByLeavesAtPiles).items()}

if __name__ == '__main__':

	state = EliminationState((2,) * 6)
	pileExcluder = 3
	pileTarget=31
	dictionaryExcluded = getExcludedLeaves(state, pileTarget, groupByLeavesAtPiles=(pileExcluder,))
	domains = getDictionaryLeafDomains(state)
	pileRange31 = frozenset(getPileRange(state, 31))

	print(pileRange31)
	# for nn in {2, 49, 19, 2, 49, 21, 8, 25, 56}:
	# 	print(nn, nn^63)

	for pile in range(state.leavesTotal):
		continue
		print(pile, set(getPileRange(state, pile)).difference(getExcludedLeaves(state, pileTarget, groupByLeavesAtPiles=(pile,)).keys()))

	for excluder, listExcluded in dictionaryExcluded.items():
		continue

		invert = int(excluder^63) # pyright: ignore[reportUnknownArgumentType, reportOperatorIssue]
		creaseNextSS = tuple(getLeavesCreaseNext(state, invert)) # pyright: ignore[reportArgumentType]
		allCreaseNextSSInRange = set(creaseNextSS).intersection(pileRange31)
		creaseBack = tuple(getLeavesCreaseBack(state, excluder)) # pyright: ignore[reportArgumentType]
		creaseNext = tuple(getLeavesCreaseNext(state, excluder)) # pyright: ignore[reportArgumentType]
		allCreaseBackInRange = set(creaseBack).intersection(pileRange31)
		allCreaseNextInRange = set(creaseNext).intersection(pileRange31)
		notExcluded = allCreaseNextInRange.difference(listExcluded)
		# print(excluder, invert, allCreaseNextSSInRange.intersection(listExcluded), notExcluded, allCreaseNextSSInRange.difference(listExcluded), set(creaseNextSS).symmetric_difference(creaseNext), creaseNextSS, allCreaseNextSSInRange)
		# print(excluder.__format__('06b'), excluder, f"{notExcluded}\t", f"{creaseNext}", sep='\t')
		print(excluder, f"{allCreaseBackInRange=}", f"{allCreaseNextInRange=}", sep='\t')
		print(excluder, f"{allCreaseBackInRange.difference(listExcluded)}", f"{allCreaseNextInRange.difference(listExcluded)}", sep='\t')

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

		def Z0Z_getPileRange(state: EliminationState, pile: int) -> Iterable[int]:
			pileRange: list[int] = []

			# odd leaves < 32.
			# ? 12 < even leaves < 32.
			# ? 24 < even leaves < 32.
			# piles 49, 51, 53, 55 need a higher start on yy=0.
			for yy in range(3):
				pileRange.extend(map(partial(mul, state.productsOfDimensions[yy]), Z0Z_alphaBeta(state, betaStop=-(yy))))

			# 32 < even leaves
			for yy in range(1):
				pileRange.extend(map(partial(Z0Z_invert, state.dimensionsTotal), map(partial(mul, state.productsOfDimensions[yy])
					, Z0Z_alphaBeta(state
						, alphaStart=yy+(state.dimensionsTotal - 2 - dimensionNearest首(pile))
						, betaStop=-(yy)
					))))
			# ? 32 < odd leaves < 52
			# ? 32 < odd leaves < 36
			for yy in range(1,3):
				pileRange.extend(map(partial(Z0Z_invert, state.dimensionsTotal), map(partial(mul, state.productsOfDimensions[yy]), Z0Z_alphaBeta(state, betaStop=-(yy)))))

			# dimension origins
			# piles 51, 53, 55 need a higher start.
			pileRange.extend(state.productsOfDimensions[1 + (首零(state.dimensionsTotal)+零 < pile):dimensionNearest首(pile+1)])
			# inverse dimension origins: 62, 61, 59, 55, 47, 31
			# pile5 needs a higher start.
			pileRange.extend(map(partial(Z0Z_invert, state.dimensionsTotal), state.productsOfDimensions[0:state.dimensionsTotal]))

			return tuple(sorted(pileRange))

		def Z0Z_getPileRangeEven(state: EliminationState, pile: int) -> Iterable[int]:
			pileRange: list[int] = []

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
			pileRange.extend(map(partial(add, 1), state.productsOfDimensions[1 + (首零(state.dimensionsTotal)+零 < pile):dimensionNearest首(pile+1)]))
			# inverse dimension origins: 62, 61, 59, 55, 47, 31
			pileRange.extend(map(partial(Z0Z_invert, state.dimensionsTotal), map(partial(add, 1), state.productsOfDimensions[1:state.dimensionsTotal])))

			return tuple(sorted(pileRange))

		for pile in range(首一(state.dimensionsTotal), 首零一(state.dimensionsTotal), 2):
			print(pile, (real:=tuple(getPileRange(state, pile))) == (computed:=Z0Z_getPileRangeEven(state, pile)), end=': ')
			# print(f"{ansiColorGreen}surplus: {set(computed).difference(real)}", f"{ansiColorMagenta}missing: {set(real).difference(computed)}{ansiColorReset}", sep='\n')
			pprint(f"{computed=}", width=180)

		for pile in range(首二(state.dimensionsTotal)+零, 首零一(state.dimensionsTotal), 2):
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

