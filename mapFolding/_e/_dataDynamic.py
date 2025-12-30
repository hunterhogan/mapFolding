from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from cytoolz.functoolz import curry as syntacticCurry
from functools import cache
from gmpy2 import bit_flip, bit_mask, is_even, is_odd
from hunterMakesPy import raiseIfNone, writePython
from mapFolding import ansiColorReset, ansiColorYellowOnBlack, decreasing, inclusive, packageSettings
from mapFolding._e import (
	between, consecutive, dimensionFourthNearest首, dimensionNearestTail, dimensionNearest首, dimensionSecondNearest首,
	dimensionThirdNearest首, exclude, getSumsOfProductsOfDimensionsNearest首, howManyDimensionsHaveOddParity,
	leafInSubHyperplane, leafOrigin, mapShapeIs2上nDimensions, pileOrigin, reverseLookup, 一, 三, 二, 四, 零, 首一, 首一二, 首三, 首二,
	首零, 首零一, 首零一二, 首零二)
from mapFolding._e.dataBaskets import EliminationState
from more_itertools import all_unique, loops
from operator import add, sub
from pathlib import Path, PurePath
from typing import Any
import pandas
import sys

# ======= Creases =================================

def getLeavesCreaseBack(state: EliminationState, leaf: int) -> Iterator[int]:
	"""1) The leaf has at most `dimensionsTotal - 1` many creases.

	2) The list is ordered by increasing dimension number, which corresponds to an increasing absolute magnitude of _change_ in
		`leaf` number.

	3) The list of creases *might* be a list of Gray codes.
	"""
	return iter(_getCreases(state, leaf, increase=False))

def getLeavesCreaseNext(state: EliminationState, leaf: int) -> Iterator[int]:
	"""1) The leaf has at most `dimensionsTotal - 1` many creases.

	2) The list is ordered by increasing dimension number, which corresponds to an increasing absolute magnitude of _change_ in
		`leaf` number.

	3) The list of creases *might* be a list of Gray codes.
	"""
	return iter(_getCreases(state, leaf, increase=True))

def _getCreases(state: EliminationState, leaf: int, *, increase: bool = True) -> tuple[int, ...]:
	return _makeCreases(leaf, state.dimensionsTotal)[increase]
@cache
def _makeCreases(leaf: int, dimensionsTotal: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
	listLeavesCrease: list[int] = [int(bit_flip(leaf, dimensionIndex)) for dimensionIndex in range(dimensionsTotal)]

	if leaf == leafOrigin: # A special case I've been unable to figure out how to incorporate in the formula.
		listLeavesCreaseNext: list[int] = [1]
		listLeavesCreaseBack: list[int] = []
	else:
		slicingIndices: int = is_odd(howManyDimensionsHaveOddParity(leaf))

		slicerBack: slice = slice(slicingIndices, dimensionNearest首(leaf) * bit_flip(slicingIndices, 0) or None)
		slicerNext: slice = slice(bit_flip(slicingIndices, 0), dimensionNearest首(leaf) * slicingIndices or None)

		if is_even(leaf):
			if slicerBack.start == 1:
				slicerBack = slice(slicerBack.start + dimensionNearestTail(leaf), slicerBack.stop)
			if slicerNext.start == 1:
				slicerNext = slice(slicerNext.start + dimensionNearestTail(leaf), slicerNext.stop)
		listLeavesCreaseBack = listLeavesCrease[slicerBack]
		listLeavesCreaseNext = listLeavesCrease[slicerNext]

		if leaf == 1: # A special case I've been unable to figure out how to incorporate in the formula.
			listLeavesCreaseBack = [0]
	return (tuple(listLeavesCreaseBack), tuple(listLeavesCreaseNext))

# ======= (mathematical) ranges of piles ====================
# TODO Ideally, figure out the formula for pile ranges instead of deconstructing leaf domains.
# TODO Second best, DRYer code.

# ------- Boolean filters for (mathematical) ranges of piles -----------------------------------

@syntacticCurry
def filterCeiling(pile: int, dimensionsTotal: int, leaf: int) -> bool:
	return pile <  int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(leaf))) - howManyDimensionsHaveOddParity(leaf) + 2 - (leaf == leafOrigin)

@syntacticCurry
def filterFloor(pile: int, leaf: int) -> bool:
	return int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin) <= pile

@syntacticCurry
def filterParity(pile: int, leaf: int) -> bool:
	return (pile & 1) == ((int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin)) & 1)

@syntacticCurry
def filterDoubleParity(pile: int, dimensionsTotal: int, leaf: int) -> bool:
	if leaf != 首零(dimensionsTotal)+零:
		return True
	return (pile >> 1 & 1) == ((int(bit_flip(0, dimensionNearestTail(leaf) + 1)) + howManyDimensionsHaveOddParity(leaf) - 1 - (leaf == leafOrigin)) >> 1 & 1)

def getPileRange(state: EliminationState, pile: int) -> Iterator[int]:
	return iter(_getPileRange(pile, state.dimensionsTotal, state.mapShape, state.leavesTotal))
@cache
def _getPileRange(pile: int, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> tuple[int, ...]:
	if (dimensionsTotal > 3) and all(dimensionLength == 2 for dimensionLength in mapShape):
		parityMatch: Callable[[int], bool] = filterParity(pile)
		pileAboveFloor: Callable[[int], bool] = filterFloor(pile)
		pileBelowCeiling: Callable[[int], bool] = filterCeiling(pile, dimensionsTotal)
		matchLargerStep: Callable[[int], bool] = filterDoubleParity(pile, dimensionsTotal)

		pileRange: Iterable[int] = range(leavesTotal)
		pileRange = filter(parityMatch, pileRange)
		pileRange = filter(pileAboveFloor, pileRange)
		pileRange = filter(pileBelowCeiling, pileRange)
		return tuple(filter(matchLargerStep, pileRange))

	return tuple(range(leavesTotal))

def getDictionaryPileRanges(state: EliminationState) -> dict[int, tuple[int, ...]]:
	"""At `pile`, which `leaf` values may be found in a `folding`: the mathematical range, not a Python `range` object."""
	return {pile: tuple(getPileRange(state, pile)) for pile in range(state.leavesTotal)}

# ======= Leaf domains ====================================

def getLeafDomain(state: EliminationState, leaf: int) -> range:
	return _getLeafDomain(leaf, state.dimensionsTotal, state.mapShape, state.leavesTotal)
@cache
def _getLeafDomain(leaf: int, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> range:
	"""The subroutines assume `dimensionLength == 2`, but I think the concept could be extended to other `mapShape`."""
	state = EliminationState(mapShape)
	if mapShapeIs2上nDimensions(state.mapShape):
		originPinned =  leaf == leafOrigin
		return range(
					state.sumsOfProductsOfDimensions[dimensionNearestTail(leaf) + inclusive]	# `start`, first value included in the `range`.
						+ howManyDimensionsHaveOddParity(leaf)
						- originPinned

					, state.sumsOfProductsOfDimensionsNearest首[dimensionNearest首(leaf)]		# `stop`, first value excluded from the `range`.
						+ 2
						- howManyDimensionsHaveOddParity(leaf)
						- originPinned

					, 2 + (2 * (leaf == 首零(dimensionsTotal)+零))								# `step`
				)
	return range(leavesTotal)

"""leaf domains are directly tied to sumsOfProductsOfDimensions and sumsOfProductsOfDimensionsNearest首

2d6
(0, 32, 48, 56, 60, 62, 63) = sumsOfProductsOfDimensionsNearest首
(0, 1, 3, 7, 15, 31, 63, 127) = sumsOfProductsOfDimensions

leaf descends from 63 in sumsOfProductsOfDimensionsNearest首
first pile is dimensionsTotal and ascends by addends in sumsOfProductsOfDimensions

leaf63 starts at pile6 = 6+0
leaf62 starts at pile7 = 6+1
leaf60 starts at pile10 = 7+3
leaf56 starts at pile17 = 10+7
leaf48 starts at pile32 = 17+15
leaf32 starts at pile63 = 32+31

2d5
sumsOfProductsOfDimensionsNearest首
(0, 16, 24, 28, 30, 31)

31, 5+0
30, 5+1
28, 6+3
24, 9+7
16, 16+15

sumsOfProductsOfDimensions
(0, 1, 3, 7, 15, 31, 63)

{0: [0],
 1: [1],
 2: [3, 5, 9, 17],
 3: [2, 7, 11, 13, 19, 21, 25],
 4: [3, 5, 6, 9, 10, 15, 18, 23, 27, 29],
 5: [2, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31],
 6: [3, 5, 6, 9, 10, 15, 17, 18, 23, 27, 29, 30],
 7: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 31],
 8: [3, 5, 6, 9, 10, 12, 15, 18, 20, 23, 27, 29, 30],
 9: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 10: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 27, 29, 30],
 11: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 12: [3, 5, 6, 9, 10, 12, 15, 18, 20, 23, 27, 29, 30],
 13: [2, 4, 7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 14: [3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 27, 29, 30],
 15: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 16: [3, 5, 6, 9, 10, 12, 15, 18, 20, 23, 24, 27, 29, 30],
 17: [2, 4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 18: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30],
 19: [4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 20: [5, 6, 9, 10, 12, 15, 18, 20, 23, 24, 27, 29, 30],
 21: [4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 22: [5, 6, 9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30],
 23: [4, 7, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 24: [5, 6, 9, 10, 12, 15, 18, 20, 23, 24, 27, 29, 30],
 25: [4, 8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 26: [9, 10, 12, 15, 17, 18, 20, 23, 24, 27, 29, 30],
 27: [8, 11, 13, 14, 19, 21, 22, 25, 26, 28, 31],
 28: [9, 10, 12, 18, 20, 23, 24, 27, 29, 30],
 29: [8, 19, 21, 22, 25, 26, 28],
 30: [17, 18, 20, 24],
 31: [16]}
"""

"""products of dimensions and sums of products emerge from the formulas in `getLeafDomain`.
state = EliminationState((2,) * 6)
domainsOfDimensionOrigins = tuple(getLeafDomain(state, leaf) for leaf in state.productsOfDimensions)[0:-1]
sumsOfDimensionOrigins = tuple(accumulate(state.productsOfDimensions))[0:-1]
sumsOfDimensionOriginsReversed = tuple(accumulate(state.productsOfDimensions[::-1], initial=-state.leavesTotal))[1:None]
for dimensionOrigin, domain, sumOrigins, sumReversed in zip(state.productsOfDimensions, domainsOfDimensionOrigins, sumsOfDimensionOrigins, sumsOfDimensionOriginsReversed, strict=False):
	print(f"{dimensionOrigin:<2}\t{domain.start == sumOrigins = }\t{sumOrigins}\t{sumReversed+2}\t{domain.stop == sumReversed+2 = }")
1       domain.start == sumOrigins = True       1       2       domain.stop == sumReversed+2 = True
2       domain.start == sumOrigins = True       3       34      domain.stop == sumReversed+2 = True
4       domain.start == sumOrigins = True       7      50      domain.stop == sumReversed+2 = True
8       domain.start == sumOrigins = True       15      58      domain.stop == sumReversed+2 = True
16      domain.start == sumOrigins = True       31     62	      domain.stop == sumReversed+2 = True
32      domain.start == sumOrigins = True       63      64      domain.stop == sumReversed+2 = True

(Note to self: in `sumReversed+2`, consider if this is better explained by `sumReversed - descending + inclusive` or something similar.)

The piles of dimension origins (sums of products of dimensions) emerge from the following formulas!

(Note: the function below is included to capture the function as it existed at this point in development. I hope the package has improved/evolved by the time you read this.)
def getLeafDomain(state: EliminationState, leaf: int) -> range:
	def workhorse(leaf: int, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> range:
		originPinned =  leaf == leafOrigin
		return range(
					int(bit_flip(0, howMany0coordinatesAtTail(leaf) + 1))									# `start`, first value included in the `range`.
						+ howManyDimensionsHaveOddParity(leaf)
						- 1 - originPinned
					, int(bit_mask(dimensionsTotal) ^ bit_mask(dimensionsTotal - dimensionNearest首(leaf)))	# `stop`, first value excluded from the `range`.
						- howManyDimensionsHaveOddParity(leaf)
						+ 2 - originPinned
					, 2 + (2 * (leaf == 首零(dimensionsTotal)+零))											# `step`
				)
	return workhorse(leaf, state.dimensionsTotal, state.mapShape, state.leavesTotal)
"""

def getDomainDimension一(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
	"""The beans and cornbread and beans and cornbread dimension.

	(leaf一零, leaf一, leaf首一, leaf首零一)
	^^^ Can you see the symmetry? ^^^

	Accurate in at least six dimensions.
	"""
	domain一零: tuple[int, ...] = tuple(getLeafDomain(state, 一+零))
	domain首一: tuple[int, ...] = tuple(getLeafDomain(state, 首一(state.dimensionsTotal)))
	return _getDomainDimension一(domain一零, domain首一, state.dimensionsTotal)
@cache
def _getDomainDimension一(domain一零: tuple[int, ...], domain首一: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int, int, int], ...]:
	domainCombined: list[tuple[int, int, int, int]] = []

	for pileOfLeaf一零 in domain一零:
		domainOfLeaf首一: tuple[int, ...] = domain首一
		pilesTotal: int = len(domainOfLeaf首一)

		listIndicesPilesExcluded: list[int] = []

		if pileOfLeaf一零 <= 首二(dimensionsTotal):
			pass

		elif 首二(dimensionsTotal) < pileOfLeaf一零 < 首一(dimensionsTotal):
			listIndicesPilesExcluded.extend([*range(1, pilesTotal // 2), *range(1 + pilesTotal // 2, 3 * pilesTotal // 4)])

		elif pileOfLeaf一零 == 首一(dimensionsTotal):
			listIndicesPilesExcluded.extend([*range(1, pilesTotal // 2)])

		elif 首一(dimensionsTotal) < pileOfLeaf一零 < 首零(dimensionsTotal)-一:
			listIndicesPilesExcluded.extend([*range(3 * pilesTotal // 4)])

		elif pileOfLeaf一零 == 首零(dimensionsTotal)-一:
			listIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4)])

		elif pileOfLeaf一零 == 首零(dimensionsTotal):
			listIndicesPilesExcluded.extend([*range(2, pilesTotal // 2)])

		domainOfLeaf首一 = tuple(exclude(domainOfLeaf首一, listIndicesPilesExcluded))

		domainCombined.extend([(pileOfLeaf一零, pileOfLeaf一零 + 1, pileOfLeaf首一, pileOfLeaf首一 + 1) for pileOfLeaf首一 in domainOfLeaf首一])

	return tuple(filter(all_unique, domainCombined))

def getDomainDimension二(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
	"""(leaf二一, leaf二一零, leaf二零, leaf二)."""
	domain二零and二: tuple[tuple[int, int], ...] = getDomain二零and二(state)
	domain二一零and二一: tuple[tuple[int, int], ...] = getDomain二一零and二一(state)
	return _getDomainDimension二(domain二零and二, domain二一零and二一, state.dimensionsTotal)
@cache
def _getDomainDimension二(domain二零and二: tuple[tuple[int, int], ...], domain二一零and二一: tuple[tuple[int, int], ...], dimensionsTotal: int) -> tuple[tuple[int, int, int, int], ...]:
	domain0corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive, domain二零and二))
	domain一corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive, domain二一零and二一))
	pilesTotal: int = len(domain一corners)

	domainCombined: list[tuple[int, int, int, int]] = []

	productsOfDimensions: tuple[int, ...] = tuple(int(bit_flip(0, dimension)) for dimension in range(dimensionsTotal + 1))

# ======= By exclusion of the indices, add pairs of corners (160 tuples) ====================
	for index, (pileOfLeaf二一零, pileOfLeaf二一) in enumerate(domain一corners):
		listIndicesPilesExcluded: list[int] = []

		tailDimensions: int = dimensionNearestTail(pileOfLeaf二一)

# ------- `excludeBelow` `index` ---------------------------------
		excludeBelow: int = index
		listIndicesPilesExcluded.extend(range(excludeBelow))

# ------- `excludeAbove` `index` ---------------------------------
		excludeAbove: int = pilesTotal
		if pileOfLeaf二一 <= 首一(dimensionsTotal):
			if tailDimensions == 1:
				excludeAbove = pilesTotal // 2 + index
				if howManyDimensionsHaveOddParity(pileOfLeaf二一) == 2:
					excludeAbove -= 1

				if (howManyDimensionsHaveOddParity(pileOfLeaf二一) == 1 and (2 < dimensionNearest首(pileOfLeaf二一))):
					excludeAbove += 2

				if (howManyDimensionsHaveOddParity(pileOfLeaf二一) == 1
					and (dimensionNearest首(pileOfLeaf二一) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf二一)) < 2)
				):
					addend: int = productsOfDimensions[dimensionsTotal-2] + 4
					excludeAbove = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))

			else:
				excludeAbove = 3 * pilesTotal // 4 + 2
				if index == 0:
					excludeAbove = 1
				elif index <= 2:
					addend = 三 + sum(productsOfDimensions[1:dimensionsTotal-2])
					excludeAbove = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
		listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

# ------- Exclude "knock-out" indices ---------------------------------
		if pileOfLeaf二一 < 首一二(dimensionsTotal):
			if tailDimensions == 4:
				addend = int(bit_flip(0, tailDimensions))
				start: int = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
				listIndicesPilesExcluded.extend([*range(start, start + tailDimensions)])
			if tailDimensions == 3:
				addend = int(bit_flip(0, tailDimensions))
				start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
				listIndicesPilesExcluded.extend([*range(start, start + tailDimensions - 1)])
				start = domain0corners.index((pileOfLeaf二一 + addend * 2, pileOfLeaf二一零 + addend * 2))
				listIndicesPilesExcluded.extend([*range(start - 1, start + tailDimensions - 1)])
			if (tailDimensions < 3)	and (2 < dimensionNearest首(pileOfLeaf二一)):
				if 5 < dimensionsTotal:
					addend = 四
					start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
					stop: int = start + addend
					step = 2
					if (tailDimensions == 1) and (dimensionNearest首(pileOfLeaf二一) == 4):
						start += 2
						stop = start + 1
					if tailDimensions == 2:
						start += 3
						if dimensionNearest首(pileOfLeaf二一) == 4:
							start -= 2
						stop = start + tailDimensions + inclusive
					if howManyDimensionsHaveOddParity(pileOfLeaf二一) == 2:
						stop = start + 1
					listIndicesPilesExcluded.extend([*range(start, stop, step)])
				if (((dimensionNearest首(pileOfLeaf二一) == 3) and (howManyDimensionsHaveOddParity(pileOfLeaf二一) == 1))
					or (dimensionNearest首(pileOfLeaf二一) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf二一)) == 3)):
					addend = pileOfLeaf二一
					start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
					stop = start + 2
					if tailDimensions == 2:
						start += 1
						stop += 1
					if dimensionNearest首(pileOfLeaf二一) == 4:
						start += 3
						stop += 4
					step = 1
					listIndicesPilesExcluded.extend([*range(start, stop, step)])
			if dimensionNearest首(pileOfLeaf二一) == 2:
				addend = 三
				start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
				listIndicesPilesExcluded.extend([*range(start, start + addend, 2)])

		domainCombined.extend([(pileOfLeaf二一, pileOfLeaf二一零, pileOfLeaf二零, pileOfLeaf二) for pileOfLeaf二零, pileOfLeaf二 in exclude(domain0corners, listIndicesPilesExcluded)])

# ======= By inclusion of the piles, add non-corners (52 tuples) ====================
	domain一nonCorners: tuple[tuple[int, int], ...] = tuple(set(domain二一零and二一).difference(set(domain一corners)))
	domainCombined.extend([(pileOfLeaf一二, pileOfLeaf二一零, pileOfLeaf二一零 - 1, pileOfLeaf一二 + 1) for pileOfLeaf二一零, pileOfLeaf一二 in domain一nonCorners])

	return tuple(sorted(filter(all_unique, set(domainCombined))))

def getDomainDimension首二(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
	"""(leaf首二, leaf首零二, leaf首零一二, leaf首一二)."""
	domain首零二and首二: tuple[tuple[int, int], ...] = getDomain首零二and首二(state)
	domain首零一二and首一二: tuple[tuple[int, int], ...] = getDomain首零一二and首一二(state)
	return _getDomainDimension首二(state.dimensionsTotal, domain首零二and首二, domain首零一二and首一二)
@cache
def _getDomainDimension首二(dimensionsTotal: int, domain首零二and首二: tuple[tuple[int, int], ...], domain首零一二and首一二: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int, int, int], ...]:
	domain0corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive, domain首零二and首二))
	domain一corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive, domain首零一二and首一二))
	pilesTotal = len(domain一corners)

	domainCombined: list[tuple[int, int, int, int]] = []

# ======= By exclusion of the indices, add pairs of corners (160 tuples) ====================
	for index, (pileOfLeaf首零二, pileOfLeaf首二) in enumerate(domain0corners):
		listIndicesPilesExcluded: list[int] = []

		tailDimensions: int = dimensionNearestTail(pileOfLeaf首零二)

# ------- `excludeBelow` `index` ---------------------------------
		excludeBelow: int = index - 1
		listIndicesPilesExcluded.extend(range(excludeBelow))

# ------- `excludeAbove` `index` ---------------------------------
		excludeAbove: int = pilesTotal
		if tailDimensions == 1:
			excludeAbove = (pilesTotal - (int((pileOfLeaf首二) ^ bit_mask(dimensionsTotal)) // 4 - 1))

			if howManyDimensionsHaveOddParity(pileOfLeaf首二) == 3 and (dimensionsTotal - dimensionNearest首(pileOfLeaf首二) >= 2):
				excludeAbove += 2

			if (howManyDimensionsHaveOddParity(pileOfLeaf首二) == 1
				and (dimensionsTotal - dimensionNearest首(pileOfLeaf首二) >= 2)
				and (dimensionNearest首(pileOfLeaf首二) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf首二)) > 3)
			):
				excludeAbove += 2

			if (howManyDimensionsHaveOddParity(pileOfLeaf首二) == 1
				and (dimensionNearest首(pileOfLeaf首二) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf首二)) > 4)
			):
				excludeAbove += 2

			if ((howManyDimensionsHaveOddParity(pileOfLeaf首二) == dimensionsTotal - dimensionNearest首(pileOfLeaf首二))
				and (dimensionNearest首(pileOfLeaf首二) >= 4)
				and (howManyDimensionsHaveOddParity(pileOfLeaf首二) > 1)
			):
				excludeAbove -= 1

		else:
			if 首零二(dimensionsTotal) <= pileOfLeaf首零二:
				excludeAbove = pilesTotal - 1
			if 首零(dimensionsTotal) < pileOfLeaf首零二 < 首零二(dimensionsTotal):
				excludeAbove = pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 8 - 1)
			if 首一二(dimensionsTotal) < pileOfLeaf首零二 <= 首零(dimensionsTotal):
				excludeAbove = pilesTotal - int(bit_mask(dimensionsTotal - 4))

			if pileOfLeaf首零二 == 首一二(dimensionsTotal):
				excludeAbove = pilesTotal - int(bit_mask(dimensionsTotal - 4)) - 1
			if pileOfLeaf首零二 < 首一二(dimensionsTotal):
				excludeAbove = pilesTotal - int(bit_mask(dimensionsTotal - 3)) - (tailDimensions == 2)
		listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

# ------- Exclude "knock-out" indices ---------------------------------
		if tailDimensions == 1 and (abs(pileOfLeaf首零二 - 首零(dimensionsTotal)) == 2) and is_even(dimensionsTotal):
			listIndicesPilesExcluded.extend([excludeAbove - 2])
		if tailDimensions != 1 and 首一二(dimensionsTotal) <= pileOfLeaf首零二 <= 首零一(dimensionsTotal):
			if (tailDimensions == 2) and (howManyDimensionsHaveOddParity(pileOfLeaf首零二) + 1 != dimensionNearest首(pileOfLeaf首零二) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf首零二))):
				listIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 8 + 2)])
				if (pileOfLeaf首零二 <= 首零(dimensionsTotal)) and is_even(dimensionsTotal):
					listIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 4 - 1)])
			if tailDimensions == 3:
				listIndicesPilesExcluded.extend([excludeAbove - 2])
			if 3 < tailDimensions:
				listIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 4)])

		domainCombined.extend([(pileOfLeaf首二, pileOfLeaf首零二, pileOfLeaf首零一二, pileOfLeaf首一二) for pileOfLeaf首零一二, pileOfLeaf首一二 in exclude(domain一corners, listIndicesPilesExcluded)])

# ======= By inclusion of the piles, add non-corners (52 tuples) ====================
	domain0nonCorners: tuple[tuple[int, int], ...] = tuple(set(domain首零二and首二).difference(set(domain0corners)))
	domainCombined.extend([(pileOfLeaf首二, pileOfLeaf首零二, pileOfLeaf首零二 - 1, pileOfLeaf首二 + 1) for pileOfLeaf首零二, pileOfLeaf首二 in domain0nonCorners])

	return tuple(sorted(filter(all_unique, set(domainCombined))))

def getDomain二零and二(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf二零 and leaf二."""
	domain二零: tuple[int, ...] = tuple(getLeafDomain(state, 二+零))
	domain二: tuple[int, ...] = tuple(getLeafDomain(state, 二))
	return _getDomain二零and二(domain二零, domain二, state.dimensionsTotal)
@cache
def _getDomain二零and二(domain二零: tuple[int, ...], domain二: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
	domainCombined: list[tuple[int, int]] = []

	domain零: tuple[int, ...] = domain二零
	domain0: tuple[int, ...] = domain二

# ======= By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	direction: Callable[[Any, Any], Any] = add
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

# ======= By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for index, pileOfLeaf零 in enumerate(filter(between(pileOrigin, 首零(dimensionsTotal)-零), domain零)):
		listIndicesPilesExcluded: list[int] = []

		tailDimensions: int = dimensionNearestTail(pileOfLeaf零 - is_odd(pileOfLeaf零))

# ******* All differences between `_getDomain二零and二` and `_getDomain二零and二` *******
		excludeBelowAddend: int = 0
		steppingBasisForUnknownReasons: int = int(bit_mask(tailDimensions - 1).bit_flip(0))

		if pileOfLeaf零 == 二:
			listIndicesPilesExcluded.extend([*range(index + 1)])
		if pileOfLeaf零 == (首一(dimensionsTotal) + 首二(dimensionsTotal) + 首三(dimensionsTotal)):
			indexDomain0 = int(7 * pilesTotal / 8)
			indexDomain0 -= pilesFewerDomain0
			listIndicesPilesExcluded.extend([indexDomain0])

# ------- `excludeBelow` `index` ---------------------------------
		excludeBelow: int = index + excludeBelowAddend
		excludeBelow -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeBelow))

# ------- `excludeAbove` `index` ---------------------------------
		if pileOfLeaf零 <= 首一(dimensionsTotal):
			excludeAbove: int = index + (3 * pilesTotal // 4)
			excludeAbove -= pilesFewerDomain0
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
		if 首一(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal):
			excludeAbove = int(pileOfLeaf零 ^ bit_mask(dimensionsTotal)) // 2
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

# ------- Exclude by stepping: exclude ((2^tailDimensions - 1) / (2^tailDimensions))-many indices, e.g., 1/2, 3/4, 15/16, after `index` -----------------
		for dimension in range(tailDimensions):
			listIndicesPilesExcluded.extend(range(steppingBasisForUnknownReasons + int(bit_mask(dimension)), pilesTotal, int(bit_flip(0, dimension + 1))))

# ------- Exclude "knock-out" indices ---------------------------------
		if tailDimensions == 1:
			if (首二(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal)-零) and (2 < dimensionNearest首(pileOfLeaf零)):
				if dimensionSecondNearest首(pileOfLeaf零) == 零:
					indexDomain0: int = pilesTotal // 2
					indexDomain0 -= pilesFewerDomain0
					if 4 < domain0[indexDomain0].bit_length():
						listIndicesPilesExcluded.extend([indexDomain0])
					if 首一(dimensionsTotal) < pileOfLeaf零:
						indexDomain0 = -(pilesTotal // 4 - is_odd(pileOfLeaf零))
						indexDomain0 -= -(pilesFewerDomain0)
						listIndicesPilesExcluded.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					indexDomain0 = pilesTotal // 2 + 2
					indexDomain0 -= pilesFewerDomain0
					if domain0[indexDomain0] < 首零(dimensionsTotal):
						listIndicesPilesExcluded.extend([indexDomain0])
					indexDomain0 = -(pilesTotal // 4 - 2)
					indexDomain0 -= -(pilesFewerDomain0)
					if 首一(dimensionsTotal) < pileOfLeaf零:
						listIndicesPilesExcluded.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 一+零:
					indexDomain0 = -(pilesTotal // 4)
					indexDomain0 -= -(pilesFewerDomain0)
					listIndicesPilesExcluded.extend([indexDomain0])

				indexDomain0 = 3 * pilesTotal // 4
				indexDomain0 -= pilesFewerDomain0
				if pileOfLeaf零 < 首一二(dimensionsTotal):
					listIndicesPilesExcluded.extend([indexDomain0])

				if dimensionThirdNearest首(pileOfLeaf零) == 零:
					if dimensionSecondNearest首(pileOfLeaf零) == 一+零:
						listIndicesPilesExcluded.extend([indexDomain0 - 2])
					if dimensionNearest首(pileOfLeaf零) == 一+零:
						listIndicesPilesExcluded.extend([indexDomain0 - 2])

		elif 首一(dimensionsTotal) + 首三(dimensionsTotal) + is_odd(pileOfLeaf零) == pileOfLeaf零:
			indexDomain0 = (3 * pilesTotal // 4) - 1
			indexDomain0 -= pilesFewerDomain0
			listIndicesPilesExcluded.extend([indexDomain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, listIndicesPilesExcluded)])

	return tuple(sorted(set(domainCombined)))

def getDomain二一零and二一(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf二一零 and leaf二一."""
	domain二一零: tuple[int, ...] = tuple(getLeafDomain(state, 二+一+零))
	domain二一: tuple[int, ...] = tuple(getLeafDomain(state, 二+一))
	return _getDomain二一零and二一(domain二一零, domain二一, state.dimensionsTotal)
@cache
def _getDomain二一零and二一(domain二一零: tuple[int, ...], domain二一: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
	domainCombined: list[tuple[int, int]] = []

	domain零: tuple[int, ...] = domain二一零
	domain0: tuple[int, ...] = domain二一

# ======= By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	direction: Callable[[Any, Any], Any] = sub
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

# ======= By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for index, pileOfLeaf零 in enumerate(filter(between(pileOrigin, 首零(dimensionsTotal)-零), domain零)):
		listIndicesPilesExcluded: list[int] = []

		tailDimensions: int = dimensionNearestTail(pileOfLeaf零 - is_odd(pileOfLeaf零))

# ******* All differences between `_getDomain二零and二` and `_getDomain二一零and二一` *******
		excludeBelowAddend: int = int(is_even(index) or tailDimensions)
		steppingBasisForUnknownReasons: int = index

# ------- `excludeBelow` `index` ---------------------------------
		excludeBelow: int = index + excludeBelowAddend
		excludeBelow -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeBelow))

# ------- `excludeAbove` `index` ---------------------------------
		if pileOfLeaf零 <= 首一(dimensionsTotal):
			excludeAbove: int = index + (3 * pilesTotal // 4)
			excludeAbove -= pilesFewerDomain0
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))
		if 首一(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal):
			excludeAbove = int(pileOfLeaf零 ^ bit_mask(dimensionsTotal)) // 2
			listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

# ------- Exclude by stepping: exclude ((2^tailDimensions - 1) / (2^tailDimensions))-many indices, e.g., 1/2, 3/4, 15/16, after `index` -----------------
		for dimension in range(tailDimensions):
			listIndicesPilesExcluded.extend(range(steppingBasisForUnknownReasons + int(bit_mask(dimension)), pilesTotal, int(bit_flip(0, dimension + 1))))

# ------- Exclude "knock-out" indices ---------------------------------
		if tailDimensions == 1:
			if (首二(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal)-零) and (2 < dimensionNearest首(pileOfLeaf零)):
				if dimensionSecondNearest首(pileOfLeaf零) == 零:
					indexDomain0: int = pilesTotal // 2
					indexDomain0 -= pilesFewerDomain0
					if 4 < domain0[indexDomain0].bit_length():
						listIndicesPilesExcluded.extend([indexDomain0])
					if 首一(dimensionsTotal) < pileOfLeaf零:
						indexDomain0 = -(pilesTotal // 4 - is_odd(pileOfLeaf零))
						indexDomain0 -= -(pilesFewerDomain0)
						listIndicesPilesExcluded.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					indexDomain0 = pilesTotal // 2 + 2
					indexDomain0 -= pilesFewerDomain0
					if domain0[indexDomain0] < 首零(dimensionsTotal):
						listIndicesPilesExcluded.extend([indexDomain0])
					indexDomain0 = -(pilesTotal // 4 - 2)
					indexDomain0 -= -(pilesFewerDomain0)
					if 首一(dimensionsTotal) < pileOfLeaf零:
						listIndicesPilesExcluded.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 一+零:
					indexDomain0 = -(pilesTotal // 4)
					indexDomain0 -= -(pilesFewerDomain0)
					listIndicesPilesExcluded.extend([indexDomain0])

				indexDomain0 = 3 * pilesTotal // 4
				indexDomain0 -= pilesFewerDomain0
				if pileOfLeaf零 < 首一二(dimensionsTotal):
					listIndicesPilesExcluded.extend([indexDomain0])

				if dimensionThirdNearest首(pileOfLeaf零) == 零:
					if dimensionSecondNearest首(pileOfLeaf零) == 一+零:
						listIndicesPilesExcluded.extend([indexDomain0 - 2])
					if dimensionNearest首(pileOfLeaf零) == 一+零:
						listIndicesPilesExcluded.extend([indexDomain0 - 2])

		elif 首一(dimensionsTotal) + 首三(dimensionsTotal) + is_odd(pileOfLeaf零) == pileOfLeaf零:
			indexDomain0 = (3 * pilesTotal // 4) - 1
			indexDomain0 -= pilesFewerDomain0
			listIndicesPilesExcluded.extend([indexDomain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, listIndicesPilesExcluded)])

	return tuple(sorted(set(domainCombined)))

def getDomain首零二and首二(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf首零二 and leaf首二."""
	domain首零二: tuple[int, ...] = tuple(getLeafDomain(state, 首零二(state.dimensionsTotal)))
	domain首二: tuple[int, ...] = tuple(getLeafDomain(state, 首二(state.dimensionsTotal)))
	return _getDomain首零二and首二(domain首零二, domain首二, state.dimensionsTotal)
@cache
def _getDomain首零二and首二(domain首零二: tuple[int, ...], domain首二: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
	domainCombined: list[tuple[int, int]] = []

	domain零: tuple[int, ...] = domain首零二
	domain0: tuple[int, ...] = domain首二

# ======= By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	direction: Callable[[Any, Any], Any] = sub
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

# ======= By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for index, pileOfLeaf零 in enumerate(domain零):
		if pileOfLeaf零 < 首零(dimensionsTotal)+零:
			continue
		listIndicesPilesExcluded: list[int] = []

		tailDimensions: int = dimensionNearestTail(direction(pileOfLeaf零, is_odd(pileOfLeaf零)))

# ------- `excludeBelow` `index` ---------------------------------
		if 首零一(dimensionsTotal) < pileOfLeaf零:
			excludeBelow: int = index + 3 - (3 * pilesTotal // 4)
		else:
			excludeBelow = 2 + (首零一(dimensionsTotal) - direction(pileOfLeaf零, is_odd(pileOfLeaf零))) // 2
		excludeBelow -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeBelow))

# ------- `excludeAbove` `index` ---------------------------------
		excludeAbove: int = index + 2 - int(bit_mask(tailDimensions))
		excludeAbove -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

# ------- Exclude by stepping: exclude ((2^tailDimensions - 1) / (2^tailDimensions))-many indices, e.g., 1/2, 3/4, 15/16, after `index` -----------------
		countFromTheEnd: int = pilesTotal - 1
		countFromTheEnd -= pilesFewerDomain0
		steppingBasisForUnknownReasons: int = countFromTheEnd - int(bit_mask(tailDimensions - 1).bit_flip(0))
		for dimension in range(tailDimensions):
			listIndicesPilesExcluded.extend(range(steppingBasisForUnknownReasons - int(bit_mask(dimension)), decreasing, decreasing * int(bit_flip(0, dimension + 1))))

# ------- Exclude "knock-out" indices ---------------------------------
		if tailDimensions == 1:
			if (dimensionThirdNearest首(pileOfLeaf零) == 一) and (二+零 <= dimensionNearest首(pileOfLeaf零)):
				indexDomain0: int = (pilesTotal // 2) + 1
				indexDomain0 -= pilesFewerDomain0
				listIndicesPilesExcluded.extend([indexDomain0])
				indexDomain0 = (pilesTotal // 4) + 1
				indexDomain0 -= pilesFewerDomain0
				listIndicesPilesExcluded.extend([indexDomain0])
				if pileOfLeaf零 < 首零一(dimensionsTotal):
					listIndicesPilesExcluded.extend([indexDomain0 - 2])

			if howManyDimensionsHaveOddParity(pileOfLeaf零) == 一:
				indexDomain0 = (pilesTotal // 4) + 3
				indexDomain0 -= pilesFewerDomain0
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					listIndicesPilesExcluded.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 二:
					listIndicesPilesExcluded.extend([indexDomain0])
				if (((dimensionNearest首(pileOfLeaf零) == dimensionsTotal - 1) and (dimensionSecondNearest首(pileOfLeaf零) == dimensionsTotal - 3))
					or (dimensionSecondNearest首(pileOfLeaf零) == 二)):
					listIndicesPilesExcluded.extend([indexDomain0 - 2])
					indexDomain0 = (pilesTotal // 2) - 1
					indexDomain0 -= pilesFewerDomain0
					listIndicesPilesExcluded.extend([indexDomain0])

		elif (首零一(dimensionsTotal) - direction(首三(dimensionsTotal), is_odd(pileOfLeaf零))) == pileOfLeaf零:
			indexDomain0 = (pilesTotal // 4) + 2
			indexDomain0 -= pilesFewerDomain0
			listIndicesPilesExcluded.extend([indexDomain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, listIndicesPilesExcluded)])

	return tuple(sorted(set(domainCombined)))

def getDomain首零一二and首一二(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf首零一二 and leaf首一二."""
	domain首零一二: tuple[int, ...] = tuple(getLeafDomain(state, 首零一二(state.dimensionsTotal)))
	domain首一二: tuple[int, ...] = tuple(getLeafDomain(state, 首一二(state.dimensionsTotal)))
	return _getDomain首零一二and首一二(domain首零一二, domain首一二, state.dimensionsTotal)
@cache
def _getDomain首零一二and首一二(domain首零一二: tuple[int, ...], domain首一二: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
	domainCombined: list[tuple[int, int]] = []

	domain零: tuple[int, ...] = domain首零一二
	domain0: tuple[int, ...] = domain首一二

# ======= By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	direction: Callable[[Any, Any], Any] = add
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

# ======= By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for index, pileOfLeaf零 in enumerate(domain零):
		if pileOfLeaf零 < 首零(dimensionsTotal):
			continue
		listIndicesPilesExcluded: list[int] = []

		tailDimensions: int = dimensionNearestTail(direction(pileOfLeaf零, is_odd(pileOfLeaf零)))

# ------- `excludeBelow` `index` ---------------------------------
		if 首零一(dimensionsTotal) < pileOfLeaf零:
			excludeBelow: int = index + 1 - (3 * pilesTotal // 4)
		else:
			excludeBelow = (首零一(dimensionsTotal) - direction(pileOfLeaf零, is_odd(pileOfLeaf零))) // 2
		excludeBelow -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeBelow))

# ------- `excludeAbove` `index` ---------------------------------
		excludeAbove: int = index + 1 - int(bit_mask(tailDimensions))
		excludeAbove -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

# ------- Exclude by stepping: exclude ((2^tailDimensions - 1) / (2^tailDimensions))-many indices, e.g., 1/2, 3/4, 15/16, after `index` -----------------
		steppingBasisForUnknownReasons: int = index
		for dimension in range(tailDimensions):
			listIndicesPilesExcluded.extend(range(steppingBasisForUnknownReasons - int(bit_mask(dimension)), decreasing, decreasing * int(bit_flip(0, dimension + 1))))

# ------- Exclude "knock-out" indices ---------------------------------
		if tailDimensions == 1:
			if (dimensionThirdNearest首(pileOfLeaf零) == 一) and (二+零 <= dimensionNearest首(pileOfLeaf零)):
				indexDomain0: int = pilesTotal // 2
				indexDomain0 -= pilesFewerDomain0
				listIndicesPilesExcluded.extend([indexDomain0])
				indexDomain0 = pilesTotal // 4
				indexDomain0 -= pilesFewerDomain0
				listIndicesPilesExcluded.extend([indexDomain0])
				if pileOfLeaf零 < 首零一(dimensionsTotal):
					listIndicesPilesExcluded.extend([indexDomain0 - 2])
			if dimensionThirdNearest首(pileOfLeaf零) == 一+零:
				indexDomain0 = pilesTotal // 4
				indexDomain0 -= pilesFewerDomain0
				if dimensionFourthNearest首(pileOfLeaf零) == 一:
					listIndicesPilesExcluded.extend([indexDomain0])
			if howManyDimensionsHaveOddParity(pileOfLeaf零) == 一:
				indexDomain0 = (pilesTotal // 4) + 2
				indexDomain0 -= pilesFewerDomain0
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					listIndicesPilesExcluded.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 二:
					listIndicesPilesExcluded.extend([indexDomain0])
				if (首零二(dimensionsTotal) < pileOfLeaf零) and (二+零 <= dimensionNearest首(pileOfLeaf零)):
					listIndicesPilesExcluded.extend([indexDomain0 - 2])
					indexDomain0 = (pilesTotal // 2) - 2
					indexDomain0 -= pilesFewerDomain0
					listIndicesPilesExcluded.extend([indexDomain0])

		elif (首零一(dimensionsTotal) - direction(首三(dimensionsTotal), is_odd(pileOfLeaf零))) == pileOfLeaf零:
			indexDomain0 = (pilesTotal // 4) + 1
			indexDomain0 -= pilesFewerDomain0
			listIndicesPilesExcluded.extend([indexDomain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, listIndicesPilesExcluded)])

	return tuple(sorted(set(domainCombined)))

def getDomain首零Plus零Conditional(state: EliminationState) -> tuple[int, ...]:
	leaf: int = 首零(state.dimensionsTotal)+零
	domain首零Plus零: tuple[int, ...] = tuple(getLeafDomain(state, leaf))
	leaf首零一: int = 首零一(state.dimensionsTotal)
	pileOfLeaf一零: int = reverseLookup(state.leavesPinned, 一+零)
	pileOfLeaf首零一: int = reverseLookup(state.leavesPinned, leaf首零一)
	return _getDomain首零Plus零Conditional(domain首零Plus零, pileOfLeaf一零, pileOfLeaf首零一, state.dimensionsTotal, state.leavesTotal)
@cache
def _getDomain首零Plus零Conditional(domain首零Plus零: tuple[int, ...], pileOfLeaf一零: int, pileOfLeaf首零一: int, dimensionsTotal: int, leavesTotal: int) -> tuple[int, ...]:
	pilesTotal: int = 首一(dimensionsTotal)

	bump: int = 1 - int(pileOfLeaf一零.bit_count() == 1)
	howMany: int = dimensionsTotal - (pileOfLeaf一零.bit_length() + bump)
	onesInBinary = int(bit_mask(howMany))
	ImaPattern: int = pilesTotal - onesInBinary

	listIndicesPilesExcluded: list[int] = []
	if pileOfLeaf一零 == 二:
		listIndicesPilesExcluded.extend([零, 一, 二]) # These symbols make this pattern jump out.

	if 二 < pileOfLeaf一零 <= 首二(dimensionsTotal):
		stop: int = pilesTotal // 2 - 1
		listIndicesPilesExcluded.extend(range(1, stop))

		aDimensionPropertyNotFullyUnderstood = 5
		for _dimension in loops(dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
			start: int = 1 + stop
			stop += (stop+1) // 2
			listIndicesPilesExcluded.extend([*range(start, stop)])

		listIndicesPilesExcluded.extend([*range(1 + stop, ImaPattern)])

	if 首二(dimensionsTotal) < pileOfLeaf一零:
		listIndicesPilesExcluded.extend([*range(1, ImaPattern)])

	bump = 1 - int((leavesTotal - pileOfLeaf首零一).bit_count() == 1)
	howMany = dimensionsTotal - ((leavesTotal - pileOfLeaf首零一).bit_length() + bump)
	onesInBinary = int(bit_mask(howMany))
	ImaPattern = pilesTotal - onesInBinary

	aDimensionPropertyNotFullyUnderstood = 5

	if pileOfLeaf首零一 == leavesTotal-二:
		listIndicesPilesExcluded.extend([-零 -1, -(一) -1])
		if aDimensionPropertyNotFullyUnderstood <= dimensionsTotal:
			listIndicesPilesExcluded.extend([-二 -1])

	if ((首零一二(dimensionsTotal) < pileOfLeaf首零一 < leavesTotal-二)
		and (首二(dimensionsTotal) < pileOfLeaf一零 <= 首零(dimensionsTotal))):
		listIndicesPilesExcluded.extend([-1])

	if 首零一二(dimensionsTotal) <= pileOfLeaf首零一 < leavesTotal-二:
		stop: int = pilesTotal // 2 - 1
		listIndicesPilesExcluded.extend(range((1 + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing))

		for _dimension in loops(dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
			start: int = 1 + stop
			stop += (stop+1) // 2
			listIndicesPilesExcluded.extend([*range((start + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing)])

		listIndicesPilesExcluded.extend([*range((1 + stop + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])

		if 二 <= pileOfLeaf一零 <= 首零(dimensionsTotal):
			listIndicesPilesExcluded.extend([零, 一, 二, pilesTotal//2])

	if ((pileOfLeaf首零一 == 首零一二(dimensionsTotal))
		and (首一(dimensionsTotal) < pileOfLeaf一零 <= 首零(dimensionsTotal))):
		listIndicesPilesExcluded.extend([-1])

	if 首零一(dimensionsTotal) < pileOfLeaf首零一 < 首零一二(dimensionsTotal):
		if pileOfLeaf一零 in [首一(dimensionsTotal), 首零(dimensionsTotal)]:
			listIndicesPilesExcluded.extend([-1])
		elif 二 < pileOfLeaf一零 < 首二(dimensionsTotal):
			listIndicesPilesExcluded.extend([0])

	if pileOfLeaf首零一 < 首零一二(dimensionsTotal):
		listIndicesPilesExcluded.extend([*range((1 + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])

	pileOfLeaf一零ARCHETYPICAL: int = 首一(dimensionsTotal)
	bump = 1 - int(pileOfLeaf一零ARCHETYPICAL.bit_count() == 1)
	howMany = dimensionsTotal - (pileOfLeaf一零ARCHETYPICAL.bit_length() + bump)
	onesInBinary = int(bit_mask(howMany))
	ImaPattern = pilesTotal - onesInBinary

	if pileOfLeaf首零一 == leavesTotal-二:
		if pileOfLeaf一零 == 二:
			listIndicesPilesExcluded.extend([零, 一, 二, pilesTotal//2 -1, pilesTotal//2])
		if 二 < pileOfLeaf一零 <= 首零(dimensionsTotal):
			IDK = ImaPattern - 1
			listIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4), *range(1 + 3 * pilesTotal // 4, IDK)])
		if 首一(dimensionsTotal) < pileOfLeaf一零 <= 首零(dimensionsTotal):
			listIndicesPilesExcluded.extend([-1])

	if pileOfLeaf首零一 == 首零一(dimensionsTotal):
		if pileOfLeaf一零 == 首零(dimensionsTotal):
			listIndicesPilesExcluded.extend([-1])
		elif (二 < pileOfLeaf一零 < 首二(dimensionsTotal)) or (首二(dimensionsTotal) < pileOfLeaf一零 < 首一(dimensionsTotal)):
			listIndicesPilesExcluded.extend([0])

	return tuple(exclude(domain首零Plus零, listIndicesPilesExcluded))

def getDictionaryLeafDomains(state: EliminationState) -> dict[int, range]:
	"""For each `leaf`, the associated Python `range` defines the mathematical domain:
	1. every `pile` at which `leaf` may be found in a `folding` and
	2. in the set of all valid foldings, every `pile` at which `leaf` must be found.
	"""  # noqa: D205
	return {leaf: getLeafDomain(state, leaf) for leaf in range(state.leavesTotal)}

# ======= Specialized tools ===============================

def getDataFrameFoldings(state: EliminationState) -> pandas.DataFrame | None:
	pathFilename = Path(f'{packageSettings.pathPackage}/tests/dataSamples/arrayFoldingsP2d{state.dimensionsTotal}.pkl')
	if pathFilename.exists():
		dataframeFoldings = pandas.DataFrame(pandas.read_pickle(pathFilename))  # noqa: S301
	else:
		message: str = f"{ansiColorYellowOnBlack}I received {state.dimensionsTotal = }, but I could not find the data at:\n\t{pathFilename!r}.{ansiColorReset}"
		sys.stderr.write(message + '\n')
		dataframeFoldings = None
	return dataframeFoldings

def makeVerificationDataLeavesDomain(listDimensions: Sequence[int], listLeaves: Sequence[int | Callable[[int], int]], pathFilename: PurePath | None = None, settings: dict[str, dict[str, Any]] | None = None) -> PurePath:
	"""Create a Python module containing combined domain data for multiple leaves across multiple mapShapes.

	This function extracts the actual combined domain (the set of valid pile position tuples) for a group of leaves from pickled
	folding data. The data is used for verification in pytest tests comparing computed domains against empirical data.

	The combined domain is a set of tuples where each tuple represents the pile positions for the specified leaves in a valid
	folding. For example, if `listLeaves` is `[4, 5, 6, 7]`, each tuple has 4 elements representing the pile where each of those
	leaves appears in a folding.

	Parameters
	----------
	listDimensions : Sequence[int]
		The dimension counts to process (e.g., `[4, 5, 6]` for 2^4, 2^5, 2^6 leaf maps).
	listLeaves : Sequence[int | Callable[[int], int]]
		The leaves whose combined domain to extract. Elements can be:
		- Integers for absolute leaf indices (e.g., `4`, `5`, `6`, `7`)
		- Callables that take `dimensionsTotal` and return a leaf index (e.g., `首二`, `首零二`)
	pathFilename : PurePath | None = None
		The output file path. If `None`, defaults to `tests/dataSamples/p2DnDomain{leafNames}.py`.
	settings : dict[str, dict[str, Any]] | None = None
		Settings for `writePython` formatter. If `None`, uses defaults.

	Returns
	-------
	pathFilename : PurePath
		The path where the module was written.

	"""
	def resolveLeaf(leafSpec: int | Callable[[int], int], dimensionsTotal: int) -> int:
		return leafSpec(dimensionsTotal) if callable(leafSpec) else leafSpec

	def getLeafName(leafSpec: int | Callable[[int], int]) -> str:
		return leafSpec.__name__ if callable(leafSpec) else str(leafSpec)

	listLeafNames: list[str] = [getLeafName(leafSpec) for leafSpec in listLeaves]
	filenameLeafPart: str = '_'.join(listLeafNames)

	if pathFilename is None:
		pathFilename = Path(f"{packageSettings.pathPackage}/tests/dataSamples/p2DnDomain{filenameLeafPart}.py")
	else:
		pathFilename = Path(pathFilename)

	dictionaryDomainsByDimensions: dict[int, list[tuple[int, ...]]] = {}

	for dimensionsTotal in listDimensions:
		mapShape: tuple[int, ...] = (2,) * dimensionsTotal
		state: EliminationState = EliminationState(mapShape)
		dataframeFoldings: pandas.DataFrame = raiseIfNone(getDataFrameFoldings(state))

		listResolvedLeaves: list[int] = [resolveLeaf(leafSpec, dimensionsTotal) for leafSpec in listLeaves]

		listCombinedTuples: list[tuple[int, ...]] = []
		for indexRow in range(len(dataframeFoldings)):
			rowFolding: pandas.Series = dataframeFoldings.iloc[indexRow]
			tuplePiles: tuple[int, ...] = tuple(int(rowFolding[rowFolding == leaf].index[0]) for leaf in listResolvedLeaves)
			listCombinedTuples.append(tuplePiles)

		listUniqueTuples: list[tuple[int, ...]] = sorted(set(listCombinedTuples))
		dictionaryDomainsByDimensions[dimensionsTotal] = listUniqueTuples

	listPythonSource: list[str] = [
		'"""Verification data for combined leaf domains.',
		'',
		'This module contains empirically extracted combined domain data for leaves',
		f'{listLeafNames} across multiple mapShape configurations.',
		'',
		'Each list is named `listDomain2D{dimensionsTotal}` where `dimensionsTotal`',
		'is the exponent in the 2^dimensionsTotal mapShape, and it contains tuples representing',
		'valid pile positions for the specified leaves. The tuple order follows the original',
		'leaf argument order.',
		'"""',
		'',
	]

	for dimensionsTotal in sorted(dictionaryDomainsByDimensions):
		variableName: str = f"listDomain2D{dimensionsTotal}"
		listPythonSource.append(f'{variableName}: list[tuple[int, ...]] = {dictionaryDomainsByDimensions[dimensionsTotal]!r}')
		listPythonSource.append('')

	pythonSource: str = '\n'.join(listPythonSource)
	writePython(pythonSource, pathFilename, settings)

	return pathFilename

# ======= In development ========================
# ruff: noqa: SIM102
# I am developing in this module because of Python's effing namespace and "circular import" issues.

def getZ0Z_precedence(state: EliminationState) -> dict[int, dict[int, list[int]]]:
	"""leaf: pile: [conditional `leafPredecessor`]."""
	return _getZ0Z_precedence(state.mapShape)
@cache
def _getZ0Z_precedence(mapShape: tuple[int, ...]) -> dict[int, dict[int, list[int]]]:
	"""Prototype.

	Some leaves are always preceded by one or more leaves. Most leaves, however, are preceded by one or more other leaves only if
	the leaf is in a specific pile.
	"""
	state = EliminationState(mapShape)
	dictionaryDomains: dict[int, range] = getDictionaryLeafDomains(state)

	dictionaryPrecedence: dict[int, dict[int, list[int]]] = {}

# ======= piles at the beginning of the leaf's domain ================
	for dimension in range(3, state.dimensionsTotal + inclusive):
		for countDown in range(dimension - 2 + decreasing, decreasing, decreasing):
			for leaf in range(state.productsOfDimensions[dimension] - sum(state.productsOfDimensions[countDown:dimension - 2]), state.leavesTotal, state.productsOfDimensions[dimension - 1]):
				dictionaryPrecedence[leaf] = {aPile: [state.productsOfDimensions[dimensionNearest首(leaf)] + state.productsOfDimensions[dimensionNearestTail(leaf)]]
							for aPile in list(dictionaryDomains[leaf])[0: getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions, dimensionFrom首=dimension - 1)[dimension - 2 - countDown] // 2]}

# ------- The beginning of domain首一Plus零 --------------------------------
	leaf = 首一(state.dimensionsTotal)+零
	dictionaryPrecedence[leaf] = {aPile: [2 * state.productsOfDimensions[dimensionNearest首(leaf)] + state.productsOfDimensions[dimensionNearestTail(leaf)]
									, 3 * state.productsOfDimensions[dimensionNearest首(leaf)] + state.productsOfDimensions[dimensionNearestTail(leaf)]]
							for aPile in list(dictionaryDomains[leaf])[1:2]}
	del leaf

# ======= leaf首零一Plus零: conditional `leafPredecessor` in all piles of its domain ===========
	leaf: int = 首零一(state.dimensionsTotal)+零
	listOfPiles = list(dictionaryDomains[leaf])
	dictionaryPrecedence[leaf] = {aPile: [] for aPile in list(dictionaryDomains[leaf])}
	sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions)
	sumsOfProductsOfDimensionsNearest首InSubHyperplane: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions, dimensionFrom首=state.dimensionsTotal - 1)
	pileStepAbsolute = 2

	for aPile in listOfPiles[listOfPiles.index(一+零): listOfPiles.index(首零(state.dimensionsTotal)-零) + inclusive]:
		dictionaryPrecedence[leaf][aPile].append(首零(state.dimensionsTotal)+零)

	for indexUniversal in range(state.dimensionsTotal - 2):
		leafPredecessorTheFirst: int = state.sumsOfProductsOfDimensions[indexUniversal + 2]
		leavesPredecessorInThisSeries: int = state.productsOfDimensions[howManyDimensionsHaveOddParity(leafPredecessorTheFirst)]
		for addend in range(leavesPredecessorInThisSeries):
			leafPredecessor = leafPredecessorTheFirst + (addend * decreasing)
			pileFirst: int = (
				sumsOfProductsOfDimensionsNearest首[indexUniversal]
				+ state.sumsOfProductsOfDimensions[2]
				+ state.productsOfDimensions[state.dimensionsTotal - (indexUniversal + 2)]
				- ((pileStepAbsolute * 2 * (howManyDimensionsHaveOddParity(leafPredecessor) - 1 + is_even(leafPredecessor)))
					* (1 + (2 == (howManyDimensionsHaveOddParity(leafPredecessor) + is_even(leafPredecessor)) == dimensionNearest首(leafPredecessor)))
				)
			)
			for aPile in listOfPiles[listOfPiles.index(pileFirst): None]:
				dictionaryPrecedence[leaf][aPile].append(leafPredecessor)

			leafPredecessor首零: int = leafPredecessor + 首零(state.dimensionsTotal)
			if (leafInSubHyperplane(leafPredecessor) == 0) and is_odd(dimensionNearestTail(leafPredecessor)):
				dictionaryPrecedence[leaf][pileFirst].append(leafPredecessor首零)
			if leafPredecessor首零 == leaf:
				continue
			pileFirst = listOfPiles[-1] - (
					pileStepAbsolute * (
					howManyDimensionsHaveOddParity(leafPredecessor首零)
					- 1
					+ is_even(leafPredecessor首零)
					- is_odd(leafPredecessor首零)
					- int(dimensionNearestTail(leafPredecessor首零) == state.dimensionsTotal - 2)
					- int(leaf < leafPredecessor首零)
				))
			for aPile in listOfPiles[listOfPiles.index(pileFirst): None]:
				dictionaryPrecedence[leaf][aPile].append(leafPredecessor首零)

			if indexUniversal < state.dimensionsTotal - 4:
				if is_odd(dimensionNearestTail(leafPredecessor - is_odd(leafPredecessor))):
					pileFirst = (
						sumsOfProductsOfDimensionsNearest首InSubHyperplane[indexUniversal]
						+ state.sumsOfProductsOfDimensions[2 + 1 + indexUniversal]
						- (pileStepAbsolute
							* 2
							* (howManyDimensionsHaveOddParity(leafPredecessor首零) - 1
								+ is_even(leafPredecessor首零) * indexUniversal
								- is_even(leafPredecessor首零) * (int(not(bool(indexUniversal))))
							)
						)
						+ state.productsOfDimensions[state.dimensionsTotal - 1
													+ addend * (int(not(bool(indexUniversal))))
													- (indexUniversal + 2)]
					)
					for aPile in listOfPiles[listOfPiles.index(pileFirst) + indexUniversal: listOfPiles.index(首零(state.dimensionsTotal)-零) - indexUniversal + inclusive]:
						dictionaryPrecedence[leaf][aPile].append(leafPredecessor首零)

	del leaf, listOfPiles, sumsOfProductsOfDimensionsNearest首, pileStepAbsolute, sumsOfProductsOfDimensionsNearest首InSubHyperplane

# ======= leaf首零Plus零: Separate logic because the distance between absolute piles is 4, not 2 ==============
# leaf has conditional `leafPredecessor` in all but the first pile of its domain
# Reminder: has UNconditional `leafPredecessor` in the first pile: leaf零
	leaf: int = 首零(state.dimensionsTotal)+零
	listOfPiles: list[int] = list(dictionaryDomains[leaf])[1: None]
	dictionaryPrecedence[leaf] = {aPile: [] for aPile in listOfPiles}
	sumsOfProductsOfDimensionsNearest首: tuple[int, ...] = getSumsOfProductsOfDimensionsNearest首(state.productsOfDimensions)
	pileStepAbsolute = 4
	for indexUniversal in range(state.dimensionsTotal - 2):
		leafPredecessorTheFirst: int = state.sumsOfProductsOfDimensions[indexUniversal + 2]
		leavesPredecessorInThisSeries = state.productsOfDimensions[howManyDimensionsHaveOddParity(leafPredecessorTheFirst)]
		for addend in range(leavesPredecessorInThisSeries):
			leafPredecessor: int = leafPredecessorTheFirst + (addend * decreasing)
			leafPredecessor首零: int = leafPredecessor + 首零(state.dimensionsTotal)
			pileFirst = sumsOfProductsOfDimensionsNearest首[indexUniversal] + 6 - (pileStepAbsolute * (howManyDimensionsHaveOddParity(leafPredecessor) - 1 + is_even(leafPredecessor)))
			for aPile in listOfPiles[listOfPiles.index(pileFirst): None]:
				dictionaryPrecedence[leaf][aPile].append(leafPredecessor)
				dictionaryPrecedence[leaf][aPile].append(leafPredecessor首零)

	del leaf, listOfPiles, sumsOfProductsOfDimensionsNearest首, pileStepAbsolute

# ======= piles at the end of the leaf's domain ================
# ------- Example of special case: has conditional `leafPredecessor` two steps before the end of the domain --------------------------
	if state.dimensionsTotal == 6:
		leaf = 22
		sliceOfPiles = slice(0, None)
		listOfPiles = list(dictionaryDomains[leaf])[sliceOfPiles]
		leafPredecessorPileFirstPileLast = [(15, 43, 43)]
		for leafPredecessor, pileFirst, pileLast in leafPredecessorPileFirstPileLast:
			for pile in listOfPiles[listOfPiles.index(pileFirst): listOfPiles.index(pileLast) + inclusive]:
				dictionaryPrecedence[leaf].setdefault(pile, []).append(leafPredecessor)

# NOTE Some leaves, such as 16,48, have `leafPredecessor`, such as leaves 40 and 56, with a larger step size.
# NOTE There may be "knock-out" leaves, such as within the domain functions, above. Or I might have to find complex formulas, such
# as in `pinPile二Crease`. Or, more likely, "knock-out" leaves might be complex formulas that I have not yet discovered.

	return dictionaryPrecedence

def getZ0Z_successor(state: EliminationState) -> dict[int, dict[int, list[int]]]:
	"""leaf: pile: [conditional `leafSuccessor`]."""
	return _getZ0Z_successor(state.mapShape)
@cache
def _getZ0Z_successor(mapShape: tuple[int, ...]) -> dict[int, dict[int, list[int]]]:
	state = EliminationState(mapShape)
	dictionaryDomains: dict[int, range] = getDictionaryLeafDomains(state)

	dictionarySuccessor: dict[int, dict[int, list[int]]] = {}

	dictionaryPrecedence: dict[int, dict[int, list[int]]] = getZ0Z_precedence(state)

	for leafLater, dictionaryPiles in dictionaryPrecedence.items():
		tupleDomainLater: tuple[int, ...] = tuple(dictionaryDomains[leafLater])
		dictionaryPilesByPredecessor: defaultdict[int, set[int]] = defaultdict(set)
		for pileLater, listLeafPredecessors in dictionaryPiles.items():
			for leafEarlier in listLeafPredecessors:
				dictionaryPilesByPredecessor[leafEarlier].add(pileLater)

		for leafEarlier, setPilesRequiring in dictionaryPilesByPredecessor.items():
			tupleDomainEarlier: tuple[int, ...] = tuple(dictionaryDomains[leafEarlier])
			listOptionalPiles: list[int] = sorted(pile for pile in tupleDomainLater if pile not in setPilesRequiring)
			for pileEarlier in tupleDomainEarlier:
				optionalLessEqualCount: int = bisect_right(listOptionalPiles, pileEarlier)
				if optionalLessEqualCount == 0:
					listSuccessors: list[int] = dictionarySuccessor.setdefault(leafEarlier, {}).setdefault(pileEarlier, [])
					if leafLater not in listSuccessors:
						listSuccessors.append(leafLater)

	return dictionarySuccessor

