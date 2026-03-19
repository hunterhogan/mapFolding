from functools import cache
from gmpy2 import bit_flip, bit_mask, is_even, is_odd
from hunterMakesPy import CallableFunction, decreasing, inclusive, raiseIfNone
from mapFolding._e import (
	dimensionFourthNearest首, dimensionIndex, dimensionNearestTail, dimensionNearest首, dimensionSecondNearest首, dimensionThirdNearest首,
	howManyDimensionsHaveOddParity, Leaf, leafOrigin, mapShapeIs2上nDimensions, Pile, pileOrigin, reverseLookup, 一, 三, 二, 四, 零, 首一, 首一二, 首三, 首二,
	首零, 首零一, 首零一二, 首零二)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.filters import between吗, consecutive吗, exclude, leafIsPinned
from more_itertools import all_unique, loops
from operator import add, sub

def getLeafDomain(state: EliminationState, leaf: Leaf) -> range:
	return _getLeafDomain(leaf, state.dimensionsTotal, state.mapShape, state.leavesTotal)
@cache
def _getLeafDomain(leaf: Leaf, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> range:
	"""The subroutines assume `dimensionLength == 2`, but I think the concept could be extended to other `mapShape`."""
	state: EliminationState = EliminationState(mapShape)
	if mapShapeIs2上nDimensions(state.mapShape):
		originPinned: bool = leaf == leafOrigin
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
	domain0corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive吗, domain二零and二))
	domain一corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive吗, domain二一零and二一))
	pilesTotal: int = len(domain一corners)

	domainCombined: list[tuple[int, int, int, int]] = []

	productsOfDimensions: tuple[int, ...] = tuple(int(bit_flip(0, dimension)) for dimension in range(dimensionsTotal + 1))

#======== By exclusion of the indices, add pairs of corners (160 tuples) ====================
	for index, (pileOfLeaf二一零, pileOfLeaf二一) in enumerate(domain一corners):
		listIndicesPilesExcluded: list[int] = []

		dimensionTail: int = dimensionNearestTail(pileOfLeaf二一)

#-------- `excludeBelow` `index` ---------------------------------
		excludeBelow: int = index
		listIndicesPilesExcluded.extend(range(excludeBelow))

#-------- `excludeAbove` `index` ---------------------------------
		excludeAbove: int = pilesTotal
		if pileOfLeaf二一 <= 首一(dimensionsTotal):
			if dimensionTail == 1:
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

#-------- Exclude "knock-out" indices ---------------------------------
		if pileOfLeaf二一 < 首一二(dimensionsTotal):
			if dimensionTail == 4:
				addend = int(bit_flip(0, dimensionTail))
				start: int = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
				listIndicesPilesExcluded.extend([*range(start, start + dimensionTail)])
			if dimensionTail == 3:
				addend = int(bit_flip(0, dimensionTail))
				start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
				listIndicesPilesExcluded.extend([*range(start, start + dimensionTail - 1)])
				start = domain0corners.index((pileOfLeaf二一 + addend * 2, pileOfLeaf二一零 + addend * 2))
				listIndicesPilesExcluded.extend([*range(start - 1, start + dimensionTail - 1)])
			if (dimensionTail < 3)	and (2 < dimensionNearest首(pileOfLeaf二一)):
				if 5 < dimensionsTotal:
					addend = 四
					start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
					stop: int = start + addend
					step: int = 2
					if (dimensionTail == 1) and (dimensionNearest首(pileOfLeaf二一) == 4):
						start += 2
						stop = start + 1
					if dimensionTail == 2:
						start += 3
						if dimensionNearest首(pileOfLeaf二一) == 4:
							start -= 2
						stop = start + dimensionTail + inclusive
					if howManyDimensionsHaveOddParity(pileOfLeaf二一) == 2:
						stop = start + 1
					listIndicesPilesExcluded.extend([*range(start, stop, step)])
				if (((dimensionNearest首(pileOfLeaf二一) == 3) and (howManyDimensionsHaveOddParity(pileOfLeaf二一) == 1))
					or (dimensionNearest首(pileOfLeaf二一) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf二一)) == 3)):
					addend = pileOfLeaf二一
					start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
					stop = start + 2
					if dimensionTail == 2:
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

#======== By inclusion of the piles, add non-corners (52 tuples) ====================
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
	domain0corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive吗, domain首零二and首二))
	domain一corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive吗, domain首零一二and首一二))
	pilesTotal: Leaf = len(domain一corners)

	domainCombined: list[tuple[int, int, int, int]] = []

#======== By exclusion of the indices, add pairs of corners (160 tuples) ====================
	for index, (pileOfLeaf首零二, pileOfLeaf首二) in enumerate(domain0corners):
		listIndicesPilesExcluded: list[int] = []

		dimensionTail: int = dimensionNearestTail(pileOfLeaf首零二)

#-------- `excludeBelow` `index` ---------------------------------
		excludeBelow: int = index - 1
		listIndicesPilesExcluded.extend(range(excludeBelow))

#-------- `excludeAbove` `index` ---------------------------------
		excludeAbove: int = pilesTotal
		if dimensionTail == 1:
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
				and (4 <= dimensionNearest首(pileOfLeaf首二))
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
				excludeAbove = pilesTotal - int(bit_mask(dimensionsTotal - 3)) - (dimensionTail == 2)
		listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

#-------- Exclude "knock-out" indices ---------------------------------
		if dimensionTail == 1 and (abs(pileOfLeaf首零二 - 首零(dimensionsTotal)) == 2) and is_even(dimensionsTotal):
			listIndicesPilesExcluded.extend([excludeAbove - 2])
		if dimensionTail != 1 and 首一二(dimensionsTotal) <= pileOfLeaf首零二 <= 首零一(dimensionsTotal):
			if (dimensionTail == 2) and (howManyDimensionsHaveOddParity(pileOfLeaf首零二) + 1 != dimensionNearest首(pileOfLeaf首零二) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf首零二))):
				listIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 8 + 2)])
				if (pileOfLeaf首零二 <= 首零(dimensionsTotal)) and is_even(dimensionsTotal):
					listIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 4 - 1)])
			if dimensionTail == 3:
				listIndicesPilesExcluded.extend([excludeAbove - 2])
			if 3 < dimensionTail:
				listIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 4)])

		domainCombined.extend([(pileOfLeaf首二, pileOfLeaf首零二, pileOfLeaf首零一二, pileOfLeaf首一二) for pileOfLeaf首零一二, pileOfLeaf首一二 in exclude(domain一corners, listIndicesPilesExcluded)])

#======== By inclusion of the piles, add non-corners (52 tuples) ====================
	domain0nonCorners: tuple[tuple[int, int], ...] = tuple(set(domain首零二and首二).difference(set(domain0corners)))
	domainCombined.extend([(pileOfLeaf首二, pileOfLeaf首零二, pileOfLeaf首零二 - 1, pileOfLeaf首二 + 1) for pileOfLeaf首零二, pileOfLeaf首二 in domain0nonCorners])

	return tuple(sorted(filter(all_unique, set(domainCombined))))

def getDomain二零and二(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf二零 and leaf二."""
	domain二零: tuple[int, ...] = tuple(getLeafDomain(state, 二+零))
	domain二: tuple[int, ...] = tuple(getLeafDomain(state, 二))
	direction: CallableFunction[[int, int], int] = add
	return _getDomains二Or二一(domain二零, domain二, direction, state.dimensionsTotal, state.sumsOfProductsOfDimensions)

def getDomain二一零and二一(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf二一零 and leaf二一."""
	domain二一零: tuple[int, ...] = tuple(getLeafDomain(state, 二+一+零))
	domain二一: tuple[int, ...] = tuple(getLeafDomain(state, 二+一))
	direction: CallableFunction[[int, int], int] = sub
	return _getDomains二Or二一(domain二一零, domain二一, direction, state.dimensionsTotal, state.sumsOfProductsOfDimensions)

@cache
def _getDomains二Or二一(domain零: tuple[int, ...], domain0: tuple[int, ...], direction: CallableFunction[[int, int], int], dimensionsTotal: int, sumsOfProductsOfDimensions: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
	if direction(0, 6009) == 6009:
		ImaDomain二零and二: bool = True
		ImaDomain二一零and二一: bool = False
	else:
		ImaDomain二零and二 = False
		ImaDomain二一零and二一 = True

	domainCombined: list[tuple[int, int]] = []

#======== By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for indexDomain零, pileOfLeaf零 in enumerate(filter(between吗(pileOrigin, 首零(dimensionsTotal)-零), domain零)):
		indicesDomain0ToExclude: list[int] = []

		dimensionTail: int = dimensionNearestTail(pileOfLeaf零 - is_odd(pileOfLeaf零))

# ******* (Almost) All differences between `_getDomain二零and二` and `_getDomain二一零and二一` *******
#-------- Two identifiers with different values -------------------
		# One default value from each option is a type of defensive coding, and the type checkers won't complain about possibly unbound values.
		excludeBelowAddend: int = 0
		steppingBasisForUnknownReasons: int = indexDomain零
		if ImaDomain二零and二:
			excludeBelowAddend = 0
			steppingBasisForUnknownReasons = int(bit_mask(dimensionTail - 1).bit_flip(0)) # How the hell did I figure out this bizarre formula?
		elif ImaDomain二一零and二一:
			excludeBelowAddend = int(is_even(indexDomain零) or dimensionTail)
			steppingBasisForUnknownReasons = indexDomain零

# - - - - Two special cases that 1) might be inherent, such as the differences in `pilesFewerDomain0`, or 2) might be because the formulas could be better. I'd bet on number 2.
		if ImaDomain二零and二:
			if pileOfLeaf零 == 二:
				indicesDomain0ToExclude.extend([*range(indexDomain零 + 1)])
			if pileOfLeaf零 == (首一(dimensionsTotal) + 首二(dimensionsTotal) + 首三(dimensionsTotal)):
				indexDomain0: int = int(7 * pilesTotal / 8)
				indexDomain0 -= pilesFewerDomain0
				indicesDomain0ToExclude.extend([indexDomain0])
# ******* end *******

#-------- `excludeBelow` `index` ---------------------------------
		excludeBelow: int = indexDomain零 + excludeBelowAddend
		excludeBelow -= pilesFewerDomain0
		indicesDomain0ToExclude.extend(range(excludeBelow))

#-------- `excludeAbove` `index` ---------------------------------
		if pileOfLeaf零 <= 首一(dimensionsTotal):
			excludeAbove: int = indexDomain零 + (3 * pilesTotal // 4)
			excludeAbove -= pilesFewerDomain0
			indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))
		if 首一(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal):
			excludeAbove = int(pileOfLeaf零 ^ bit_mask(dimensionsTotal)) // 2
			indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))

#-------- Exclude by stepping: exclude ((2^dimensionTail - 1) / (2^dimensionTail))-many indices, e.g., 1/2, 3/4, 15/16, after `index` -----------------
		for dimension in range(dimensionTail):
			indicesDomain0ToExclude.extend(range(steppingBasisForUnknownReasons + int(bit_mask(dimension)), pilesTotal, int(bit_flip(0, dimension + 1))))

#-------- Exclude "knock-out" indices ---------------------------------
		if dimensionTail == 1:
			if (首二(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal)-零) and (2 < dimensionNearest首(pileOfLeaf零)):
				if dimensionSecondNearest首(pileOfLeaf零) == 零:
					indexDomain0: int = pilesTotal // 2
					indexDomain0 -= pilesFewerDomain0
					if 4 < domain0[indexDomain0].bit_length():
						indicesDomain0ToExclude.extend([indexDomain0])
					if 首一(dimensionsTotal) < pileOfLeaf零:
						indexDomain0 = -(pilesTotal // 4 - is_odd(pileOfLeaf零))
						indexDomain0 -= -(pilesFewerDomain0)
						indicesDomain0ToExclude.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					indexDomain0 = pilesTotal // 2 + 2
					indexDomain0 -= pilesFewerDomain0
					if domain0[indexDomain0] < 首零(dimensionsTotal):
						indicesDomain0ToExclude.extend([indexDomain0])
					indexDomain0 = -(pilesTotal // 4 - 2)
					indexDomain0 -= -(pilesFewerDomain0)
					if 首一(dimensionsTotal) < pileOfLeaf零:
						indicesDomain0ToExclude.extend([indexDomain0])

				if dimensionSecondNearest首(pileOfLeaf零) == 一+零:
					indexDomain0 = -(pilesTotal // 4)
					indexDomain0 -= -(pilesFewerDomain0)
					indicesDomain0ToExclude.extend([indexDomain0])

				indexDomain0 = 3 * pilesTotal // 4
				indexDomain0 -= pilesFewerDomain0
				if pileOfLeaf零 < 首一二(dimensionsTotal):
# NOTE My thinking: because       首一二(dimensionsTotal)
					dimensionIndexPart首: int = dimensionsTotal
					dimensionIndexPart一: int = dimensionIndex(一)
					dimensionIndexPart二: int = dimensionIndex(二)

					# Compute the index from the head `首`
					indexSumsOfProductsOfDimensions: int = dimensionIndexPart首 - (dimensionIndexPart一 + dimensionIndexPart二)

					addend: int = sumsOfProductsOfDimensions[indexSumsOfProductsOfDimensions]
					if ImaDomain二一零and二一:
						addend -= 1 # decreasing?
					pileOfLeaf0: int = addend + 首零(dimensionsTotal)
					indexDomain0 = domain0.index(pileOfLeaf0)

					indicesDomain0ToExclude.extend([indexDomain0])

				if dimensionThirdNearest首(pileOfLeaf零) == 零:
					if dimensionSecondNearest首(pileOfLeaf零) == 一+零:
						indicesDomain0ToExclude.extend([indexDomain0 - 2])
					if dimensionNearest首(pileOfLeaf零) == 一+零:
						indicesDomain0ToExclude.extend([indexDomain0 - 2])

		elif 首一(dimensionsTotal) + 首三(dimensionsTotal) + is_odd(pileOfLeaf零) == pileOfLeaf零:
			indexDomain0 = (3 * pilesTotal // 4) - 1
			indexDomain0 -= pilesFewerDomain0
			indicesDomain0ToExclude.extend([indexDomain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, indicesDomain0ToExclude)])

#======== By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

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

#======== By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	direction: CallableFunction[[int, int], int] = sub
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

#======== By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for index, pileOfLeaf零 in enumerate(domain零):
		if pileOfLeaf零 < 首零(dimensionsTotal)+零:
			continue
		listIndicesPilesExcluded: list[int] = []

		dimensionTail: int = dimensionNearestTail(direction(pileOfLeaf零, is_odd(pileOfLeaf零)))

#-------- `excludeBelow` `index` ---------------------------------
		if 首零一(dimensionsTotal) < pileOfLeaf零:
			excludeBelow: int = index + 3 - (3 * pilesTotal // 4)
		else:
			excludeBelow = 2 + (首零一(dimensionsTotal) - direction(pileOfLeaf零, is_odd(pileOfLeaf零))) // 2
		excludeBelow -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeBelow))

#-------- `excludeAbove` `index` ---------------------------------
		excludeAbove: int = index + 2 - int(bit_mask(dimensionTail))
		excludeAbove -= pilesFewerDomain0
		listIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

#-------- Exclude by stepping: exclude ((2^dimensionTail - 1) / (2^dimensionTail))-many indices, e.g., 1/2, 3/4, 15/16, after `index` -----------------
		countFromTheEnd: int = pilesTotal - 1
		countFromTheEnd -= pilesFewerDomain0
		steppingBasisForUnknownReasons: int = countFromTheEnd - int(bit_mask(dimensionTail - 1).bit_flip(0))
		for dimension in range(dimensionTail):
			listIndicesPilesExcluded.extend(range(steppingBasisForUnknownReasons - int(bit_mask(dimension)), decreasing, decreasing * int(bit_flip(0, dimension + 1))))

#-------- Exclude "knock-out" indices ---------------------------------
		if dimensionTail == 1:
			if (dimensionThirdNearest首(pileOfLeaf零) == 一) and (二+零 <= dimensionNearest首(pileOfLeaf零)):
				indexDomain0: int = (pilesTotal // 2) + 1
				indexDomain0 -= pilesFewerDomain0
				listIndicesPilesExcluded.extend([indexDomain0])
				indexDomain0: int = (pilesTotal // 4) + 1
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
	direction: CallableFunction[[int, int], int] = add
	return _getDomain首零一二and首一二(domain首零一二, domain首一二, direction, state.dimensionsTotal)
@cache
def _getDomain首零一二and首一二(domain零: tuple[int, ...], domain0: tuple[int, ...], direction: CallableFunction[[int, int], int], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
	domainCombined: list[tuple[int, int]] = []

#======== By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for indexDomain零, pileOfLeaf零 in enumerate(domain零):
		if pileOfLeaf零 < 首零(dimensionsTotal):
			continue
		indicesDomain0ToExclude: list[int] = []

		dimensionTail: int = dimensionNearestTail(direction(pileOfLeaf零, is_odd(pileOfLeaf零)))

#-------- `excludeBelow` `index` ---------------------------------
		if 首零一(dimensionsTotal) < pileOfLeaf零:
			excludeBelow: int = indexDomain零 + 1 - (3 * pilesTotal // 4)
		else:
			excludeBelow = (首零一(dimensionsTotal) - direction(pileOfLeaf零, is_odd(pileOfLeaf零))) // 2
		excludeBelow -= pilesFewerDomain0
		indicesDomain0ToExclude.extend(range(excludeBelow))

#-------- `excludeAbove` `index` ---------------------------------
		excludeAbove: int = indexDomain零 + 1 - int(bit_mask(dimensionTail))
		excludeAbove -= pilesFewerDomain0
		indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))

#-------- Exclude by stepping: exclude ((2^dimensionTail - 1) / (2^dimensionTail))-many indices, e.g., 1/2, 3/4, 15/16, after `index` -----------------
		steppingBasisForUnknownReasons: int = indexDomain零
		for dimension in range(dimensionTail):
			indicesDomain0ToExclude.extend(range(steppingBasisForUnknownReasons - int(bit_mask(dimension)), decreasing, decreasing * int(bit_flip(0, dimension + 1))))

#-------- Exclude "knock-out" indices ---------------------------------
		if dimensionTail == 1:
			if (dimensionThirdNearest首(pileOfLeaf零) == 一) and (二+零 <= dimensionNearest首(pileOfLeaf零)):
				indexDomain0: int = pilesTotal // 2
				indexDomain0 -= pilesFewerDomain0
				indicesDomain0ToExclude.extend([indexDomain0])
				indexDomain0: int = pilesTotal // 4
				indexDomain0 -= pilesFewerDomain0
				indicesDomain0ToExclude.extend([indexDomain0])
				if pileOfLeaf零 < 首零一(dimensionsTotal):
					indicesDomain0ToExclude.extend([indexDomain0 - 2])
			if dimensionThirdNearest首(pileOfLeaf零) == 一+零:
				indexDomain0 = pilesTotal // 4
				indexDomain0 -= pilesFewerDomain0
				if dimensionFourthNearest首(pileOfLeaf零) == 一:
					indicesDomain0ToExclude.extend([indexDomain0])
			if howManyDimensionsHaveOddParity(pileOfLeaf零) == 一:
				indexDomain0 = (pilesTotal // 4) + 2
				indexDomain0 -= pilesFewerDomain0
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					indexDomain0 = domain0.index(首零(dimensionsTotal) - 一)
					indicesDomain0ToExclude.extend([indexDomain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 二:
					indicesDomain0ToExclude.extend([indexDomain0])
				if (首零二(dimensionsTotal) < pileOfLeaf零) and (二+零 <= dimensionNearest首(pileOfLeaf零)):
					indicesDomain0ToExclude.extend([indexDomain0 - 2])
					indexDomain0 = (pilesTotal // 2) - 2
					indexDomain0 -= pilesFewerDomain0
					indicesDomain0ToExclude.extend([indexDomain0])

		elif (首零一(dimensionsTotal) - direction(首三(dimensionsTotal), is_odd(pileOfLeaf零))) == pileOfLeaf零:
			indexDomain0 = (pilesTotal // 4) + 1
			indexDomain0 -= pilesFewerDomain0
			indicesDomain0ToExclude.extend([indexDomain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, indicesDomain0ToExclude)])

#======== By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

	return tuple(sorted(set(domainCombined)))

def getLeaf首零Plus零Domain(state: EliminationState, leaf: Leaf | None = None) -> tuple[Pile, ...]:
	"""Get the full domain of `leaf首零Plus零` that is valid in all cases, or if `leaf一零` and `leaf首零一` are pinned in `state.permutationSpace`, get a domain of `leaf首零Plus零` customized to `pileOfLeaf一零` and `pileOfLeaf首零一`."""
	if leaf is None:
		leaf = (零)+首零(state.dimensionsTotal)
	domain首零Plus零: tuple[Pile, ...] = tuple(getLeafDomain(state, leaf))
	leaf一零: Leaf = 一+零
	leaf首零一: Leaf = 首零一(state.dimensionsTotal)
	if leafIsPinned(state.permutationSpace, leaf一零) and leafIsPinned(state.permutationSpace, leaf首零一):
		pileOfLeaf一零: Pile = raiseIfNone(reverseLookup(state.permutationSpace, leaf一零))
		pileOfLeaf首零一: Pile = raiseIfNone(reverseLookup(state.permutationSpace, leaf首零一))
		domain首零Plus零 = _getLeaf首零Plus零Domain(domain首零Plus零, pileOfLeaf一零, pileOfLeaf首零一, state.dimensionsTotal, state.leavesTotal)
	return domain首零Plus零
@cache
def _getLeaf首零Plus零Domain(domain首零Plus零: tuple[Pile, ...], pileOfLeaf一零: Pile, pileOfLeaf首零一: Pile, dimensionsTotal: int, leavesTotal: int) -> tuple[Pile, ...]:
	pilesTotal: int = 首一(dimensionsTotal)

	bump: int = 1 - int(pileOfLeaf一零.bit_count() == 1)
	howMany: int = dimensionsTotal - (pileOfLeaf一零.bit_length() + bump)
	onesInBinary: int = int(bit_mask(howMany))
	ImaPattern: int = pilesTotal - onesInBinary

	listIndicesPilesExcluded: list[int] = []
	if pileOfLeaf一零 == 二:
		listIndicesPilesExcluded.extend([零, 一, 二]) # These symbols make this pattern jump out.

	if 二 < pileOfLeaf一零 <= 首二(dimensionsTotal):
		stop: int = pilesTotal // 2 - 1
		listIndicesPilesExcluded.extend(range(1, stop))

		aDimensionPropertyNotFullyUnderstood: int = 5
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
			IDK: int = ImaPattern - 1
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

