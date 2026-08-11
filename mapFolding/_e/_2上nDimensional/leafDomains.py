from __future__ import annotations

from functools import cache
from gmpy2 import bit_flip, bit_mask, is_even as isEven吗, is_odd as isOdd吗
from hunterMakesPy import decreasing, inclusive, raiseIfNone
from mapFolding._e import getDomainLeaf, leafOrigin
from mapFolding._e._2上nDimensional import (
	dimensionFourthNearest首, dimensionIndex, dimensionNearestTail, dimensionNearest首, dimensionSecondNearest首, dimensionThirdNearest首,
	howManyDimensionsHaveOddParity, 一, 三, 二, 四, 零, 首一, 首一二, 首三, 首二, 首零, 首零一, 首零一二, 首零二)
from mapFolding._e.dataBaskets import EliminationState
from mapFolding.beDRY import mapShapeIs2上nDimensions
from more_itertools import all_unique as allUnique吗, loops
from operator import add, sub
from typing import TYPE_CHECKING
from Z0Z_tools import consecutive吗, exclude, reverseLookup

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction
	from mapFolding._e.theTypes import Leaf, Pile

def getDomainDimension一(state: EliminationState) -> tuple[tuple[int, int, int, int], ...]:
	"""The beans and cornbread and beans and cornbread dimension.

	(leaf一零, leaf一, leaf首一, leaf首零一)
	^^^ Can you see the symmetry? ^^^
	"""
	domain一零: tuple[int, ...] = tuple(getDomainLeaf(state, 一 + 零))
	domain首一: tuple[int, ...] = tuple(getDomainLeaf(state, 首一(state.dimensionsTotal)))
	return _getDomainDimension一(domain一零, domain首一, state.dimensionsTotal)
@cache
def _getDomainDimension一(domain一零: tuple[int, ...], domain首一: tuple[int, ...], dimensionsTotal: int) -> tuple[tuple[int, int, int, int], ...]:
	domainCombined: list[tuple[int, int, int, int]] = []

	for pileOfLeaf一零 in domain一零:
		domainOfLeaf首一: tuple[int, ...] = domain首一
		pilesTotal: int = len(domainOfLeaf首一)

		boxOfIndicesPilesExcluded: list[int] = []

		if pileOfLeaf一零 <= 首二(dimensionsTotal):
			pass

		elif 首二(dimensionsTotal) < pileOfLeaf一零 < 首一(dimensionsTotal):
			boxOfIndicesPilesExcluded.extend([*range(1, pilesTotal // 2), *range(1 + pilesTotal // 2, 3 * pilesTotal // 4)])

		elif pileOfLeaf一零 == 首一(dimensionsTotal):
			boxOfIndicesPilesExcluded.extend([*range(1, pilesTotal // 2)])

		elif 首一(dimensionsTotal) < pileOfLeaf一零 < 首零(dimensionsTotal) - 一:
			boxOfIndicesPilesExcluded.extend([*range(3 * pilesTotal // 4)])

		elif pileOfLeaf一零 == 首零(dimensionsTotal) - 一:
			boxOfIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4)])

		elif pileOfLeaf一零 == 首零(dimensionsTotal):
			boxOfIndicesPilesExcluded.extend([*range(2, pilesTotal // 2)])

		domainOfLeaf首一 = tuple(exclude(domainOfLeaf首一, boxOfIndicesPilesExcluded))

		domainCombined.extend([(pileOfLeaf一零, pileOfLeaf一零 + 1, pileOfLeaf首一, pileOfLeaf首一 + 1) for pileOfLeaf首一 in domainOfLeaf首一])

	return tuple(filter(allUnique吗, domainCombined))

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
	for 次, (pileOfLeaf二一零, pileOfLeaf二一) in enumerate(domain一corners):
		boxOfIndicesPilesExcluded: list[int] = []

		dimensionTail: int = dimensionNearestTail(pileOfLeaf二一)

#-------- `excludeBelow` `次` ---------------------------------
		excludeBelow: int = 次
		boxOfIndicesPilesExcluded.extend(range(excludeBelow))

#-------- `excludeAbove` `次` ---------------------------------
		excludeAbove: int = pilesTotal
		if pileOfLeaf二一 <= 首一(dimensionsTotal):
			if dimensionTail == 1:
				excludeAbove = pilesTotal // 2 + 次
				if howManyDimensionsHaveOddParity(pileOfLeaf二一) == 2:
					excludeAbove -= 1

				if (howManyDimensionsHaveOddParity(pileOfLeaf二一) == 1 and (2 < dimensionNearest首(pileOfLeaf二一))):
					excludeAbove += 2

				if (howManyDimensionsHaveOddParity(pileOfLeaf二一) == 1
					and (dimensionNearest首(pileOfLeaf二一) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf二一)) < 2)
				):
					addend: int = productsOfDimensions[dimensionsTotal - 2] + 4
					excludeAbove = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))

			else:
				excludeAbove = 3 * pilesTotal // 4 + 2
				if 次 == 0:
					excludeAbove = 1
				elif 次 <= 2:
					addend = 三 + sum(productsOfDimensions[1:dimensionsTotal - 2])
					excludeAbove = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
		boxOfIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

#-------- Exclude "knock-out" indices ---------------------------------
		if pileOfLeaf二一 < 首一二(dimensionsTotal):
			if dimensionTail == 4:
				addend = int(bit_flip(0, dimensionTail))
				start: int = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
				boxOfIndicesPilesExcluded.extend([*range(start, start + dimensionTail)])
			if dimensionTail == 3:
				addend = int(bit_flip(0, dimensionTail))
				start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
				boxOfIndicesPilesExcluded.extend([*range(start, start + dimensionTail - 1)])
				start = domain0corners.index((pileOfLeaf二一 + addend * 2, pileOfLeaf二一零 + addend * 2))
				boxOfIndicesPilesExcluded.extend([*range(start - 1, start + dimensionTail - 1)])
			if (dimensionTail < 3) and (2 < dimensionNearest首(pileOfLeaf二一)):
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
					boxOfIndicesPilesExcluded.extend([*range(start, stop, step)])
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
					boxOfIndicesPilesExcluded.extend([*range(start, stop, step)])
			if dimensionNearest首(pileOfLeaf二一) == 2:
				addend = 三
				start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
				boxOfIndicesPilesExcluded.extend([*range(start, start + addend, 2)])

		domainCombined.extend([(pileOfLeaf二一, pileOfLeaf二一零, pileOfLeaf二零, pileOfLeaf二) for pileOfLeaf二零, pileOfLeaf二 in exclude(domain0corners, boxOfIndicesPilesExcluded)])

#======== By inclusion of the piles, add non-corners (52 tuples) ====================
	domain一nonCorners: tuple[tuple[int, int], ...] = tuple(set(domain二一零and二一).difference(set(domain一corners)))
	domainCombined.extend([(pileOfLeaf一二, pileOfLeaf二一零, pileOfLeaf二一零 - 1, pileOfLeaf一二 + 1) for pileOfLeaf二一零, pileOfLeaf一二 in domain一nonCorners])

	return tuple(sorted(filter(allUnique吗, set(domainCombined))))

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
	for 次, (pileOfLeaf首零二, pileOfLeaf首二) in enumerate(domain0corners):
		boxOfIndicesPilesExcluded: list[int] = []

		dimensionTail: int = dimensionNearestTail(pileOfLeaf首零二)

#-------- `excludeBelow` `次` ---------------------------------
		excludeBelow: int = 次 - 1
		boxOfIndicesPilesExcluded.extend(range(excludeBelow))

#-------- `excludeAbove` `次` ---------------------------------
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
		boxOfIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

#-------- Exclude "knock-out" indices ---------------------------------
		if dimensionTail == 1 and (abs(pileOfLeaf首零二 - 首零(dimensionsTotal)) == 2) and isEven吗(dimensionsTotal):
			boxOfIndicesPilesExcluded.extend([excludeAbove - 2])
		if dimensionTail != 1 and 首一二(dimensionsTotal) <= pileOfLeaf首零二 <= 首零一(dimensionsTotal):
			if (dimensionTail == 2) and (howManyDimensionsHaveOddParity(pileOfLeaf首零二) + 1 != dimensionNearest首(pileOfLeaf首零二) - raiseIfNone(dimensionSecondNearest首(pileOfLeaf首零二))):
				boxOfIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 8 + 2)])
				if (pileOfLeaf首零二 <= 首零(dimensionsTotal)) and isEven吗(dimensionsTotal):
					boxOfIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 4 - 1)])
			if dimensionTail == 3:
				boxOfIndicesPilesExcluded.extend([excludeAbove - 2])
			if 3 < dimensionTail:
				boxOfIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(dimensionsTotal)) // 4)])

		domainCombined.extend([(pileOfLeaf首二, pileOfLeaf首零二, pileOfLeaf首零一二, pileOfLeaf首一二) for pileOfLeaf首零一二, pileOfLeaf首一二 in exclude(domain一corners, boxOfIndicesPilesExcluded)])

#======== By inclusion of the piles, add non-corners (52 tuples) ====================
	domain0nonCorners: tuple[tuple[int, int], ...] = tuple(set(domain首零二and首二).difference(set(domain0corners)))
	domainCombined.extend([(pileOfLeaf首二, pileOfLeaf首零二, pileOfLeaf首零二 - 1, pileOfLeaf首二 + 1) for pileOfLeaf首零二, pileOfLeaf首二 in domain0nonCorners])

	return tuple(sorted(filter(allUnique吗, set(domainCombined))))

def getDomain二零and二(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf二零 and leaf二."""
	domain二零: tuple[int, ...] = tuple(getDomainLeaf(state, 二 + 零))
	domain二: tuple[int, ...] = tuple(getDomainLeaf(state, 二))
	direction: CallableFunction[[int, int], int] = add
	return _getDomains二Or二一(domain二零, domain二, direction, state.dimensionsTotal, state.sumsOfProductsOfDimensions)

def getDomain二一零and二一(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf二一零 and leaf二一."""
	domain二一零: tuple[int, ...] = tuple(getDomainLeaf(state, 二 + 一 + 零))
	domain二一: tuple[int, ...] = tuple(getDomainLeaf(state, 二 + 一))
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

	for 次Domain零, pileOfLeaf零 in enumerate(filter((首零(dimensionsTotal) - 零).__ge__, domain零)):
		indicesDomain0ToExclude: list[int] = []

		dimensionTail: int = dimensionNearestTail(pileOfLeaf零 - isOdd吗(pileOfLeaf零))

# ******* (Almost) All differences between `_getDomain二零and二` and `_getDomain二一零and二一` *******
#-------- Two identifiers with different values -------------------
		# One default value from each option is a type of defensive coding, and the type checkers won't complain about possibly unbound values.
		excludeBelowAddend: int = 0
		steppingBasisForUnknownReasons: int = 次Domain零
		if ImaDomain二零and二:
			excludeBelowAddend = 0
			steppingBasisForUnknownReasons = int(bit_mask(dimensionTail - 1).bit_flip(0))  # How the hell did I figure out this bizarre formula?
		elif ImaDomain二一零and二一:
			excludeBelowAddend = int(isEven吗(次Domain零) or dimensionTail)
			steppingBasisForUnknownReasons = 次Domain零

# - - - - Two special cases that 1) might be inherent, such as the differences in `pilesFewerDomain0`, or 2) might be because the formulas could be better. I'd bet on number 2.
		if ImaDomain二零and二:
			if pileOfLeaf零 == 二:
				indicesDomain0ToExclude.extend([*range(次Domain零 + 1)])
			if pileOfLeaf零 == (首一(dimensionsTotal) + 首二(dimensionsTotal) + 首三(dimensionsTotal)):
				次Domain0: int = int(7 * pilesTotal / 8)
				次Domain0 -= pilesFewerDomain0
				indicesDomain0ToExclude.extend([次Domain0])
# ******* end *******

#-------- `excludeBelow` `次` ---------------------------------
		excludeBelow: int = 次Domain零 + excludeBelowAddend
		excludeBelow -= pilesFewerDomain0
		indicesDomain0ToExclude.extend(range(excludeBelow))

#-------- `excludeAbove` `次` ---------------------------------
		if pileOfLeaf零 <= 首一(dimensionsTotal):
			excludeAbove: int = 次Domain零 + (3 * pilesTotal // 4)
			excludeAbove -= pilesFewerDomain0
			indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))
		if 首一(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal):
			excludeAbove = int(pileOfLeaf零 ^ bit_mask(dimensionsTotal)) // 2
			indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))

#-------- Exclude by stepping: exclude ((2^dimensionTail - 1) / (2^dimensionTail))-many indices, e.g., 1/2, 3/4, 15/16, after `次` -----------------
		for dimension in range(dimensionTail):
			indicesDomain0ToExclude.extend(range(steppingBasisForUnknownReasons + int(bit_mask(dimension)), pilesTotal, int(bit_flip(0, dimension + 1))))

#-------- Exclude "knock-out" indices ---------------------------------
		if dimensionTail == 1:
			if (首二(dimensionsTotal) < pileOfLeaf零 < 首零(dimensionsTotal) - 零) and (2 < dimensionNearest首(pileOfLeaf零)):
				if dimensionSecondNearest首(pileOfLeaf零) == 零:
					次Domain0: int = pilesTotal // 2
					次Domain0 -= pilesFewerDomain0
					if 4 < domain0[次Domain0].bit_length():
						indicesDomain0ToExclude.extend([次Domain0])
					if 首一(dimensionsTotal) < pileOfLeaf零:
						次Domain0 = -(pilesTotal // 4 - isOdd吗(pileOfLeaf零))
						次Domain0 -= -(pilesFewerDomain0)
						indicesDomain0ToExclude.extend([次Domain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					次Domain0 = pilesTotal // 2 + 2
					次Domain0 -= pilesFewerDomain0
					if domain0[次Domain0] < 首零(dimensionsTotal):
						indicesDomain0ToExclude.extend([次Domain0])
					次Domain0 = -(pilesTotal // 4 - 2)
					次Domain0 -= -(pilesFewerDomain0)
					if 首一(dimensionsTotal) < pileOfLeaf零:
						indicesDomain0ToExclude.extend([次Domain0])

				if dimensionSecondNearest首(pileOfLeaf零) == 一 + 零:
					次Domain0 = -(pilesTotal // 4)
					次Domain0 -= -(pilesFewerDomain0)
					indicesDomain0ToExclude.extend([次Domain0])

				次Domain0 = 3 * pilesTotal // 4
				次Domain0 -= pilesFewerDomain0
				if pileOfLeaf零 < 首一二(dimensionsTotal):
					dimensionIndexPart首: int = dimensionsTotal
					dimensionIndexPart一: int = dimensionIndex(一)
					dimensionIndexPart二: int = dimensionIndex(二)

					# Compute the 次 from the head `首`
					次SumsOfProductsOfDimensions: int = dimensionIndexPart首 - (dimensionIndexPart一 + dimensionIndexPart二)

					addend: int = sumsOfProductsOfDimensions[次SumsOfProductsOfDimensions]
					if ImaDomain二一零and二一:
						addend -= 1  # decreasing?
					pileOfLeaf0: int = addend + 首零(dimensionsTotal)
					次Domain0 = domain0.index(pileOfLeaf0)

					indicesDomain0ToExclude.extend([次Domain0])

				if dimensionThirdNearest首(pileOfLeaf零) == 零:
					if dimensionSecondNearest首(pileOfLeaf零) == 一 + 零:
						indicesDomain0ToExclude.extend([次Domain0 - 2])
					if dimensionNearest首(pileOfLeaf零) == 一 + 零:
						indicesDomain0ToExclude.extend([次Domain0 - 2])

		elif 首一(dimensionsTotal) + 首三(dimensionsTotal) + isOdd吗(pileOfLeaf零) == pileOfLeaf零:
			次Domain0 = (3 * pilesTotal // 4) - 1
			次Domain0 -= pilesFewerDomain0
			indicesDomain0ToExclude.extend([次Domain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, indicesDomain0ToExclude)])

#======== By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

	return tuple(sorted(set(domainCombined)))

def getDomain首零二and首二(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf首零二 and leaf首二."""
	domain首零二: tuple[int, ...] = tuple(getDomainLeaf(state, 首零二(state.dimensionsTotal)))
	domain首二: tuple[int, ...] = tuple(getDomainLeaf(state, 首二(state.dimensionsTotal)))
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

	for 次, pileOfLeaf零 in enumerate(domain零):
		if pileOfLeaf零 < 首零(dimensionsTotal) + 零:
			continue
		boxOfIndicesPilesExcluded: list[int] = []

		dimensionTail: int = dimensionNearestTail(direction(pileOfLeaf零, isOdd吗(pileOfLeaf零)))

#-------- `excludeBelow` `次` ---------------------------------
		if 首零一(dimensionsTotal) < pileOfLeaf零:
			excludeBelow: int = 次 + 3 - (3 * pilesTotal // 4)
		else:
			excludeBelow = 2 + (首零一(dimensionsTotal) - direction(pileOfLeaf零, isOdd吗(pileOfLeaf零))) // 2
		excludeBelow -= pilesFewerDomain0
		boxOfIndicesPilesExcluded.extend(range(excludeBelow))

#-------- `excludeAbove` `次`------------------------------
		excludeAbove: int = 次 + 2 - int(bit_mask(dimensionTail))
		excludeAbove -= pilesFewerDomain0
		boxOfIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

#-------- Exclude by stepping: exclude ((2^dimensionTail - 1) / (2^dimensionTail))-many indices, e.g., 1/2, 3/4, 15/16, after `次` -----------------
		countFromTheEnd: int = pilesTotal - 1
		countFromTheEnd -= pilesFewerDomain0
		steppingBasisForUnknownReasons: int = countFromTheEnd - int(bit_mask(dimensionTail - 1).bit_flip(0))
		for dimension in range(dimensionTail):
			boxOfIndicesPilesExcluded.extend(range(steppingBasisForUnknownReasons - int(bit_mask(dimension)), decreasing, decreasing * int(bit_flip(0, dimension + 1))))

#-------- Exclude "knock-out" indices ---------------------------------
		if dimensionTail == 1:
			if (dimensionThirdNearest首(pileOfLeaf零) == 一) and (二 + 零 <= dimensionNearest首(pileOfLeaf零)):
				次Domain0: int = (pilesTotal // 2) + 1
				次Domain0 -= pilesFewerDomain0
				boxOfIndicesPilesExcluded.extend([次Domain0])
				次Domain0: int = (pilesTotal // 4) + 1
				次Domain0 -= pilesFewerDomain0
				boxOfIndicesPilesExcluded.extend([次Domain0])
				if pileOfLeaf零 < 首零一(dimensionsTotal):
					boxOfIndicesPilesExcluded.extend([次Domain0 - 2])

			if howManyDimensionsHaveOddParity(pileOfLeaf零) == 一:
				次Domain0 = (pilesTotal // 4) + 3
				次Domain0 -= pilesFewerDomain0
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					boxOfIndicesPilesExcluded.extend([次Domain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 二:
					boxOfIndicesPilesExcluded.extend([次Domain0])
				if (((dimensionNearest首(pileOfLeaf零) == dimensionsTotal - 1) and (dimensionSecondNearest首(pileOfLeaf零) == dimensionsTotal - 3))
					or (dimensionSecondNearest首(pileOfLeaf零) == 二)):
					boxOfIndicesPilesExcluded.extend([次Domain0 - 2])
					次Domain0 = (pilesTotal // 2) - 1
					次Domain0 -= pilesFewerDomain0
					boxOfIndicesPilesExcluded.extend([次Domain0])

		elif (首零一(dimensionsTotal) - direction(首三(dimensionsTotal), isOdd吗(pileOfLeaf零))) == pileOfLeaf零:
			次Domain0 = (pilesTotal // 4) + 2
			次Domain0 -= pilesFewerDomain0
			boxOfIndicesPilesExcluded.extend([次Domain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, boxOfIndicesPilesExcluded)])

	return tuple(sorted(set(domainCombined)))

def getDomain首零一二and首一二(state: EliminationState) -> tuple[tuple[int, int], ...]:
	"""Combined domain of leaf首零一二 and leaf首一二."""
	domain首零一二: tuple[int, ...] = tuple(getDomainLeaf(state, 首零一二(state.dimensionsTotal)))
	domain首一二: tuple[int, ...] = tuple(getDomainLeaf(state, 首一二(state.dimensionsTotal)))
	direction: CallableFunction[[int, int], int] = add
	return _getDomain首零一二and首一二(domain首零一二, domain首一二, direction, state.dimensionsTotal)
@cache
def _getDomain首零一二and首一二(domain零: tuple[int, ...], domain0: tuple[int, ...], direction: CallableFunction[[int, int], int], dimensionsTotal: int) -> tuple[tuple[int, int], ...]:
	domainCombined: list[tuple[int, int]] = []

#======== By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for 次Domain零, pileOfLeaf零 in enumerate(domain零):
		if pileOfLeaf零 < 首零(dimensionsTotal):
			continue
		indicesDomain0ToExclude: list[int] = []

		dimensionTail: int = dimensionNearestTail(direction(pileOfLeaf零, isOdd吗(pileOfLeaf零)))

#-------- `excludeBelow` `次` ---------------------------------
		if 首零一(dimensionsTotal) < pileOfLeaf零:
			excludeBelow: int = 次Domain零 + 1 - (3 * pilesTotal // 4)
		else:
			excludeBelow = (首零一(dimensionsTotal) - direction(pileOfLeaf零, isOdd吗(pileOfLeaf零))) // 2
		excludeBelow -= pilesFewerDomain0
		indicesDomain0ToExclude.extend(range(excludeBelow))

#-------- `excludeAbove` `次` ---------------------------------
		excludeAbove: int = 次Domain零 + 1 - int(bit_mask(dimensionTail))
		excludeAbove -= pilesFewerDomain0
		indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))

#-------- Exclude by stepping: exclude ((2^dimensionTail - 1) / (2^dimensionTail))-many indices, e.g., 1/2, 3/4, 15/16, after `次` -----------------
		steppingBasisForUnknownReasons: int = 次Domain零
		for dimension in range(dimensionTail):
			indicesDomain0ToExclude.extend(range(steppingBasisForUnknownReasons - int(bit_mask(dimension)), decreasing, decreasing * int(bit_flip(0, dimension + 1))))

#-------- Exclude "knock-out" indices ---------------------------------
		if dimensionTail == 1:
			if (dimensionThirdNearest首(pileOfLeaf零) == 一) and (二 + 零 <= dimensionNearest首(pileOfLeaf零)):
				次Domain0: int = pilesTotal // 2
				次Domain0 -= pilesFewerDomain0
				indicesDomain0ToExclude.extend([次Domain0])
				次Domain0: int = pilesTotal // 4
				次Domain0 -= pilesFewerDomain0
				indicesDomain0ToExclude.extend([次Domain0])
				if pileOfLeaf零 < 首零一(dimensionsTotal):
					indicesDomain0ToExclude.extend([次Domain0 - 2])
			if dimensionThirdNearest首(pileOfLeaf零) == 一 + 零:
				次Domain0 = pilesTotal // 4
				次Domain0 -= pilesFewerDomain0
				if dimensionFourthNearest首(pileOfLeaf零) == 一:
					indicesDomain0ToExclude.extend([次Domain0])
			if howManyDimensionsHaveOddParity(pileOfLeaf零) == 一:
				次Domain0 = (pilesTotal // 4) + 2
				次Domain0 -= pilesFewerDomain0
				if dimensionSecondNearest首(pileOfLeaf零) == 一:
					次Domain0 = domain0.index(首零(dimensionsTotal) - 一)
					indicesDomain0ToExclude.extend([次Domain0])
				if dimensionSecondNearest首(pileOfLeaf零) == 二:
					indicesDomain0ToExclude.extend([次Domain0])
				if (首零二(dimensionsTotal) < pileOfLeaf零) and (二 + 零 <= dimensionNearest首(pileOfLeaf零)):
					indicesDomain0ToExclude.extend([次Domain0 - 2])
					次Domain0 = (pilesTotal // 2) - 2
					次Domain0 -= pilesFewerDomain0
					indicesDomain0ToExclude.extend([次Domain0])

		elif (首零一(dimensionsTotal) - direction(首三(dimensionsTotal), isOdd吗(pileOfLeaf零))) == pileOfLeaf零:
			次Domain0 = (pilesTotal // 4) + 1
			次Domain0 -= pilesFewerDomain0
			indicesDomain0ToExclude.extend([次Domain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, indicesDomain0ToExclude)])

#======== By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

	return tuple(sorted(set(domainCombined)))

@cache
def _getDomainLeaf(leaf: Leaf, dimensionsTotal: int, mapShape: tuple[int, ...], leavesTotal: int) -> range:
	"""The subroutines assume `dimensionLength == 2`, but I think the concept could be extended to other `mapShape`."""
	state: EliminationState = EliminationState(mapShape)
	if mapShapeIs2上nDimensions(state.mapShape):
		originPinned: bool = leaf == leafOrigin
		return range(
					state.sumsOfProductsOfDimensions[dimensionNearestTail(leaf) + inclusive]  		# `start`, first value included in the `range`.
						+ howManyDimensionsHaveOddParity(leaf)
						- originPinned

					, state.sumsOfProductsOfDimensionsNearest首[dimensionNearest首(leaf)]  			# `stop`, first value excluded from the `range`.
						+ 2
						- howManyDimensionsHaveOddParity(leaf)
						- originPinned

					, 2 + (2 * (leaf == 首零(dimensionsTotal) + 零))								# `step`
				)
	return range(leavesTotal)

def getDomainLeaf首零Plus零(state: EliminationState, leaf: Leaf | None = None) -> tuple[Pile, ...]:
	"""Get the full domain of `leaf首零Plus零` that is valid in all cases, or if `leaf一零` and `leaf首零一` are pinned in `state.permutationSpace`, get a domain of `leaf首零Plus零` customized to `pileOfLeaf一零` and `pileOfLeaf首零一`."""
	if leaf is None:
		leaf = (零) + 首零(state.dimensionsTotal)
	domain首零Plus零: tuple[Pile, ...] = tuple(getDomainLeaf(state, leaf))
	leaf一零: Leaf = 一 + 零
	leaf首零一: Leaf = 首零一(state.dimensionsTotal)
	if state.permutationSpace.leafPinned吗(leaf一零) and state.permutationSpace.leafPinned吗(leaf首零一):
		pileOfLeaf一零: Pile = raiseIfNone(reverseLookup(state.permutationSpace, leaf一零))
		pileOfLeaf首零一: Pile = raiseIfNone(reverseLookup(state.permutationSpace, leaf首零一))
		domain首零Plus零 = _getDomainLeaf首零Plus零(domain首零Plus零, pileOfLeaf一零, pileOfLeaf首零一, state.dimensionsTotal, state.leavesTotal)
	return domain首零Plus零
@cache
def _getDomainLeaf首零Plus零(domain首零Plus零: tuple[Pile, ...], pileOfLeaf一零: Pile, pileOfLeaf首零一: Pile, dimensionsTotal: int, leavesTotal: int) -> tuple[Pile, ...]:
	pilesTotal: int = 首一(dimensionsTotal)

	bump: int = 1 - int(pileOfLeaf一零.bit_count() == 1)
	howMany: int = dimensionsTotal - (pileOfLeaf一零.bit_length() + bump)
	onesInBinary: int = int(bit_mask(howMany))
	ImaPattern: int = pilesTotal - onesInBinary

	boxOfIndicesPilesExcluded: list[int] = []
	if pileOfLeaf一零 == 二:
		boxOfIndicesPilesExcluded.extend([零, 一, 二])  # These symbols make this pattern jump out.

	if 二 < pileOfLeaf一零 <= 首二(dimensionsTotal):
		stop: int = pilesTotal // 2 - 1
		boxOfIndicesPilesExcluded.extend(range(1, stop))

		aDimensionPropertyNotFullyUnderstood: int = 5
		for _dimension in loops(dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
			start: int = 1 + stop
			stop += (stop + 1) // 2
			boxOfIndicesPilesExcluded.extend([*range(start, stop)])

		boxOfIndicesPilesExcluded.extend([*range(1 + stop, ImaPattern)])

	if 首二(dimensionsTotal) < pileOfLeaf一零:
		boxOfIndicesPilesExcluded.extend([*range(1, ImaPattern)])

	bump = 1 - int((leavesTotal - pileOfLeaf首零一).bit_count() == 1)
	howMany = dimensionsTotal - ((leavesTotal - pileOfLeaf首零一).bit_length() + bump)
	onesInBinary = int(bit_mask(howMany))
	ImaPattern = pilesTotal - onesInBinary

	aDimensionPropertyNotFullyUnderstood = 5

	if pileOfLeaf首零一 == leavesTotal - 二:
		boxOfIndicesPilesExcluded.extend([-零 - 1, -(一) - 1])
		if aDimensionPropertyNotFullyUnderstood <= dimensionsTotal:
			boxOfIndicesPilesExcluded.extend([-二 - 1])

	if ((首零一二(dimensionsTotal) < pileOfLeaf首零一 < leavesTotal - 二)
		and (首二(dimensionsTotal) < pileOfLeaf一零 <= 首零(dimensionsTotal))):
		boxOfIndicesPilesExcluded.extend([-1])

	if 首零一二(dimensionsTotal) <= pileOfLeaf首零一 < leavesTotal - 二:
		stop: int = pilesTotal // 2 - 1
		boxOfIndicesPilesExcluded.extend(range((1 + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing))

		for _dimension in loops(dimensionsTotal - aDimensionPropertyNotFullyUnderstood):
			start: int = 1 + stop
			stop += (stop + 1) // 2
			boxOfIndicesPilesExcluded.extend([*range((start + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing)])

		boxOfIndicesPilesExcluded.extend([*range((1 + stop + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])

		if 二 <= pileOfLeaf一零 <= 首零(dimensionsTotal):
			boxOfIndicesPilesExcluded.extend([零, 一, 二, pilesTotal // 2])

	if ((pileOfLeaf首零一 == 首零一二(dimensionsTotal))
		and (首一(dimensionsTotal) < pileOfLeaf一零 <= 首零(dimensionsTotal))):
		boxOfIndicesPilesExcluded.extend([-1])

	if 首零一(dimensionsTotal) < pileOfLeaf首零一 < 首零一二(dimensionsTotal):
		if pileOfLeaf一零 in {首一(dimensionsTotal), 首零(dimensionsTotal)}:
			boxOfIndicesPilesExcluded.extend([-1])
		elif 二 < pileOfLeaf一零 < 首二(dimensionsTotal):
			boxOfIndicesPilesExcluded.extend([0])

	if pileOfLeaf首零一 < 首零一二(dimensionsTotal):
		boxOfIndicesPilesExcluded.extend([*range((1 + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])

	pileOfLeaf一零ARCHETYPICAL: int = 首一(dimensionsTotal)
	bump = 1 - int(pileOfLeaf一零ARCHETYPICAL.bit_count() == 1)
	howMany = dimensionsTotal - (pileOfLeaf一零ARCHETYPICAL.bit_length() + bump)
	onesInBinary = int(bit_mask(howMany))
	ImaPattern = pilesTotal - onesInBinary

	if pileOfLeaf首零一 == leavesTotal - 二:
		if pileOfLeaf一零 == 二:
			boxOfIndicesPilesExcluded.extend([零, 一, 二, pilesTotal // 2 - 1, pilesTotal // 2])
		if 二 < pileOfLeaf一零 <= 首零(dimensionsTotal):
			IDK: int = ImaPattern - 1
			boxOfIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4), *range(1 + 3 * pilesTotal // 4, IDK)])
		if 首一(dimensionsTotal) < pileOfLeaf一零 <= 首零(dimensionsTotal):
			boxOfIndicesPilesExcluded.extend([-1])

	if pileOfLeaf首零一 == 首零一(dimensionsTotal):
		if pileOfLeaf一零 == 首零(dimensionsTotal):
			boxOfIndicesPilesExcluded.extend([-1])
		elif (二 < pileOfLeaf一零 < 首二(dimensionsTotal)) or (首二(dimensionsTotal) < pileOfLeaf一零 < 首一(dimensionsTotal)):
			boxOfIndicesPilesExcluded.extend([0])

	return tuple(exclude(domain首零Plus零, boxOfIndicesPilesExcluded))
