"""Compute joint pile domains and restrictions from pinned leaves.

(AI generated docstring)

You can enumerate possible pile assignments for specific leaf groups in maps whose
axis lengths are `2`. Combined domains describe joint assignments; the special
head-plus-one domain can also use pinned leaves from the workbench state.

Contents
--------
Functions
	getDomainDimension一
		Compute combined pile choices for the dimension-one leaves.
	getDomainDimension二
		Compute combined pile choices for the dimension-two leaves.
	getDomainDimension首二
		Compute combined pile choices for the head-relative dimension-two leaves.
	getDomainLeaf首零Plus零
		Compute pile choices for the head-plus-one leaf using pinned neighbors.
	getDomain二一零and二一
		Compute combined pile choices for the leaves seven and six.
	getDomain二零and二
		Compute combined pile choices for the leaves five and four.
	getDomain首零一二and首一二
		Compute combined pile choices for the head-relative zero-one-two and one-two leaves.
	getDomain首零二and首二
		Compute combined pile choices for the head-relative zero-two and two leaves.
"""
from __future__ import annotations

from functools import cache
from gmpy2 import bit_flip, bit_mask, is_even as isEven吗, is_odd as isOdd吗
from hunterMakesPy import decreasing, inclusive, raiseIfNone
from mapFolding import leafOrigin
from mapFolding._e import getDomainLeaf
from mapFolding._e.dataBaskets import StateElimination
from mapFolding._e.p2上nDimensional import (
	dimensionIndex, 一, 三, 二, 四, 工dimensionTail, 工dimension首一, 工dimension首三, 工dimension首二, 工dimension首零, 工totalDimensionsOdd, 零, 首一, 首一二, 首三,
	首二, 首零, 首零一, 首零一二, 首零二)
from mapFolding.beDRY import mapShapeIs2上nDimensions
from more_itertools import all_unique as allUnique吗, loops
from operator import add, sub
from typing import TYPE_CHECKING
from Z0Z_tools import consecutive吗, exclude, reverseLookup

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction
	from mapFolding.theTypes import Leaf, Pile

"""# DEVELOPMENT leaf domains are directly tied to mapShapeProductsSums and mapShape首ProductsSums

2d6
(0, 32, 48, 56, 60, 62, 63) = mapShape首ProductsSums
(0, 1, 3, 7, 15, 31, 63, 127) = mapShapeProductsSums

leaf descends from 63 in mapShape首ProductsSums
first pile is totalDimensions and ascends by addends in mapShapeProductsSums

leaf63 starts at pile6 = 6+0
leaf62 starts at pile7 = 6+1
leaf60 starts at pile10 = 7+3
leaf56 starts at pile17 = 10+7
leaf48 starts at pile32 = 17+15
leaf32 starts at pile63 = 32+31

2d5
mapShape首ProductsSums
(0, 16, 24, 28, 30, 31)

31, 5+0
30, 5+1
28, 6+3
24, 9+7
16, 16+15

mapShapeProductsSums
(0, 1, 3, 7, 15, 31, 63)

{
	0: [0],
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
	31: [16]
}
"""

"""products of dimensions and sums of products emerge from the formulas in `getDomainLeaf`.
state = StateElimination((2,) * 6)
domainsOfDimensionOrigins = tuple(getDomainLeaf(state, leaf) for leaf in state.mapShapeProducts)[0:-1]
sumsOfDimensionOrigins = tuple(accumulate(state.mapShapeProducts))[0:-1]
sumsOfDimensionOriginsReversed = tuple(accumulate(state.mapShapeProducts[::-1], initial=-state.totalLeaves))[1:None]
for dimensionOrigin, domain, sumOrigins, sumReversed in zip(state.mapShapeProducts, domainsOfDimensionOrigins, sumsOfDimensionOrigins, sumsOfDimensionOriginsReversed, strict=False):
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
def getDomainLeaf(state: StateElimination, leaf: int) -> range:
	def workhorse(leaf: int, totalDimensions: int, mapShape: tuple[int, ...], totalLeaves: int) -> range:
		originPinned =  leaf == leafOrigin
		return range(
					int(bit_flip(0, howMany0coordinatesAtTail(leaf) + 1))									# `start`, first value included in the `range`.
						+ 工totalDimensionsOdd(leaf)
						- 1 - originPinned
					, int(bit_mask(totalDimensions) ^ bit_mask(totalDimensions - 工dimension首零(leaf)))	# `stop`, first value excluded from the `range`.
						- 工totalDimensionsOdd(leaf)
						+ 2 - originPinned
					, 2 + (2 * (leaf == 首零(totalDimensions)+零))											# `step`
				)
	return workhorse(leaf, state.totalDimensions, state.mapShape, state.totalLeaves)
"""

def getDomainDimension一(state: StateElimination) -> tuple[tuple[int, int, int, int], ...]:
	"""Compute combined pile choices for the dimension-one leaves.

	(AI generated docstring)

	You can enumerate joint pile assignments for a fixed group of leaves in `state` by combining
	individual pile domains and coordinate restrictions. You obtain assignments from the map shape
	without inspecting `state.permutationSpace`.

	Parameters
	----------
	state : StateElimination
		Elimination state for a map whose axis lengths are all `2`. Supply enough dimensions for the
		selected leaves and validate the map shape yourself.

	Returns
	-------
	domainCombined : tuple[tuple[int, int, int, int], ...]
		Possible pile assignments in the leaf order described below. Each assignment
		contains a pile index for each selected leaf.

	Leaf Order
	----------
	The positional-numeral system [1] identifies the leaves
	in the following order.

		```python
		(一 + 零, 一, 首一(state.totalDimensions), 首零一(state.totalDimensions))
		```

	You obtain individual pile ranges from `mapFolding._e.leafDomains.getDomainLeaf` [2]. Check the
	combined assignments against other folding constraints.

	References
	----------
	[1] `mapFolding._e.p2上nDimensional.semiotics`

	[2] `mapFolding._e.leafDomains.getDomainLeaf`
	"""
	domain一零: tuple[int, ...] = tuple(getDomainLeaf(state, 一 + 零))
	domain首一: tuple[int, ...] = tuple(getDomainLeaf(state, 首一(state.totalDimensions)))
	return _getDomainDimension一(domain一零, domain首一, state.totalDimensions)
@cache
def _getDomainDimension一(domain一零: tuple[int, ...], domain首一: tuple[int, ...], totalDimensions: int) -> tuple[tuple[int, int, int, int], ...]:
	domainCombined: list[tuple[int, int, int, int]] = []

	for pileOfLeaf一零 in domain一零:
		domainOfLeaf首一: tuple[int, ...] = domain首一
		pilesTotal: int = len(domainOfLeaf首一)

		boxOfIndicesPilesExcluded: list[int] = []

		if pileOfLeaf一零 <= 首二(totalDimensions):
			pass

		elif 首二(totalDimensions) < pileOfLeaf一零 < 首一(totalDimensions):
			boxOfIndicesPilesExcluded.extend([*range(1, pilesTotal // 2), *range(1 + pilesTotal // 2, 3 * pilesTotal // 4)])

		elif pileOfLeaf一零 == 首一(totalDimensions):
			boxOfIndicesPilesExcluded.extend([*range(1, pilesTotal // 2)])

		elif 首一(totalDimensions) < pileOfLeaf一零 < 首零(totalDimensions) - 一:
			boxOfIndicesPilesExcluded.extend([*range(3 * pilesTotal // 4)])

		elif pileOfLeaf一零 == 首零(totalDimensions) - 一:
			boxOfIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4)])

		elif pileOfLeaf一零 == 首零(totalDimensions):
			boxOfIndicesPilesExcluded.extend([*range(2, pilesTotal // 2)])

		domainOfLeaf首一 = tuple(exclude(domainOfLeaf首一, boxOfIndicesPilesExcluded))

		domainCombined.extend([(pileOfLeaf一零, pileOfLeaf一零 + 1, pileOfLeaf首一, pileOfLeaf首一 + 1) for pileOfLeaf首一 in domainOfLeaf首一])

	return tuple(filter(allUnique吗, domainCombined))

def getDomainDimension二(state: StateElimination) -> tuple[tuple[int, int, int, int], ...]:
	"""Compute combined pile choices for the dimension-two leaves.

	(AI generated docstring)

	You can enumerate joint pile assignments for a fixed group of leaves in `state` by combining
	individual pile domains and coordinate restrictions. You obtain assignments from the map shape
	without inspecting `state.permutationSpace`.

	Parameters
	----------
	state : StateElimination
		Elimination state for a map whose axis lengths are all `2`. Supply enough dimensions for the
		selected leaves and validate the map shape yourself.

	Returns
	-------
	domainCombined : tuple[tuple[int, int, int, int], ...]
		Possible pile assignments in the leaf order described below. Each assignment
		contains a pile index for each selected leaf.

	Leaf Order
	----------
	The positional-numeral system [1] identifies the leaves
	in the following order.

		```python
		(二 + 一, 二 + 一 + 零, 二 + 零, 二)
		```

	You obtain individual pile ranges from `mapFolding._e.leafDomains.getDomainLeaf` [2]. Check the
	combined assignments against other folding constraints.

	References
	----------
	[1] `mapFolding._e.p2上nDimensional.semiotics`

	[2] `mapFolding._e.leafDomains.getDomainLeaf`
	"""
	domain二零and二: tuple[tuple[int, int], ...] = getDomain二零and二(state)
	domain二一零and二一: tuple[tuple[int, int], ...] = getDomain二一零and二一(state)
	return _getDomainDimension二(domain二零and二, domain二一零and二一, state.totalDimensions)
@cache
def _getDomainDimension二(domain二零and二: tuple[tuple[int, int], ...], domain二一零and二一: tuple[tuple[int, int], ...], totalDimensions: int) -> tuple[tuple[int, int, int, int], ...]:
	domain0corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive吗, domain二零and二))
	domain一corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive吗, domain二一零and二一))
	pilesTotal: int = len(domain一corners)

	domainCombined: list[tuple[int, int, int, int]] = []

	mapShapeProducts: tuple[int, ...] = tuple(int(bit_flip(0, dimension)) for dimension in range(totalDimensions + 1))

#======== By exclusion of the indices, add pairs of corners (160 tuples) ====================
	for 次, (pileOfLeaf二一零, pileOfLeaf二一) in enumerate(domain一corners):
		boxOfIndicesPilesExcluded: list[int] = []

		dimensionTail: int = 工dimensionTail(pileOfLeaf二一)

#-------- `excludeBelow` `次` ---------------------------------
		excludeBelow: int = 次
		boxOfIndicesPilesExcluded.extend(range(excludeBelow))

#-------- `excludeAbove` `次` ---------------------------------
		excludeAbove: int = pilesTotal
		if pileOfLeaf二一 <= 首一(totalDimensions):
			if dimensionTail == 1:
				excludeAbove = pilesTotal // 2 + 次
				if 工totalDimensionsOdd(pileOfLeaf二一) == 2:
					excludeAbove -= 1

				if (工totalDimensionsOdd(pileOfLeaf二一) == 1 and (2 < 工dimension首零(pileOfLeaf二一))):
					excludeAbove += 2

				if (工totalDimensionsOdd(pileOfLeaf二一) == 1
					and (工dimension首零(pileOfLeaf二一) - raiseIfNone(工dimension首一(pileOfLeaf二一)) < 2)
				):
					addend: int = mapShapeProducts[totalDimensions - 2] + 4
					excludeAbove = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))

			else:
				excludeAbove = 3 * pilesTotal // 4 + 2
				if 次 == 0:
					excludeAbove = 1
				elif 次 <= 2:
					addend = 三 + sum(mapShapeProducts[1:totalDimensions - 2])
					excludeAbove = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
		boxOfIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

#-------- Exclude "knock-out" indices ---------------------------------
		if pileOfLeaf二一 < 首一二(totalDimensions):
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
			if (dimensionTail < 3) and (2 < 工dimension首零(pileOfLeaf二一)):
				if 5 < totalDimensions:
					addend = 四
					start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
					stop: int = start + addend
					step: int = 2
					if (dimensionTail == 1) and (工dimension首零(pileOfLeaf二一) == 4):
						start += 2
						stop = start + 1
					if dimensionTail == 2:
						start += 3
						if 工dimension首零(pileOfLeaf二一) == 4:
							start -= 2
						stop = start + dimensionTail + inclusive
					if 工totalDimensionsOdd(pileOfLeaf二一) == 2:
						stop = start + 1
					boxOfIndicesPilesExcluded.extend([*range(start, stop, step)])
				if (((工dimension首零(pileOfLeaf二一) == 3) and (工totalDimensionsOdd(pileOfLeaf二一) == 1))
					or (工dimension首零(pileOfLeaf二一) - raiseIfNone(工dimension首一(pileOfLeaf二一)) == 3)):
					addend = pileOfLeaf二一
					start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
					stop = start + 2
					if dimensionTail == 2:
						start += 1
						stop += 1
					if 工dimension首零(pileOfLeaf二一) == 4:
						start += 3
						stop += 4
					step = 1
					boxOfIndicesPilesExcluded.extend([*range(start, stop, step)])
			if 工dimension首零(pileOfLeaf二一) == 2:
				addend = 三
				start = domain0corners.index((pileOfLeaf二一 + addend, pileOfLeaf二一零 + addend))
				boxOfIndicesPilesExcluded.extend([*range(start, start + addend, 2)])

		domainCombined.extend([(pileOfLeaf二一, pileOfLeaf二一零, pileOfLeaf二零, pileOfLeaf二) for pileOfLeaf二零, pileOfLeaf二 in exclude(domain0corners, boxOfIndicesPilesExcluded)])

#======== By inclusion of the piles, add non-corners (52 tuples) ====================
	domain一nonCorners: tuple[tuple[int, int], ...] = tuple(set(domain二一零and二一).difference(set(domain一corners)))
	domainCombined.extend([(pileOfLeaf一二, pileOfLeaf二一零, pileOfLeaf二一零 - 1, pileOfLeaf一二 + 1) for pileOfLeaf二一零, pileOfLeaf一二 in domain一nonCorners])

	return tuple(sorted(filter(allUnique吗, set(domainCombined))))

def getDomainDimension首二(state: StateElimination) -> tuple[tuple[int, int, int, int], ...]:
	"""Compute combined pile choices for the head-relative dimension-two leaves.

	(AI generated docstring)

	You can enumerate joint pile assignments for a fixed group of leaves in `state` by combining
	individual pile domains and coordinate restrictions. You obtain assignments from the map shape
	without inspecting `state.permutationSpace`.

	Parameters
	----------
	state : StateElimination
		Elimination state for a map whose axis lengths are all `2`. Supply enough dimensions for the
		selected leaves and validate the map shape yourself.

	Returns
	-------
	domainCombined : tuple[tuple[int, int, int, int], ...]
		Possible pile assignments in the leaf order described below. Each assignment
		contains a pile index for each selected leaf.

	Leaf Order
	----------
	The positional-numeral system [1] identifies the leaves
	in the following order.

		```python
		(首二(state.totalDimensions), 首零二(state.totalDimensions), 首零一二(state.totalDimensions), 首一二(state.totalDimensions))
		```

	You obtain individual pile ranges from `mapFolding._e.leafDomains.getDomainLeaf` [2]. Check the
	combined assignments against other folding constraints.

	References
	----------
	[1] `mapFolding._e.p2上nDimensional.semiotics`

	[2] `mapFolding._e.leafDomains.getDomainLeaf`
	"""
	domain首零二and首二: tuple[tuple[int, int], ...] = getDomain首零二and首二(state)
	domain首零一二and首一二: tuple[tuple[int, int], ...] = getDomain首零一二and首一二(state)
	return _getDomainDimension首二(state.totalDimensions, domain首零二and首二, domain首零一二and首一二)
@cache
def _getDomainDimension首二(totalDimensions: int, domain首零二and首二: tuple[tuple[int, int], ...], domain首零一二and首一二: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int, int, int], ...]:
	domain0corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive吗, domain首零二and首二))
	domain一corners: tuple[tuple[int, int], ...] = tuple(filter(consecutive吗, domain首零一二and首一二))
	pilesTotal: Leaf = len(domain一corners)

	domainCombined: list[tuple[int, int, int, int]] = []

#======== By exclusion of the indices, add pairs of corners (160 tuples) ====================
	for 次, (pileOfLeaf首零二, pileOfLeaf首二) in enumerate(domain0corners):
		boxOfIndicesPilesExcluded: list[int] = []

		dimensionTail: int = 工dimensionTail(pileOfLeaf首零二)

#-------- `excludeBelow` `次` ---------------------------------
		excludeBelow: int = 次 - 1
		boxOfIndicesPilesExcluded.extend(range(excludeBelow))

#-------- `excludeAbove` `次` ---------------------------------
		excludeAbove: int = pilesTotal
		if dimensionTail == 1:
			excludeAbove = (pilesTotal - (int((pileOfLeaf首二) ^ bit_mask(totalDimensions)) // 4 - 1))

			if 工totalDimensionsOdd(pileOfLeaf首二) == 3 and (totalDimensions - 工dimension首零(pileOfLeaf首二) >= 2):
				excludeAbove += 2

			if (工totalDimensionsOdd(pileOfLeaf首二) == 1
				and (totalDimensions - 工dimension首零(pileOfLeaf首二) >= 2)
				and (工dimension首零(pileOfLeaf首二) - raiseIfNone(工dimension首一(pileOfLeaf首二)) > 3)
			):
				excludeAbove += 2

			if (工totalDimensionsOdd(pileOfLeaf首二) == 1
				and (工dimension首零(pileOfLeaf首二) - raiseIfNone(工dimension首一(pileOfLeaf首二)) > 4)
			):
				excludeAbove += 2

			if ((工totalDimensionsOdd(pileOfLeaf首二) == totalDimensions - 工dimension首零(pileOfLeaf首二))
				and (4 <= 工dimension首零(pileOfLeaf首二))
				and (工totalDimensionsOdd(pileOfLeaf首二) > 1)
			):
				excludeAbove -= 1

		else:
			if 首零二(totalDimensions) <= pileOfLeaf首零二:
				excludeAbove = pilesTotal - 1
			if 首零(totalDimensions) < pileOfLeaf首零二 < 首零二(totalDimensions):
				excludeAbove = pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(totalDimensions)) // 8 - 1)
			if 首一二(totalDimensions) < pileOfLeaf首零二 <= 首零(totalDimensions):
				excludeAbove = pilesTotal - int(bit_mask(totalDimensions - 4))

			if pileOfLeaf首零二 == 首一二(totalDimensions):
				excludeAbove = pilesTotal - int(bit_mask(totalDimensions - 4)) - 1
			if pileOfLeaf首零二 < 首一二(totalDimensions):
				excludeAbove = pilesTotal - int(bit_mask(totalDimensions - 3)) - (dimensionTail == 2)
		boxOfIndicesPilesExcluded.extend(range(excludeAbove, pilesTotal))

#-------- Exclude "knock-out" indices ---------------------------------
		if dimensionTail == 1 and (abs(pileOfLeaf首零二 - 首零(totalDimensions)) == 2) and isEven吗(totalDimensions):
			boxOfIndicesPilesExcluded.extend([excludeAbove - 2])
		if dimensionTail != 1 and 首一二(totalDimensions) <= pileOfLeaf首零二 <= 首零一(totalDimensions):
			if (dimensionTail == 2) and (工totalDimensionsOdd(pileOfLeaf首零二) + 1 != 工dimension首零(pileOfLeaf首零二) - raiseIfNone(工dimension首一(pileOfLeaf首零二))):
				boxOfIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(totalDimensions)) // 8 + 2)])
				if (pileOfLeaf首零二 <= 首零(totalDimensions)) and isEven吗(totalDimensions):
					boxOfIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(totalDimensions)) // 4 - 1)])
			if dimensionTail == 3:
				boxOfIndicesPilesExcluded.extend([excludeAbove - 2])
			if 3 < dimensionTail:
				boxOfIndicesPilesExcluded.extend([pilesTotal - (int(pileOfLeaf首零二 ^ bit_mask(totalDimensions)) // 4)])

		domainCombined.extend([(pileOfLeaf首二, pileOfLeaf首零二, pileOfLeaf首零一二, pileOfLeaf首一二) for pileOfLeaf首零一二, pileOfLeaf首一二 in exclude(domain一corners, boxOfIndicesPilesExcluded)])

#======== By inclusion of the piles, add non-corners (52 tuples) ====================
	domain0nonCorners: tuple[tuple[int, int], ...] = tuple(set(domain首零二and首二).difference(set(domain0corners)))
	domainCombined.extend([(pileOfLeaf首二, pileOfLeaf首零二, pileOfLeaf首零二 - 1, pileOfLeaf首二 + 1) for pileOfLeaf首零二, pileOfLeaf首二 in domain0nonCorners])

	return tuple(sorted(filter(allUnique吗, set(domainCombined))))

def getDomain二零and二(state: StateElimination) -> tuple[tuple[int, int], ...]:
	"""Compute combined pile choices for the leaves five and four.

	(AI generated docstring)

	You can enumerate joint pile assignments for a fixed group of leaves in `state` by combining
	individual pile domains and coordinate restrictions. You obtain assignments from the map shape
	without inspecting `state.permutationSpace`.

	Parameters
	----------
	state : StateElimination
		Elimination state for a map whose axis lengths are all `2`. Supply enough dimensions for the
		selected leaves and validate the map shape yourself.

	Returns
	-------
	domainCombined : tuple[tuple[int, int], ...]
		Possible pile assignments in the leaf order described below. Each assignment
		contains a pile index for each selected leaf.

	Leaf Order
	----------
	The positional-numeral system [1] identifies the leaves
	in the following order.

		```python
		(二 + 零, 二)
		```

	You obtain individual pile ranges from `mapFolding._e.leafDomains.getDomainLeaf` [2]. Check the
	combined assignments against other folding constraints.

	References
	----------
	[1] `mapFolding._e.p2上nDimensional.semiotics`

	[2] `mapFolding._e.leafDomains.getDomainLeaf`
	"""
	domain二零: tuple[int, ...] = tuple(getDomainLeaf(state, 二 + 零))
	domain二: tuple[int, ...] = tuple(getDomainLeaf(state, 二))
	direction: CallableFunction[[int, int], int] = add
	return _getDomains二Or二一(domain二零, domain二, direction, state.totalDimensions, state.mapShapeProductsSums)

def getDomain二一零and二一(state: StateElimination) -> tuple[tuple[int, int], ...]:
	"""Compute combined pile choices for the leaves seven and six.

	(AI generated docstring)

	You can enumerate joint pile assignments for a fixed group of leaves in `state` by combining
	individual pile domains and coordinate restrictions. You obtain assignments from the map shape
	without inspecting `state.permutationSpace`.

	Parameters
	----------
	state : StateElimination
		Elimination state for a map whose axis lengths are all `2`. Supply enough dimensions for the
		selected leaves and validate the map shape yourself.

	Returns
	-------
	domainCombined : tuple[tuple[int, int], ...]
		Possible pile assignments in the leaf order described below. Each assignment
		contains a pile index for each selected leaf.

	Leaf Order
	----------
	The positional-numeral system [1] identifies the leaves
	in the following order.

		```python
		(二 + 一 + 零, 二 + 一)
		```

	You obtain individual pile ranges from `mapFolding._e.leafDomains.getDomainLeaf` [2]. Check the
	combined assignments against other folding constraints.

	References
	----------
	[1] `mapFolding._e.p2上nDimensional.semiotics`

	[2] `mapFolding._e.leafDomains.getDomainLeaf`
	"""
	domain二一零: tuple[int, ...] = tuple(getDomainLeaf(state, 二 + 一 + 零))
	domain二一: tuple[int, ...] = tuple(getDomainLeaf(state, 二 + 一))
	direction: CallableFunction[[int, int], int] = sub
	return _getDomains二Or二一(domain二一零, domain二一, direction, state.totalDimensions, state.mapShapeProductsSums)

@cache
def _getDomains二Or二一(domain零: tuple[int, ...], domain0: tuple[int, ...], direction: CallableFunction[[int, int], int], totalDimensions: int, mapShapeProductsSums: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
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

	for 次Domain零, pileOfLeaf零 in enumerate(filter((首零(totalDimensions) - 零).__ge__, domain零)):
		indicesDomain0ToExclude: list[int] = []

		dimensionTail: int = 工dimensionTail(pileOfLeaf零 - isOdd吗(pileOfLeaf零))

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
			if pileOfLeaf零 == (首一(totalDimensions) + 首二(totalDimensions) + 首三(totalDimensions)):
				次Domain0: int = int(7 * pilesTotal / 8)
				次Domain0 -= pilesFewerDomain0
				indicesDomain0ToExclude.extend([次Domain0])
# ******* end *******

#-------- `excludeBelow` `次` ---------------------------------
		excludeBelow: int = 次Domain零 + excludeBelowAddend
		excludeBelow -= pilesFewerDomain0
		indicesDomain0ToExclude.extend(range(excludeBelow))

#-------- `excludeAbove` `次` ---------------------------------
		if pileOfLeaf零 <= 首一(totalDimensions):
			excludeAbove: int = 次Domain零 + (3 * pilesTotal // 4)
			excludeAbove -= pilesFewerDomain0
			indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))
		if 首一(totalDimensions) < pileOfLeaf零 < 首零(totalDimensions):
			excludeAbove = int(pileOfLeaf零 ^ bit_mask(totalDimensions)) // 2
			indicesDomain0ToExclude.extend(range(excludeAbove, pilesTotal))

#-------- Exclude by stepping: exclude ((2^dimensionTail - 1) / (2^dimensionTail))-many indices, e.g., 1/2, 3/4, 15/16, after `次` -----------------
		for dimension in range(dimensionTail):
			indicesDomain0ToExclude.extend(range(steppingBasisForUnknownReasons + int(bit_mask(dimension)), pilesTotal, int(bit_flip(0, dimension + 1))))

#-------- Exclude "knock-out" indices ---------------------------------
		if dimensionTail == 1:
			if (首二(totalDimensions) < pileOfLeaf零 < 首零(totalDimensions) - 零) and (2 < 工dimension首零(pileOfLeaf零)):
				if 工dimension首一(pileOfLeaf零) == 零:
					次Domain0: int = pilesTotal // 2
					次Domain0 -= pilesFewerDomain0
					if 4 < domain0[次Domain0].bit_length():
						indicesDomain0ToExclude.extend([次Domain0])
					if 首一(totalDimensions) < pileOfLeaf零:
						次Domain0 = -(pilesTotal // 4 - isOdd吗(pileOfLeaf零))
						次Domain0 -= -(pilesFewerDomain0)
						indicesDomain0ToExclude.extend([次Domain0])
				if 工dimension首一(pileOfLeaf零) == 一:
					次Domain0 = pilesTotal // 2 + 2
					次Domain0 -= pilesFewerDomain0
					if domain0[次Domain0] < 首零(totalDimensions):
						indicesDomain0ToExclude.extend([次Domain0])
					次Domain0 = -(pilesTotal // 4 - 2)
					次Domain0 -= -(pilesFewerDomain0)
					if 首一(totalDimensions) < pileOfLeaf零:
						indicesDomain0ToExclude.extend([次Domain0])

				if 工dimension首一(pileOfLeaf零) == 一 + 零:
					次Domain0 = -(pilesTotal // 4)
					次Domain0 -= -(pilesFewerDomain0)
					indicesDomain0ToExclude.extend([次Domain0])

				次Domain0 = 3 * pilesTotal // 4
				次Domain0 -= pilesFewerDomain0
				# DEVELOPMENT b/c 首一二(totalDimensions)
				if pileOfLeaf零 < 首一二(totalDimensions):
					dimensionIndexPart首: int = totalDimensions
					dimensionIndexPart一: int = dimensionIndex(一)
					dimensionIndexPart二: int = dimensionIndex(二)

					# Compute the 次 from the head `首`
					次MapShapeProductsSums: int = dimensionIndexPart首 - (dimensionIndexPart一 + dimensionIndexPart二)

					addend: int = mapShapeProductsSums[次MapShapeProductsSums]
					if ImaDomain二一零and二一:
						addend -= 1  # decreasing?
					pileOfLeaf0: int = addend + 首零(totalDimensions)
					次Domain0 = domain0.index(pileOfLeaf0)

					indicesDomain0ToExclude.extend([次Domain0])

				if 工dimension首二(pileOfLeaf零) == 零:
					if 工dimension首一(pileOfLeaf零) == 一 + 零:
						indicesDomain0ToExclude.extend([次Domain0 - 2])
					if 工dimension首零(pileOfLeaf零) == 一 + 零:
						indicesDomain0ToExclude.extend([次Domain0 - 2])

		elif 首一(totalDimensions) + 首三(totalDimensions) + isOdd吗(pileOfLeaf零) == pileOfLeaf零:
			次Domain0 = (3 * pilesTotal // 4) - 1
			次Domain0 -= pilesFewerDomain0
			indicesDomain0ToExclude.extend([次Domain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, indicesDomain0ToExclude)])

#======== By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

	return tuple(sorted(set(domainCombined)))

def getDomain首零二and首二(state: StateElimination) -> tuple[tuple[int, int], ...]:
	"""Compute combined pile choices for the head-relative zero-two and two leaves.

	(AI generated docstring)

	You can enumerate joint pile assignments for a fixed group of leaves in `state` by combining
	individual pile domains and coordinate restrictions. You obtain assignments from the map shape
	without inspecting `state.permutationSpace`.

	Parameters
	----------
	state : StateElimination
		Elimination state for a map whose axis lengths are all `2`. Supply enough dimensions for the
		selected leaves and validate the map shape yourself.

	Returns
	-------
	domainCombined : tuple[tuple[int, int], ...]
		Possible pile assignments in the leaf order described below. Each assignment
		contains a pile index for each selected leaf.

	Leaf Order
	----------
	The positional-numeral system [1] identifies the leaves
	in the following order.

		```python
		(首零二(state.totalDimensions), 首二(state.totalDimensions))
		```

	You obtain individual pile ranges from `mapFolding._e.leafDomains.getDomainLeaf` [2]. Check the
	combined assignments against other folding constraints.

	References
	----------
	[1] `mapFolding._e.p2上nDimensional.semiotics`

	[2] `mapFolding._e.leafDomains.getDomainLeaf`
	"""
	domain首零二: tuple[int, ...] = tuple(getDomainLeaf(state, 首零二(state.totalDimensions)))
	domain首二: tuple[int, ...] = tuple(getDomainLeaf(state, 首二(state.totalDimensions)))
	return _getDomain首零二and首二(domain首零二, domain首二, state.totalDimensions)
@cache
def _getDomain首零二and首二(domain首零二: tuple[int, ...], domain首二: tuple[int, ...], totalDimensions: int) -> tuple[tuple[int, int], ...]:
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
		if pileOfLeaf零 < 首零(totalDimensions) + 零:
			continue
		boxOfIndicesPilesExcluded: list[int] = []

		dimensionTail: int = 工dimensionTail(direction(pileOfLeaf零, isOdd吗(pileOfLeaf零)))

#-------- `excludeBelow` `次` ---------------------------------
		if 首零一(totalDimensions) < pileOfLeaf零:
			excludeBelow: int = 次 + 3 - (3 * pilesTotal // 4)
		else:
			excludeBelow = 2 + (首零一(totalDimensions) - direction(pileOfLeaf零, isOdd吗(pileOfLeaf零))) // 2
		excludeBelow -= pilesFewerDomain0
		boxOfIndicesPilesExcluded.extend(range(excludeBelow))

#-------- `excludeAbove` `次` ------------------------------
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
			if (工dimension首二(pileOfLeaf零) == 一) and (二 + 零 <= 工dimension首零(pileOfLeaf零)):
				次Domain0: int = (pilesTotal // 2) + 1
				次Domain0 -= pilesFewerDomain0
				boxOfIndicesPilesExcluded.extend([次Domain0])
				次Domain0: int = (pilesTotal // 4) + 1
				次Domain0 -= pilesFewerDomain0
				boxOfIndicesPilesExcluded.extend([次Domain0])
				if pileOfLeaf零 < 首零一(totalDimensions):
					boxOfIndicesPilesExcluded.extend([次Domain0 - 2])

			if 工totalDimensionsOdd(pileOfLeaf零) == 一:
				次Domain0 = (pilesTotal // 4) + 3
				次Domain0 -= pilesFewerDomain0
				if 工dimension首一(pileOfLeaf零) == 一:
					boxOfIndicesPilesExcluded.extend([次Domain0])
				if 工dimension首一(pileOfLeaf零) == 二:
					boxOfIndicesPilesExcluded.extend([次Domain0])
				if (((工dimension首零(pileOfLeaf零) == totalDimensions - 1) and (工dimension首一(pileOfLeaf零) == totalDimensions - 3))
					or (工dimension首一(pileOfLeaf零) == 二)):
					boxOfIndicesPilesExcluded.extend([次Domain0 - 2])
					次Domain0 = (pilesTotal // 2) - 1
					次Domain0 -= pilesFewerDomain0
					boxOfIndicesPilesExcluded.extend([次Domain0])

		elif (首零一(totalDimensions) - direction(首三(totalDimensions), isOdd吗(pileOfLeaf零))) == pileOfLeaf零:
			次Domain0 = (pilesTotal // 4) + 2
			次Domain0 -= pilesFewerDomain0
			boxOfIndicesPilesExcluded.extend([次Domain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, boxOfIndicesPilesExcluded)])

	return tuple(sorted(set(domainCombined)))

def getDomain首零一二and首一二(state: StateElimination) -> tuple[tuple[int, int], ...]:
	"""Compute combined pile choices for the head-relative zero-one-two and one-two leaves.

	(AI generated docstring)

	You can enumerate joint pile assignments for a fixed group of leaves in `state` by combining
	individual pile domains and coordinate restrictions. You obtain assignments from the map shape
	without inspecting `state.permutationSpace`.

	Parameters
	----------
	state : StateElimination
		Elimination state for a map whose axis lengths are all `2`. Supply enough dimensions for the
		selected leaves and validate the map shape yourself.

	Returns
	-------
	domainCombined : tuple[tuple[int, int], ...]
		Possible pile assignments in the leaf order described below. Each assignment
		contains a pile index for each selected leaf.

	Leaf Order
	----------
	The positional-numeral system [1] identifies the leaves
	in the following order.

		```python
		(首零一二(state.totalDimensions), 首一二(state.totalDimensions))
		```

	You obtain individual pile ranges from `mapFolding._e.leafDomains.getDomainLeaf` [2]. Check the
	combined assignments against other folding constraints.

	References
	----------
	[1] `mapFolding._e.p2上nDimensional.semiotics`

	[2] `mapFolding._e.leafDomains.getDomainLeaf`
	"""
	domain首零一二: tuple[int, ...] = tuple(getDomainLeaf(state, 首零一二(state.totalDimensions)))
	domain首一二: tuple[int, ...] = tuple(getDomainLeaf(state, 首一二(state.totalDimensions)))
	direction: CallableFunction[[int, int], int] = add
	return _getDomain首零一二and首一二(domain首零一二, domain首一二, direction, state.totalDimensions)
@cache
def _getDomain首零一二and首一二(domain零: tuple[int, ...], domain0: tuple[int, ...], direction: CallableFunction[[int, int], int], totalDimensions: int) -> tuple[tuple[int, int], ...]:
	domainCombined: list[tuple[int, int]] = []

#======== By exclusion of the indices, add non-consecutive piles (54 pairs) ====================
	pilesTotal: int = len(domain零)
	pilesFewerDomain0: int = pilesTotal - len(domain0)

	for 次Domain零, pileOfLeaf零 in enumerate(domain零):
		if pileOfLeaf零 < 首零(totalDimensions):
			continue
		indicesDomain0ToExclude: list[int] = []

		dimensionTail: int = 工dimensionTail(direction(pileOfLeaf零, isOdd吗(pileOfLeaf零)))

#-------- `excludeBelow` `次` ---------------------------------
		if 首零一(totalDimensions) < pileOfLeaf零:
			excludeBelow: int = 次Domain零 + 1 - (3 * pilesTotal // 4)
		else:
			excludeBelow = (首零一(totalDimensions) - direction(pileOfLeaf零, isOdd吗(pileOfLeaf零))) // 2
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
			if (工dimension首二(pileOfLeaf零) == 一) and (二 + 零 <= 工dimension首零(pileOfLeaf零)):
				次Domain0: int = pilesTotal // 2
				次Domain0 -= pilesFewerDomain0
				indicesDomain0ToExclude.extend([次Domain0])
				次Domain0: int = pilesTotal // 4
				次Domain0 -= pilesFewerDomain0
				indicesDomain0ToExclude.extend([次Domain0])
				if pileOfLeaf零 < 首零一(totalDimensions):
					indicesDomain0ToExclude.extend([次Domain0 - 2])
			if 工dimension首二(pileOfLeaf零) == 一 + 零:
				次Domain0 = pilesTotal // 4
				次Domain0 -= pilesFewerDomain0
				if 工dimension首三(pileOfLeaf零) == 一:
					indicesDomain0ToExclude.extend([次Domain0])
			if 工totalDimensionsOdd(pileOfLeaf零) == 一:
				次Domain0 = (pilesTotal // 4) + 2
				次Domain0 -= pilesFewerDomain0
				if 工dimension首一(pileOfLeaf零) == 一:
					次Domain0 = domain0.index(首零(totalDimensions) - 一)
					indicesDomain0ToExclude.extend([次Domain0])
				if 工dimension首一(pileOfLeaf零) == 二:
					indicesDomain0ToExclude.extend([次Domain0])
				if (首零二(totalDimensions) < pileOfLeaf零) and (二 + 零 <= 工dimension首零(pileOfLeaf零)):
					indicesDomain0ToExclude.extend([次Domain0 - 2])
					次Domain0 = (pilesTotal // 2) - 2
					次Domain0 -= pilesFewerDomain0
					indicesDomain0ToExclude.extend([次Domain0])

		elif (首零一(totalDimensions) - direction(首三(totalDimensions), isOdd吗(pileOfLeaf零))) == pileOfLeaf零:
			次Domain0 = (pilesTotal // 4) + 1
			次Domain0 -= pilesFewerDomain0
			indicesDomain0ToExclude.extend([次Domain0])

		domainCombined.extend([(pileOfLeaf零, pileOfLeaf0) for pileOfLeaf0 in exclude(domain0, indicesDomain0ToExclude)])

#======== By inclusion of the piles, add consecutive piles (22 pairs)  ====================
	domainCombined.extend([(pile, direction(pile, 零)) for pile in domain零 if direction(pile, 零) in domain0])

	return tuple(sorted(set(domainCombined)))

@cache
def _getDomainLeaf(leaf: Leaf, totalDimensions: int, mapShape: tuple[int, ...], totalLeaves: int) -> range:
	"""Supply `dimensionLength == 2` for these subroutines. You may be able to extend the concept to other `mapShape` values."""
	state: StateElimination = StateElimination(mapShape)
	if mapShapeIs2上nDimensions(state.mapShape):
		originPinned: bool = leaf == leafOrigin
		return range(
					state.mapShapeProductsSums[工dimensionTail(leaf) + inclusive]  	# `start`, first value included in the `range`.
						+ 工totalDimensionsOdd(leaf)
						- originPinned

					, state.mapShape首ProductsSums[工dimension首零(leaf)]  			# `stop`, first value excluded from the `range`.
						+ 2
						- 工totalDimensionsOdd(leaf)
						- originPinned

					, 2 + (2 * (leaf == 首零(totalDimensions) + 零))				# `step`
				)
	return range(totalLeaves)

def getDomainLeaf首零Plus零(state: StateElimination, leaf: Leaf | None = None) -> tuple[Pile, ...]:
	"""Compute pile choices for the head-plus-one leaf using pinned neighbors.

	(AI generated docstring)

	You can retrieve the pile domain of the special head-plus-one leaf in `state`. When both neighboring
	reference leaves are pinned in `state.permutationSpace`, you receive a domain narrowed by their
	pinned pile positions. Otherwise, you receive the complete individual domain.

	Parameters
	----------
	state : StateElimination
		Elimination state that provides the map shape and dimension count.
	leaf : Leaf | None = None
		Leaf index to query. `None` selects the special head-plus-one leaf. Supply that same leaf
		when using the restrictions based on pinned neighbors.

	Returns
	-------
	domain首零Plus零 : tuple[Pile, ...]
		Possible pile indices, with conditional exclusions when both reference leaves are
		pinned. You leave `state` unchanged.

	Reference Leaves
	----------------
	The default leaf is `零 + 首零(state.totalDimensions)`.
	The reference leaves are `一 + 零` and `首零一(state.totalDimensions)`, using the coordinate
	vocabulary [1]. The initial domain comes from `mapFolding._e.leafDomains.getDomainLeaf` [2].

	References
	----------
	[1] `mapFolding._e.p2上nDimensional.semiotics`

	[2] `mapFolding._e.leafDomains.getDomainLeaf`
	"""
	if leaf is None:
		leaf = (零) + 首零(state.totalDimensions)
	domain首零Plus零: tuple[Pile, ...] = tuple(getDomainLeaf(state, leaf))
	leaf一零: Leaf = 一 + 零
	leaf首零一: Leaf = 首零一(state.totalDimensions)
	if state.permutationSpace.leafPinned吗(leaf一零) and state.permutationSpace.leafPinned吗(leaf首零一):
		pileOfLeaf一零: Pile = raiseIfNone(reverseLookup(state.permutationSpace, leaf一零))
		pileOfLeaf首零一: Pile = raiseIfNone(reverseLookup(state.permutationSpace, leaf首零一))
		domain首零Plus零 = _getDomainLeaf首零Plus零(domain首零Plus零, pileOfLeaf一零, pileOfLeaf首零一, state.totalDimensions, state.totalLeaves)
	return domain首零Plus零
@cache
def _getDomainLeaf首零Plus零(domain首零Plus零: tuple[Pile, ...], pileOfLeaf一零: Pile, pileOfLeaf首零一: Pile, totalDimensions: int, totalLeaves: int) -> tuple[Pile, ...]:
	pilesTotal: int = 首一(totalDimensions)

	bump: int = 1 - int(pileOfLeaf一零.bit_count() == 1)
	howMany: int = totalDimensions - (pileOfLeaf一零.bit_length() + bump)
	onesInBinary: int = int(bit_mask(howMany))
	ImaPattern: int = pilesTotal - onesInBinary

	boxOfIndicesPilesExcluded: list[int] = []
	if pileOfLeaf一零 == 二:
		boxOfIndicesPilesExcluded.extend([零, 一, 二])  # These symbols make this pattern jump out.

	if 二 < pileOfLeaf一零 <= 首二(totalDimensions):
		stop: int = pilesTotal // 2 - 1
		boxOfIndicesPilesExcluded.extend(range(1, stop))

		aDimensionPropertyNotFullyUnderstood: int = 5
		for _dimension in loops(totalDimensions - aDimensionPropertyNotFullyUnderstood):
			start: int = 1 + stop
			stop += (stop + 1) // 2
			boxOfIndicesPilesExcluded.extend([*range(start, stop)])

		boxOfIndicesPilesExcluded.extend([*range(1 + stop, ImaPattern)])

	if 首二(totalDimensions) < pileOfLeaf一零:
		boxOfIndicesPilesExcluded.extend([*range(1, ImaPattern)])

	bump = 1 - int((totalLeaves - pileOfLeaf首零一).bit_count() == 1)
	howMany = totalDimensions - ((totalLeaves - pileOfLeaf首零一).bit_length() + bump)
	onesInBinary = int(bit_mask(howMany))
	ImaPattern = pilesTotal - onesInBinary

	aDimensionPropertyNotFullyUnderstood = 5

	if pileOfLeaf首零一 == totalLeaves - 二:
		boxOfIndicesPilesExcluded.extend([-零 - 1, -(一) - 1])
		if aDimensionPropertyNotFullyUnderstood <= totalDimensions:
			boxOfIndicesPilesExcluded.extend([-二 - 1])

	if ((首零一二(totalDimensions) < pileOfLeaf首零一 < totalLeaves - 二)
		and (首二(totalDimensions) < pileOfLeaf一零 <= 首零(totalDimensions))):
		boxOfIndicesPilesExcluded.extend([-1])

	if 首零一二(totalDimensions) <= pileOfLeaf首零一 < totalLeaves - 二:
		stop: int = pilesTotal // 2 - 1
		boxOfIndicesPilesExcluded.extend(range((1 + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing))

		for _dimension in loops(totalDimensions - aDimensionPropertyNotFullyUnderstood):
			start: int = 1 + stop
			stop += (stop + 1) // 2
			boxOfIndicesPilesExcluded.extend([*range((start + inclusive) * decreasing, (stop + inclusive) * decreasing, decreasing)])

		boxOfIndicesPilesExcluded.extend([*range((1 + stop + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])

		if 二 <= pileOfLeaf一零 <= 首零(totalDimensions):
			boxOfIndicesPilesExcluded.extend([零, 一, 二, pilesTotal // 2])

	if ((pileOfLeaf首零一 == 首零一二(totalDimensions))
		and (首一(totalDimensions) < pileOfLeaf一零 <= 首零(totalDimensions))):
		boxOfIndicesPilesExcluded.extend([-1])

	if 首零一(totalDimensions) < pileOfLeaf首零一 < 首零一二(totalDimensions):
		if pileOfLeaf一零 in {首一(totalDimensions), 首零(totalDimensions)}:
			boxOfIndicesPilesExcluded.extend([-1])
		elif 二 < pileOfLeaf一零 < 首二(totalDimensions):
			boxOfIndicesPilesExcluded.extend([0])

	if pileOfLeaf首零一 < 首零一二(totalDimensions):
		boxOfIndicesPilesExcluded.extend([*range((1 + inclusive) * decreasing, (ImaPattern + inclusive) * decreasing, decreasing)])

	pileOfLeaf一零ARCHETYPICAL: int = 首一(totalDimensions)
	bump = 1 - int(pileOfLeaf一零ARCHETYPICAL.bit_count() == 1)
	howMany = totalDimensions - (pileOfLeaf一零ARCHETYPICAL.bit_length() + bump)
	onesInBinary = int(bit_mask(howMany))
	ImaPattern = pilesTotal - onesInBinary

	if pileOfLeaf首零一 == totalLeaves - 二:
		if pileOfLeaf一零 == 二:
			boxOfIndicesPilesExcluded.extend([零, 一, 二, pilesTotal // 2 - 1, pilesTotal // 2])
		if 二 < pileOfLeaf一零 <= 首零(totalDimensions):
			IDK: int = ImaPattern - 1
			boxOfIndicesPilesExcluded.extend([*range(1, 3 * pilesTotal // 4), *range(1 + 3 * pilesTotal // 4, IDK)])
		if 首一(totalDimensions) < pileOfLeaf一零 <= 首零(totalDimensions):
			boxOfIndicesPilesExcluded.extend([-1])

	if pileOfLeaf首零一 == 首零一(totalDimensions):
		if pileOfLeaf一零 == 首零(totalDimensions):
			boxOfIndicesPilesExcluded.extend([-1])
		elif (二 < pileOfLeaf一零 < 首二(totalDimensions)) or (首二(totalDimensions) < pileOfLeaf一零 < 首一(totalDimensions)):
			boxOfIndicesPilesExcluded.extend([0])

	return tuple(exclude(domain首零Plus零, boxOfIndicesPilesExcluded))
