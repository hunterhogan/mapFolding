# ruff: noqa ERA001
from collections.abc import Callable, Iterator
from copy import deepcopy
from gmpy2 import bit_mask, bit_test, is_even, is_odd
from hunterMakesPy import raiseIfNone
from itertools import chain, repeat
from mapFolding import exclude, Z0Z_key
from mapFolding._e import (
	coordinatesOf0AtTail, decreasing, dimensionNearest首, dimensionSecondNearest首, fullRange, getDictionaryAddends4Next,
	getDictionaryAddends4Prior, getDictionaryPileToLeaves, getLeafDomain, leaf0, leafSubHyperplane, origin, ptount, 一, 三,
	二, 零, 首一, 首一二, 首二, 首零, 首零一, 首零一二, 首零二)
from mapFolding._e.patternFinder import getExcludingDictionary
from mapFolding.algorithms.iff import pinnedLeavesHasAViolation
from mapFolding.dataBaskets import EliminationState
from math import log, prod
from more_itertools import extract, interleave_longest, pairwise
from operator import add

def listPinnedLeavesDefault(state: EliminationState) -> EliminationState:
	state.listPinnedLeaves = [{origin: leaf0, 零: 零, state.leavesTotal - 零: 首零(state.dimensionsTotal)}]
	return state

# ======= append `pinnedLeaves` at `pile` if qualified =======

def appendPinnedLeavesAtPile(state: EliminationState, listLeavesAtPile: list[int]) -> EliminationState:
	for leaf in listLeavesAtPile:
		if _disqualifyAppendingLeafAtPile(state, leaf):
			continue
		if Z0Z_disqualifyDictionary(state, leaf):
			continue

		# NOTE handle beans and cornbread.
		if leaf in [一+零, 零, 首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)]:
			stateCopy: EliminationState = deepcopy(state)
			stateCopy.pinnedLeaves[stateCopy.pile] = leaf
			if leaf in [一+零, 首一(state.dimensionsTotal)]:
				stateCopy.pile += 1
				getDictionary: Callable[[EliminationState], dict[int, list[int]]] = getDictionaryAddends4Next
			else:
				stateCopy.pile -= 1
				getDictionary = getDictionaryAddends4Prior
			leafCornbread: int = addendsToListLeavesAtPile(getDictionary(stateCopy)[leaf], leaf, []).pop()
			if _disqualifyAppendingLeafAtPile(stateCopy, leafCornbread):
				continue
			if Z0Z_disqualifyDictionary(stateCopy, leafCornbread):
				continue
			pinnedLeaves: dict[int, int] = stateCopy.pinnedLeaves.copy()
			pinnedLeaves[stateCopy.pile] = leafCornbread

		else:
			pinnedLeaves = state.pinnedLeaves.copy()
			pinnedLeaves[state.pile] = leaf

		state.listPinnedLeaves.append(pinnedLeaves.copy())

	return state

def _disqualifyAppendingLeafAtPile(state: EliminationState, leaf: int) -> bool:
		return any([
			_alreadyPinned(state, leaf)
			, _tooSmall(state, leaf)
			, _tooLarge(state, leaf)
			, _pileNotInRange(state, leaf)
			, _pileOccupied(state, leaf)
			# , Z0Z_TESTexcluding2d5(state, leaf)
		])

def Z0Z_TESTexcluding2d5(state: EliminationState, leaf: int) -> bool:

	for leafExcluder in range(state.leavesTotal):
		excludingDictionary: dict[int, dict[int, list[int]]] | None = getExcludingDictionary(state, leafExcluder)
		if excludingDictionary is None:
			continue

		if leaf != leafExcluder:
			if leafExcluder in state.pinnedLeaves.values():
				pileExcluder: int = Z0Z_key(state.pinnedLeaves, leafExcluder)
				if pileExcluder in excludingDictionary:
					dictionaryIndicesPilesExcluded: dict[int, list[int]] = excludingDictionary[pileExcluder]
					if leaf in dictionaryIndicesPilesExcluded:
						listIndicesPilesExcluded: list[int] = dictionaryIndicesPilesExcluded[leaf]
						domainOfPilesForLeaf: list[int] = list(getLeafDomain(state, leaf))
						listPilesExcluded: list[int] = list(extract(domainOfPilesForLeaf, listIndicesPilesExcluded))
						if state.pile in listPilesExcluded:
							return True

	return False

def _alreadyPinned(state: EliminationState, leaf: int) -> bool:
	return leaf in state.pinnedLeaves.values()

def _pileNotInRange(state: EliminationState, leaf: int) -> bool:
	return state.pile not in list(getLeafDomain(state, leaf))

def _pileOccupied(state: EliminationState, leaf: int) -> bool:
	return state.pile in state.pinnedLeaves

def _tooLarge(state: EliminationState, leaf: int) -> bool:
	return leaf > state.leavesTotal - 零

def _tooSmall(state: EliminationState, leaf: int) -> bool:
	return leaf < 0

# ======= Dictionaries. =======

def Z0Z_disqualifyDictionary(state: EliminationState, leaf: int) -> bool:
		stateCopy: EliminationState = deepcopy(state)
		stateCopy.pinnedLeaves[stateCopy.pile] = leaf
		return any([
			_productsOfDimensionsNotInOrder(state, leaf)
			, _dimensionOriginNotFirst(state, leaf)
			, Z0Z_kNotBefore_rLeaf(state, leaf)
			# , Z0Z_kNotBefore_rDictionary(stateCopy.pinnedLeaves)
			, pinnedLeavesHasAViolation(stateCopy)
		])

def _dimensionOriginNotFirst(state: EliminationState, leaf: int) -> bool:
	listDimensionOrigins:  list[int] = [prod(state.mapShape[0:dimension], start=1) for dimension in range(零, state.dimensionsTotal + fullRange)]

	dimensionOrigin: int = leaf
	pileOfDimensionOrigin: int = state.pile
	if dimensionOrigin in listDimensionOrigins:
		for pile, k in state.pinnedLeaves.items():
			if (dimensionOrigin < k) and (k % dimensionOrigin == 0) and (pile < pileOfDimensionOrigin):
				return True

	for dimensionOrigin in listDimensionOrigins:
		if (leaf != dimensionOrigin) and (leaf % dimensionOrigin == 0):
			pilesOpenBeforeLeaf: set[int] = set(range(state.pile)).difference(state.pinnedLeaves.keys())
			if dimensionOrigin in state.pinnedLeaves.values():
				pileOfDimensionOrigin = Z0Z_key(state.pinnedLeaves, dimensionOrigin)
				if state.pile < pileOfDimensionOrigin:
					return True
			else:
				pilesOpenBeforeLeaf = pilesOpenBeforeLeaf.intersection(getLeafDomain(state, dimensionOrigin))
				if not pilesOpenBeforeLeaf:
					return True
				else:
					pilesOpenBeforeLeaf.remove(min(pilesOpenBeforeLeaf))

	return False

def _productsOfDimensionsNotInOrder(state: EliminationState, leaf: int) -> bool:
	# _exclude_rBefore_k
	productsOfDimensions:  list[int] = [prod(state.mapShape[0:dimension], start=1) for dimension in range(state.dimensionsTotal + fullRange)]

	r: int = leaf
	pileOf_r: int = state.pile
	if r in productsOfDimensions[一:None]:
		pilesOpenBefore_r: set[int] = set(range(pileOf_r)).difference(state.pinnedLeaves.keys())
		dimensionIndexOf_r: int = productsOfDimensions.index(r)
		for k in productsOfDimensions[一:dimensionIndexOf_r]:
			if k in state.pinnedLeaves.values():
				pileOf_k: int = Z0Z_key(state.pinnedLeaves, k)
				if pileOf_r < pileOf_k:
					return True
			else:
				pilesOpenBefore_r = pilesOpenBefore_r.intersection(getLeafDomain(state, k))
				if not pilesOpenBefore_r:
					return True
				else:
					pilesOpenBefore_r.remove(min(pilesOpenBefore_r))

	k: int = leaf
	pileOf_k = state.pile
	if k in productsOfDimensions[origin:-(一)]:
		pilesOpenAfter_k: set[int] = set(range(pileOf_k, state.leavesTotal)).difference(state.pinnedLeaves.keys())
		dimensionIndexOf_k: int = productsOfDimensions.index(k)
		for r in productsOfDimensions[dimensionIndexOf_k + 零: -(一)]:
			if r in state.pinnedLeaves.values():
				pileOf_r = Z0Z_key(state.pinnedLeaves, r)
				if pileOf_r < pileOf_k:
					return True
			else:
				pilesOpenAfter_k = pilesOpenAfter_k.intersection(getLeafDomain(state, r))
				if not pilesOpenAfter_k:
					return True
				else:
					pilesOpenAfter_k.remove(max(pilesOpenAfter_k))

	return False

def rr(tupleElement: tuple[tuple[int, int], tuple[int, int]]) -> bool:
	return (1 < tupleElement[0][1]) and (1 < tupleElement[1][1])

def ff(tupleElement: tuple[tuple[int, int], tuple[int, int]]) -> bool:
	return dimensionNearest首(tupleElement[0][1]) <= coordinatesOf0AtTail(tupleElement[1][1])

def Z0Z_kNotBefore_rLeaf(state: EliminationState, leaf: int) -> bool:
	qq: Iterator[tuple[tuple[int, int], tuple[int, int]]] = chain(zip(repeat((state.pile, leaf)), sorted(state.pinnedLeaves.items())), zip(sorted(state.pinnedLeaves.items()), repeat((state.pile, leaf))))
	return any((pileOf_k > pileOf_r for (pileOf_k, _k), (pileOf_r, _r) in filter(ff, filter(rr, qq))))

def Z0Z_kNotBefore_rDictionary(pinnedLeaves: dict[int, int]) -> bool:
	"""
	leaf1 is a dimension origin: its addends up to [-1], which equate to leaves 3, 5, 9, 17, come before the dimension origins, 2, 4, 8, 16.

	This is due to:
	leaf	{dimension: increase}
	0			{0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32} <- dimension origins
	1			{1: 3, 2: 5, 3: 9, 4: 17, 5: 33}

	If leaf2 were before leaf3, it would interpose the crease from leaf1 to leaf3 in dimension1.

	Similarly, leaf2 addends up to [-1], which equate to leaves 6, 10, 18 come before dimension origins, 4, 8, 16.
	2			{0: 3, 2: 6, 3: 10, 4: 18, 5: 34}

	The rule against interposing is so strong it extends to leaf3, which is not a dimension origin, but is the first increase from leaf1.
	leaf3 addends up to [-1], which equate to leaves 7, 11, 19, come before the dimension origins, 4, 8, 16.
	3			{2: 7, 3: 11, 4: 19, 5: 35}

	leaf4 is the dimension2 origin and its increases 12 and 20 come before dimension origins 8 and 16.
	4			{0: 5, 1: 6, 3: 12, 4: 20, 5: 36}

	leaf5, 0b101, 二 + 零, which absolutely has the coordinates of 1 in dimension2, 二, and 1 in dimension0, 零, comes before all multiples of 4.

	leaf6, 二 + 一, is the same as leaf5.

	leaf7, 二 + 一 + 零, is also the same as leaf5 and leaf6!

	leaf9, 三 + 零, comes before the dimension3 origin leaf8, as described above, and before all multiples of 8, or 三.

	Furthermore, all leaves between 三+零 and 三+二+一+零, inclusive, come before 三 (8) and its multiples.

	The same thing happens at the next dimension, 四. leaves 17-31 all come before 16, 32, and 48. This example is a 6 dimensional
	map. Because all leaves less than 32 must come before leaf32, it cannot appear before pile 32. It's fixed at the last pile, of course.

	wow.
	"""
	qq: Iterator[tuple[tuple[int, int], tuple[int, int]]] = pairwise(sorted(pinnedLeaves.items()))
	return any((pileOf_k > pileOf_r for (pileOf_k, _k), (pileOf_r, _r) in filter(ff, filter(rr, qq))))


# ======= Subroutines for analyzing a specific `pile`. =======

def addendsToListLeavesAtPile(listAddends: list[int], leafAddend: int, listIndicesExcluded: list[int]) -> list[int]:
	return list(map(add, repeat(leafAddend), exclude(listAddends, listIndicesExcluded)))

def pinPileOriginFixed(state: EliminationState) -> list[int]:
	listLeavesAtPile: list[int] = [leaf0]
	return listLeavesAtPile

def pinPile零Fixed(state: EliminationState) -> list[int]:
	listLeavesAtPile: list[int] = [零]
	return listLeavesAtPile

def pinPile一Addend(state: EliminationState) -> list[int]:
	listAddendIndicesExcluded: list[int] = []
	leafAtPileLess1: int = state.pinnedLeaves[state.pile - 1]
	leafAt首Less一: int | None = state.pinnedLeaves.get(state.leavesTotal - 一)
	if leafAt首Less一 and (0 < coordinatesOf0AtTail(leafAt首Less一)):
		listAddendIndicesExcluded.extend([*range(coordinatesOf0AtTail(leafAt首Less一) - 1, state.dimensionsTotal - 2)])
	return addendsToListLeavesAtPile(getDictionaryAddends4Next(state)[leafAtPileLess1], leafAtPileLess1, listAddendIndicesExcluded)

def pinPile一零Addend(state: EliminationState) -> list[int]:
	listAddendIndicesExcluded: list[int] = []
	listAddendIndicesExcluded.append(0)
	leafAtPileLess1: int = state.pinnedLeaves[state.pile - 1]
	leafAt首Less一: int = state.pinnedLeaves[state.leavesTotal - 一]
	if is_even(leafAt首Less一) and (leafAtPileLess1 == 首零(state.dimensionsTotal)+零):
		listAddendIndicesExcluded.extend([*range(coordinatesOf0AtTail(leafAt首Less一) + 1, state.dimensionsTotal)])
	return addendsToListLeavesAtPile(getDictionaryAddends4Next(state)[leafAtPileLess1], leafAtPileLess1, listAddendIndicesExcluded)

def pinPile二Addend(state: EliminationState) -> list[int]:
	listAddendIndicesExcluded: list[int] = []
	leafAtPileLess1: int = state.pinnedLeaves[state.pile - 1]
	if is_odd(leafAtPileLess1):
		listAddendIndicesExcluded.extend([*range(leafAtPileLess1.bit_length() - 1, 5), ptount(leafAtPileLess1)])
	leafAt首Less一: int = state.pinnedLeaves[state.leavesTotal - 一]
	if is_even(leafAtPileLess1) and is_even(leafAt首Less一):
		listAddendIndicesExcluded.extend([*range(coordinatesOf0AtTail(leafSubHyperplane(leafAt首Less一)) - 一, (state.dimensionsTotal - 3))])
	if is_odd(leafAtPileLess1):
		listAddendIndicesExcluded.append((int(log(leafSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5)
	leafAt首Less一零: int = state.pinnedLeaves[state.leavesTotal - (一+零)]
	if is_even(leafAtPileLess1) and leafAt首Less一零:
		listAddendIndicesExcluded.extend([*range(state.dimensionsTotal - 3)][(state.dimensionsTotal - 3) - ((state.dimensionsTotal - 2) - leafSubHyperplane(leafAt首Less一零 - (leafAt首Less一零.bit_count() - is_even(leafAt首Less一零))).bit_count()) % (state.dimensionsTotal - 2) - is_even(leafAt首Less一零): None])  # noqa: E501
	leafAt一: int = state.pinnedLeaves[一]
	if (leafAt一 == 首零(state.dimensionsTotal)+零):
		listAddendIndicesExcluded.extend([(int(log(leafSubHyperplane(leafAt首Less一), state.mapShape[0])) + 4) % 5, coordinatesOf0AtTail(leafAt首Less一零) - 1])
	if (leafAt一 == 首零(state.dimensionsTotal)+零) and (leafAt首Less一零 > 首零(state.dimensionsTotal)+零):
		listAddendIndicesExcluded.extend([*range(int(leafAt首Less一零 - 2**(leafAt首Less一零.bit_length() - 1)).bit_length() - 1, state.dimensionsTotal - 2)])
	if ((leafAt一 == 首零(state.dimensionsTotal)+零) and (0 < leafAtPileLess1 - leafAt一 <= 2**(state.dimensionsTotal - 4)) and (0 < (leafAt首Less一 - leafAtPileLess1) <= 2**(state.dimensionsTotal - 3))):
		listAddendIndicesExcluded.extend([ptount(leafAtPileLess1), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
	return addendsToListLeavesAtPile(getDictionaryAddends4Next(state)[leafAtPileLess1], leafAtPileLess1, listAddendIndicesExcluded)

def pinPile首零Less零Leaf(state: EliminationState) -> list[int]:
	productsOfDimensions:		list[int] = [prod(state.mapShape[0:dimension]) for dimension in range(state.dimensionsTotal + 1)]
	sumsProductsOfDimensions:	list[int] = [sum(productsOfDimensions[0:dimension]) for dimension in range(state.dimensionsTotal + 1)]

	dictionaryPileToLeaves: dict[int, list[int]] = getDictionaryPileToLeaves(state)
	listRemoveLeaves: list[int] = []

	pileExcluder: int = 一
	leafAtPileExcluder: int = state.pinnedLeaves[pileExcluder]
	for dimension in range(state.dimensionsTotal):
		if dimension < state.dimensionsTotal - 2:
			leaf: int = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([一, 首零(state.dimensionsTotal) + leafAtPileExcluder])
		if 0 < dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([一 + leafAtPileExcluder])
		if dimension == 1:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + leafAtPileExcluder + 零])
		if dimension == state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首一(state.dimensionsTotal), 首一(state.dimensionsTotal) + leafAtPileExcluder])
	del pileExcluder

	pileExcluder = state.leavesTotal - 一
	leafAtPileExcluder = state.pinnedLeaves[pileExcluder]
	for dimension in range(state.dimensionsTotal):
		if dimension == 0:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([一])
		if dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首一(state.dimensionsTotal) + leafAtPileExcluder])
		if 0 < dimension < state.dimensionsTotal - 2:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([2**dimension, 首一(state.dimensionsTotal) + leafAtPileExcluder - (2**dimension - 零)])
		if 0 < dimension < state.dimensionsTotal - 3:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([零 + leafAtPileExcluder])
		if 0 < dimension < state.dimensionsTotal - 1:
			leaf = dictionaryPileToLeaves[pileExcluder][dimension]
			if leaf == leafAtPileExcluder:
				listRemoveLeaves.extend([首一(state.dimensionsTotal)])
	del pileExcluder

	"""pileExcluder = 一+零
	(2,) * 6:
	21      [28]    False   [2, 4, 8, 19, 21, 25, 35, 49, 52, 56]   [2, 4, 8, 19, 21, 25, 28, 35, 49, 52, 56]
	25      [59]    False   [2, 8, 19, 25, 41, 49, 56]      [2, 8, 19, 25, 41, 49, 56, 59]
	"""
	pileExcluder = 一+零
	leafAtPileExcluder = state.pinnedLeaves[pileExcluder]

	if leafAtPileExcluder == 三+二+零:
		listRemoveLeaves.extend([二+一+零, 首零(state.dimensionsTotal)+二+零])

	if leafAtPileExcluder == 首一(state.dimensionsTotal)+二+零:
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首一二(state.dimensionsTotal)+零, 首零一二(state.dimensionsTotal)])

	if leafAtPileExcluder == 首一二(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零二(state.dimensionsTotal)+零])

	if leafAtPileExcluder == 首零一(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])

	if is_odd(leafAtPileExcluder):
		listRemoveLeaves.extend([leafAtPileExcluder, productsOfDimensions[raiseIfNone(dimensionSecondNearest首(leafAtPileExcluder))]])

		if leafAtPileExcluder < 首零(state.dimensionsTotal):
			comebackOffset: int = sumsProductsOfDimensions[ptount(leafAtPileExcluder) + 1]
			listRemoveLeaves.extend([
				一
				, leafAtPileExcluder + 首零(state.dimensionsTotal)-零
				, leafAtPileExcluder + 首零(state.dimensionsTotal)-零 - comebackOffset
			])
			if ptount(leafAtPileExcluder) == 1:
				listRemoveLeaves.extend([
					productsOfDimensions[dimensionNearest首(leafAtPileExcluder)] + comebackOffset
					, 首零(state.dimensionsTotal) + comebackOffset
				])

		if 首零(state.dimensionsTotal) < leafAtPileExcluder:
			listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, productsOfDimensions[dimensionNearest首(leafAtPileExcluder) - 1]])

	del pileExcluder

	"""pileExcluder = state.leavesTotal - (一+零)
	(2,) * 5:
	22      [28]    False   [2, 4, 8, 19, 21, 22, 25, 26]   [2, 4, 8, 19, 21, 22, 25, 26, 28]

	38      [21]    False   [2, 4, 16, 35, 37, 38, 49, 50]  [2, 4, 16, 21, 35, 37, 38, 49, 50]
	42      [37]    False   [2, 4, 14, 16, 35, 38, 41, 42, 49, 50]  [2, 4, 14, 16, 35, 37, 38, 41, 42, 49, 50]
	"""
	pileExcluder = state.leavesTotal - (一+零)
	leafAtPileExcluder = state.pinnedLeaves[pileExcluder]
	if 首零(state.dimensionsTotal) < leafAtPileExcluder:
		listRemoveLeaves.extend([首零一(state.dimensionsTotal)+零, leafAtPileExcluder])

		if is_even(leafAtPileExcluder):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)])
			bit = 1
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
				listRemoveLeaves.extend([state.leavesTotal - sum(productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 2
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
				if 1 < coordinatesOf0AtTail(leafAtPileExcluder):
					listRemoveLeaves.extend([state.leavesTotal - sum(productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 3
			if bit_test(leafAtPileExcluder, bit):
				if 1 < coordinatesOf0AtTail(leafAtPileExcluder):
					listRemoveLeaves.extend([2**bit])
					listRemoveLeaves.extend([state.leavesTotal - sum(productsOfDimensions[bit: state.dimensionsTotal - 2])])
				if coordinatesOf0AtTail(leafAtPileExcluder) < bit:
					listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])

			sheepOrGoat = 0

			shepherdOfDimensions: int = 2**(state.dimensionsTotal - 5)
			if (leafAtPileExcluder//shepherdOfDimensions) & bit_mask(5) == 0b10101:
				listRemoveLeaves.extend([0b000100])
				sheepOrGoat = ptount(leafAtPileExcluder//shepherdOfDimensions)
				if 0 < sheepOrGoat < state.dimensionsTotal - 3:
					comebackOffset = 2**dimensionNearest首(leafAtPileExcluder) - 0b100
					listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])
				if 0 < sheepOrGoat < state.dimensionsTotal - 4:
					comebackOffset = 2**raiseIfNone(dimensionSecondNearest首(leafAtPileExcluder)) - 0b100
					listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])

		if is_odd(leafAtPileExcluder):
			listRemoveLeaves.extend([一])
			if leafAtPileExcluder & bit_mask(4) == 0b001001:
				listRemoveLeaves.extend([0b001011])
			sheepOrGoat = ptount(leafAtPileExcluder)
			if 0 < sheepOrGoat < state.dimensionsTotal - 3:
				comebackOffset = 2**dimensionNearest首(leafAtPileExcluder) - 0b10
				listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])
			if 0 < sheepOrGoat < state.dimensionsTotal - 4:
				comebackOffset = 2**raiseIfNone(dimensionSecondNearest首(leafAtPileExcluder)) - 0b10
				listRemoveLeaves.extend([leafAtPileExcluder - comebackOffset])

	pileExcluder = 二
	leafAtPileExcluder = state.pinnedLeaves[pileExcluder]

	if is_even(leafAtPileExcluder):
		listRemoveLeaves.extend([一, leafAtPileExcluder + 1, 首零(state.dimensionsTotal)+一+零])
	if is_odd(leafAtPileExcluder):
		listRemoveLeaves.extend([leafAtPileExcluder - 1])
		if 首一(state.dimensionsTotal) < leafAtPileExcluder < 首零(state.dimensionsTotal):
			listRemoveLeaves.extend([首一(state.dimensionsTotal)+一+零, 首零一(state.dimensionsTotal)+零])
		if 首零(state.dimensionsTotal) < leafAtPileExcluder:
			listRemoveLeaves.extend([首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)+零])
			bit = 1
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 2
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 3
			if bit_test(leafAtPileExcluder, bit):
				listRemoveLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 零])
			bit = 4
			if bit_test(leafAtPileExcluder, bit) and (leafAtPileExcluder.bit_length() > 5):
				listRemoveLeaves.extend([首零一二(state.dimensionsTotal)])
	del pileExcluder

	""" # ------- Tools for creating rules ------------------------
	listRemoveLeaves: list[int] = []
	dictionaryExcludedLeaves = getExcludedLeaves(state, state.pile, (pileExcluder,))
	if print1Time < 1:
		print1Time += 1
		pprint(dictionaryExcludedLeaves, width=140)

	listExcludedLeavesGoal = dictionaryExcludedLeaves[leafAtPileExcluder]
	print(leafAtPileExcluder, sorted(set(listExcludedLeavesGoal).difference(set(listRemoveLeaves)))
		, listExcludedLeavesGoal == sorted(set(listRemoveLeaves)), sorted(set(listRemoveLeaves)), listExcludedLeavesGoal, sep='\t')
	"""

	leafAt一: int = state.pinnedLeaves[一]
	leafAt首Less一: int = state.pinnedLeaves[state.leavesTotal - 一]
	leafAt一零: int = state.pinnedLeaves[一+零]
	leafAt首Less一零: int = state.pinnedLeaves[state.leavesTotal - (一+零)]

	if (leafAt一零 != 首零一(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.append(一)
	if (leafAt首Less一零 != 首零(state.dimensionsTotal)+零 + getDictionaryAddends4Prior(state)[首零(state.dimensionsTotal)+零][0]) and (leafAt一 == 一+零):
		listRemoveLeaves.append(首一(state.dimensionsTotal))

	if (leafAt一 == 首二(state.dimensionsTotal)+零) and (leafAt首Less一 == 首零一(state.dimensionsTotal)):
		listRemoveLeaves.extend([首二(state.dimensionsTotal), 首零一二(state.dimensionsTotal)])
	if leafAt一 == 首零(state.dimensionsTotal)+零:
		listRemoveLeaves.extend([首一(state.dimensionsTotal), leafAt首Less一 + 零])
	if leafAt一.bit_length() < state.dimensionsTotal - 2:
		listRemoveLeaves.extend([一, leafAt首Less一 + 一])

	""" # ------- Tools for creating rules ------------------------
	tuplePilesExcluders = (一, state.leavesTotal - 一)
	(leavesAtPilesExcluders) = (leafAt一, leafAt首Less一)
	listRemoveLeaves: list[int] = []
	dictionaryExcludedLeaves = getExcludedLeaves(state, state.pile, tuplePilesExcluders)
	if print1Time < 1:
		print1Time += 1
		pprint(dictionaryExcludedLeaves, width=140)
	listExcludedLeavesGoal = dictionaryExcludedLeaves[((leavesAtPilesExcluders))]
	print(leavesAtPilesExcluders, sorted(set(listExcludedLeavesGoal).difference(set(listRemoveLeaves)))
		, listExcludedLeavesGoal == sorted(set(listRemoveLeaves)), sorted(set(listRemoveLeaves)), listExcludedLeavesGoal, sep='\t')

	"""
	return sorted(set(dictionaryPileToLeaves[state.pile]).difference(set(listRemoveLeaves)))

def pinPile首Less一Addend(state: EliminationState) -> list[int]:
	listAddendIndicesExcluded: list[int] = []
	leafAtPilePlus1: int = state.pinnedLeaves[state.pile + 1]
	leafAt一: int | None = state.pinnedLeaves.get(一)
	if leafAt一 and (leafAt一.bit_length() < state.dimensionsTotal):
		listAddendIndicesExcluded.extend([*range(零, leafAt一.bit_length())])
	return addendsToListLeavesAtPile(getDictionaryAddends4Prior(state)[leafAtPilePlus1], leafAtPilePlus1, listAddendIndicesExcluded)

def pinPile首Less一零Addend(state: EliminationState) -> list[int]:
	listAddendIndicesExcluded: list[int] = []
	leafAtPilePlus1: int = state.pinnedLeaves[state.pile + 1]
	if leafAtPilePlus1 < 首零一(state.dimensionsTotal):
		listAddendIndicesExcluded.append(-1)
	leafAt一: int = state.pinnedLeaves[一]
	if (leafAtPilePlus1 == 首零(state.dimensionsTotal)+零) and (leafAt一 != 一+零):
		listAddendIndicesExcluded.extend([*range(leafAt一.bit_length() - 2)])
	return addendsToListLeavesAtPile(getDictionaryAddends4Prior(state)[leafAtPilePlus1], leafAtPilePlus1, listAddendIndicesExcluded)

def pinPile首Less零Fixed(state: EliminationState) -> list[int]:
	listLeavesAtPile: list[int] = [首零(state.dimensionsTotal)]
	return listLeavesAtPile

# ======= Flow control ===============================================

def nextPinnedLeavesWorkbench(state: EliminationState, pileProcessingOrder: list[int] | None = None, queueStopBefore: int | None = None) -> EliminationState:
	if pileProcessingOrder is None:
		pileProcessingOrder = pileProcessingOrderDefault(state)

	state.pinnedLeaves = {}

	for pile in pileProcessingOrder:
		if pile == queueStopBefore:
			break
		if not all(pile in pinnedLeaves for pinnedLeaves in state.listPinnedLeaves):
			state.pinnedLeaves = next(pinnedLeaves.copy() for pinnedLeaves in state.listPinnedLeaves if pile not in pinnedLeaves)
			state.listPinnedLeaves.remove(state.pinnedLeaves)
			state = whereNext(state, pileProcessingOrder)
			break
	return state

def pileProcessingOrderDefault(state: EliminationState) -> list[int]:
	pileProcessingOrder: list[int] = [origin, 零, state.leavesTotal - 零]
	pileProcessingOrder.extend([一, state.leavesTotal - 一])
	pileProcessingOrder.extend(interleave_longest(range(一, 首零(state.dimensionsTotal)), range(state.leavesTotal - (一+零), 首零(state.dimensionsTotal) + decreasing, decreasing)))
	return pileProcessingOrder

def whereNext(state: EliminationState, pileProcessingOrder: list[int] | None = None) -> EliminationState:
	if pileProcessingOrder is None:
		pileProcessingOrder = pileProcessingOrderDefault(state)
	state.pile = next(pile for pile in pileProcessingOrder if pile not in state.pinnedLeaves)
	return state
