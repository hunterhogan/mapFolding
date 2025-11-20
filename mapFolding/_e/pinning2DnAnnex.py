# ruff: noqa ERA001
from collections.abc import Callable
from copy import deepcopy  # noqa: TC003
from gmpy2 import bit_mask, bit_test, is_even, is_odd
from hunterMakesPy import raiseIfNone
from itertools import repeat
from mapFolding import exclude
from mapFolding._e import (
	coordinatesOf0AtTail, decreasing, dimensionNearest首, dimensionSecondNearest首, getDictionaryAddends4Next,
	getDictionaryAddends4Prior, getDictionaryPileToIndexLeaves, getIndexLeafDomain, indexLeaf0, indexLeafSubHyperplane,
	origin, ptount, 一, 零, 首一, 首零, 首零一)
from mapFolding._e.patternFinder import getExcludingDictionary, numeralOfLengthInBase
from mapFolding.algorithms.iff import pinnedLeavesHasAViolation
from mapFolding.dataBaskets import EliminationState
from math import log, prod
from more_itertools import extract, interleave_longest
from operator import add

def listPinnedLeavesDefault(state: EliminationState) -> EliminationState:
	state.listPinnedLeaves = [{origin: indexLeaf0, 零: 零, state.leavesTotal - 零: 首零(state.dimensionsTotal)}]
	return state

# ======= append `pinnedLeaves` at `pile` if qualified =======

def appendPinnedLeavesAtPile(state: EliminationState, listIndexLeavesAtPile: list[int]) -> EliminationState:
	for indexLeaf in listIndexLeavesAtPile:
		if _disqualifyAppendingIndexLeafAtPile(state, indexLeaf):
			continue

		# NOTE handle beans and cornbread.
		if indexLeaf in [一+零, 零, 首一(state.dimensionsTotal), 首零一(state.dimensionsTotal)]:
			stateCopy: EliminationState = deepcopy(state)
			stateCopy.pinnedLeaves[stateCopy.pile] = indexLeaf
			if indexLeaf in [一+零, 首一(state.dimensionsTotal)]:
				stateCopy.pile += 1
				getDictionary: Callable[[EliminationState], dict[int, list[int]]] = getDictionaryAddends4Next
			else:
				stateCopy.pile -= 1
				getDictionary = getDictionaryAddends4Prior
			indexLeafCornbread: int = addendsToListIndexLeavesAtPile(getDictionary(stateCopy)[indexLeaf], indexLeaf, []).pop()
			if _disqualifyAppendingIndexLeafAtPile(stateCopy, indexLeafCornbread):
				continue
			pinnedLeaves: dict[int, int] = stateCopy.pinnedLeaves.copy()
			pinnedLeaves[stateCopy.pile] = indexLeafCornbread

		else:
			pinnedLeaves = state.pinnedLeaves.copy()
			pinnedLeaves[state.pile] = indexLeaf

		state.listPinnedLeaves.append(pinnedLeaves.copy())

	return state

def _disqualifyAppendingIndexLeafAtPile(state: EliminationState, indexLeaf: int) -> bool:
		return any([
			_alreadyPinned(state, indexLeaf)
			, _tooSmall(state, indexLeaf)
			, _tooLarge(state, indexLeaf)
			, _pileNotInRange(state, indexLeaf)
			, _pileOccupied(state, indexLeaf)
			, _exclude_rBefore_k(state, indexLeaf)
			, pinnedLeavesHasAViolation(state, indexLeaf)
			# , Z0Z_TESTexcluding2d5(state, indexLeaf)
		])

def Z0Z_TESTexcluding2d5(state: EliminationState, indexLeaf: int) -> bool:

	for indexLeafExcluder in range(state.leavesTotal):
		excludingDictionary: dict[int, dict[int, list[int]]] | None = getExcludingDictionary(state, indexLeafExcluder)
		if excludingDictionary is None:
			continue

		if indexLeaf != indexLeafExcluder:
			if indexLeafExcluder in state.pinnedLeaves.values():
				pileExcluder: int = {indexLeaf: pile for pile, indexLeaf in state.pinnedLeaves.items()}[indexLeafExcluder]
				if pileExcluder in excludingDictionary:
					dictionaryIndicesPilesExcluded: dict[int, list[int]] = excludingDictionary[pileExcluder]
					if indexLeaf in dictionaryIndicesPilesExcluded:
						listIndicesPilesExcluded: list[int] = dictionaryIndicesPilesExcluded[indexLeaf]
						domainOfPilesForIndexLeaf: list[int] = list(getIndexLeafDomain(state, indexLeaf))
						listPilesExcluded: list[int] = list(extract(domainOfPilesForIndexLeaf, listIndicesPilesExcluded))
						if state.pile in listPilesExcluded:
							return True

	return False

def _alreadyPinned(state: EliminationState, indexLeaf: int) -> bool:
	return indexLeaf in state.pinnedLeaves.values()

def _tooSmall(state: EliminationState, indexLeaf: int) -> bool:
	return indexLeaf < 0

def _tooLarge(state: EliminationState, indexLeaf: int) -> bool:
	return indexLeaf > state.leavesTotal - 零

def _pileNotInRange(state: EliminationState, indexLeaf: int) -> bool:
	return state.pile not in list(getIndexLeafDomain(state, indexLeaf))

def _pileOccupied(state: EliminationState, indexLeaf: int) -> bool:
	return state.pile in state.pinnedLeaves

def _exclude_rBefore_k(state: EliminationState, indexLeaf: int) -> bool:

	productsOfDimensions:  list[int] = [prod(state.mapShape[0:dimension], start=1) for dimension in range(state.dimensionsTotal + 1)]

	indexLeaf2Pile: dict[int, int] = {indexLeaf: pile for pile, indexLeaf in state.pinnedLeaves.items()}

	r: int = indexLeaf
	pileOf_r: int = state.pile
	if r in productsOfDimensions[一:None]:
		pilesOpenBefore_r: set[int] = set(range(pileOf_r)).difference(state.pinnedLeaves.keys())
		dimensionIndexOf_r: int = productsOfDimensions.index(r)
		for k in productsOfDimensions[一:dimensionIndexOf_r]:
			if k in indexLeaf2Pile:
				pileOf_k: int = indexLeaf2Pile[k]
				if pileOf_r < pileOf_k:
					print(f"{k=}, {r=}, {pileOf_k=}, {pileOf_r=}")
					return True
			else:
				pilesOpenBefore_r = pilesOpenBefore_r.intersection(getIndexLeafDomain(state, k))
				if not pilesOpenBefore_r:
					return True
				else:
					pilesOpenBefore_r.remove(min(pilesOpenBefore_r))

	k: int = indexLeaf
	pileOf_k = state.pile
	if k in productsOfDimensions[0:-(一)]:
		pilesOpenAfter_k: set[int] = set(range(pileOf_k, state.leavesTotal)).difference(state.pinnedLeaves.keys())
		dimensionIndexOf_k: int = productsOfDimensions.index(k)
		for r in productsOfDimensions[dimensionIndexOf_k + 1: -(一)]:
			if r in indexLeaf2Pile:
				pileOf_r = indexLeaf2Pile[r]
				if pileOf_r < pileOf_k:
					print(f"{k=}, {r=}, {pileOf_k=}, {pileOf_r=}")
					return True
			else:
				pilesOpenAfter_k = pilesOpenAfter_k.intersection(getIndexLeafDomain(state, r))
				if not pilesOpenAfter_k:
					return True
				else:
					pilesOpenAfter_k.remove(max(pilesOpenAfter_k))

	return False

# ======= Subroutines for analyzing some specific `pile`. =======

def addendsToListIndexLeavesAtPile(listAddends: list[int], indexLeafAddend: int, listIndicesExcluded: list[int]) -> list[int]:
	return list(map(add, repeat(indexLeafAddend), exclude(listAddends, listIndicesExcluded)))

def pinPileOriginFixed(state: EliminationState) -> list[int]:
	listIndexLeavesAtPile: list[int] = [indexLeaf0]
	return listIndexLeavesAtPile

def pinPile零Fixed(state: EliminationState) -> list[int]:
	listIndexLeavesAtPile: list[int] = [零]
	return listIndexLeavesAtPile

def pinPile一Addend(state: EliminationState) -> list[int]:
	ordinal: Callable[[int | list[int], str, int | list[int]], int] = numeralOfLengthInBase(positions=state.dimensionsTotal, base=state.mapShape[0])
	listAddendIndicesExcluded: list[int] = []
	indexLeafAtPileLess1: int = state.pinnedLeaves[state.pile - 1]
	indexLeafAt11ones0: int | None = state.pinnedLeaves.get(ordinal([1,1],'1',0))
	if indexLeafAt11ones0 and (0 < coordinatesOf0AtTail(indexLeafAt11ones0)):
		listAddendIndicesExcluded.extend([*range(coordinatesOf0AtTail(indexLeafAt11ones0) - 1, state.dimensionsTotal - 2)])
	return addendsToListIndexLeavesAtPile(getDictionaryAddends4Next(state)[indexLeafAtPileLess1], indexLeafAtPileLess1, listAddendIndicesExcluded)

def pinPile一零Addend(state: EliminationState) -> list[int]:
	ordinal: Callable[[int | list[int], str, int | list[int]], int] = numeralOfLengthInBase(positions=state.dimensionsTotal, base=state.mapShape[0])
	listAddendIndicesExcluded: list[int] = []
	listAddendIndicesExcluded.append(0)
	indexLeafAtPileLess1: int = state.pinnedLeaves[state.pile - 1]
	indexLeafAt11ones0: int = state.pinnedLeaves[ordinal([1,1],'1',0)]
	if is_even(indexLeafAt11ones0) and (indexLeafAtPileLess1 == ordinal([1,0],'0',1)):
		listAddendIndicesExcluded.extend([*range(coordinatesOf0AtTail(indexLeafAt11ones0) + 1, state.dimensionsTotal)])
	return addendsToListIndexLeavesAtPile(getDictionaryAddends4Next(state)[indexLeafAtPileLess1], indexLeafAtPileLess1, listAddendIndicesExcluded)

def pinPile二Addend(state: EliminationState) -> list[int]:
	ordinal: Callable[[int | list[int], str, int | list[int]], int] = numeralOfLengthInBase(positions=state.dimensionsTotal, base=state.mapShape[0])
	listAddendIndicesExcluded: list[int] = []
	indexLeafAtPileLess1: int = state.pinnedLeaves[state.pile - 1]
	if is_odd(indexLeafAtPileLess1):
		listAddendIndicesExcluded.extend([*range(indexLeafAtPileLess1.bit_length() - 1, 5), ptount(indexLeafAtPileLess1)])
	indexLeafAt11ones0: int = state.pinnedLeaves[ordinal([1,1],'1',0)]
	if is_even(indexLeafAtPileLess1) and is_even(indexLeafAt11ones0):
		listAddendIndicesExcluded.extend([*range(coordinatesOf0AtTail(indexLeafSubHyperplane(indexLeafAt11ones0)) - 一, (state.dimensionsTotal - 3))])
	if is_odd(indexLeafAtPileLess1):
		listAddendIndicesExcluded.append((int(log(indexLeafSubHyperplane(indexLeafAt11ones0), state.mapShape[0])) + 4) % 5)
	indexLeafAt11ones01: int = state.pinnedLeaves[ordinal([1,1],'1',[0,1])]
	if is_even(indexLeafAtPileLess1) and indexLeafAt11ones01:
		listAddendIndicesExcluded.extend([*range(state.dimensionsTotal - 3)][(state.dimensionsTotal - 3) - ((state.dimensionsTotal - 2) - indexLeafSubHyperplane(indexLeafAt11ones01 - (indexLeafAt11ones01.bit_count() - is_even(indexLeafAt11ones01))).bit_count()) % (state.dimensionsTotal - 2) - is_even(indexLeafAt11ones01): None])  # noqa: E501
	indexLeafAt__10: int = state.pinnedLeaves[一]
	if (indexLeafAt__10 == ordinal([1,0],'0',1)):
		listAddendIndicesExcluded.extend([(int(log(indexLeafSubHyperplane(indexLeafAt11ones0), state.mapShape[0])) + 4) % 5, coordinatesOf0AtTail(indexLeafAt11ones01) - 1])
	if (indexLeafAt__10 == ordinal([1,0],'0',1)) and (indexLeafAt11ones01 > ordinal([1,0],'0',1)):
		listAddendIndicesExcluded.extend([*range(int(indexLeafAt11ones01 - 2**(indexLeafAt11ones01.bit_length() - 1)).bit_length() - 1, state.dimensionsTotal - 2)])
	if ((indexLeafAt__10 == ordinal([1,0],'0',1)) and (0 < indexLeafAtPileLess1 - indexLeafAt__10 <= 2**(state.dimensionsTotal - 4)) and (0 < (indexLeafAt11ones0 - indexLeafAtPileLess1) <= 2**(state.dimensionsTotal - 3))):
		listAddendIndicesExcluded.extend([ptount(indexLeafAtPileLess1), state.dimensionsTotal - 3, state.dimensionsTotal - 4])
	return addendsToListIndexLeavesAtPile(getDictionaryAddends4Next(state)[indexLeafAtPileLess1], indexLeafAtPileLess1, listAddendIndicesExcluded)

def pinPile01ones1IndexLeaf(state: EliminationState) -> list[int]:
	productsOfDimensions:		list[int] = [prod(state.mapShape[0:dimension]) for dimension in range(state.dimensionsTotal + 1)]
	sumsProductsOfDimensions:	list[int] = [sum(productsOfDimensions[0:dimension]) for dimension in range(state.dimensionsTotal + 1)]

	dictionaryPileToLeaves: dict[int, list[int]] = getDictionaryPileToIndexLeaves(state)
	ordinal: Callable[[int | list[int], str, int | list[int]], int] = numeralOfLengthInBase(positions=state.dimensionsTotal, base=state.mapShape[0])
	listRemoveIndexLeaves: list[int] = []

	pileExcluder: int = 一
	indexLeafAtPileExcluder: int = state.pinnedLeaves[pileExcluder]
	for d in range(state.dimensionsTotal):
		if d < state.dimensionsTotal - 2:
			indexLeaf: int = dictionaryPileToLeaves[pileExcluder][d]
			if indexLeaf == indexLeafAtPileExcluder:
				listRemoveIndexLeaves.extend([一, 首零(state.dimensionsTotal) + indexLeafAtPileExcluder])
		if 0 < d < state.dimensionsTotal - 2:
			indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
			if indexLeaf == indexLeafAtPileExcluder:
				listRemoveIndexLeaves.extend([一 + indexLeafAtPileExcluder])
		if d == 1:
			indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
			if indexLeaf == indexLeafAtPileExcluder:
				listRemoveIndexLeaves.extend([首零(state.dimensionsTotal) + indexLeafAtPileExcluder + 1])
		if d == state.dimensionsTotal - 2:
			indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
			if indexLeaf == indexLeafAtPileExcluder:
				listRemoveIndexLeaves.extend([ordinal([0,1],'0',0), ordinal([0,1],'0',0) + indexLeafAtPileExcluder])
	del pileExcluder

	pileExcluder = ordinal([1,1],'1',0)
	indexLeafAtPileExcluder = state.pinnedLeaves[pileExcluder]
	for d in range(state.dimensionsTotal):
		if d == 0:
			indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
			if indexLeaf == indexLeafAtPileExcluder:
				listRemoveIndexLeaves.extend([一])
		if d < state.dimensionsTotal - 2:
			indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
			if indexLeaf == indexLeafAtPileExcluder:
				listRemoveIndexLeaves.extend([ordinal([0,1],'0',0) + indexLeafAtPileExcluder])
		if 0 < d < state.dimensionsTotal - 2:
			indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
			if indexLeaf == indexLeafAtPileExcluder:
				listRemoveIndexLeaves.extend([2**d, ordinal([0,1],'0',0) + indexLeafAtPileExcluder - (2**d - 0b000001)])
		if 0 < d < state.dimensionsTotal - 3:
			indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
			if indexLeaf == indexLeafAtPileExcluder:
				listRemoveIndexLeaves.extend([0b000001 + indexLeafAtPileExcluder])
		if 0 < d < state.dimensionsTotal - 1:
			indexLeaf = dictionaryPileToLeaves[pileExcluder][d]
			if indexLeaf == indexLeafAtPileExcluder:
				listRemoveIndexLeaves.extend([ordinal([0,1],'0',0)])
	del pileExcluder

	"""pileExcluder = 一+零
	(2,) * 6:
	21      [28]    False   [2, 4, 8, 19, 21, 25, 35, 49, 52, 56]   [2, 4, 8, 19, 21, 25, 28, 35, 49, 52, 56]
	25      [59]    False   [2, 8, 19, 25, 41, 49, 56]      [2, 8, 19, 25, 41, 49, 56, 59]
	"""
	pileExcluder = 一+零
	indexLeafAtPileExcluder = state.pinnedLeaves[pileExcluder]

	if indexLeafAtPileExcluder == 0b001101:
		listRemoveIndexLeaves.extend([0b000111, ordinal([1], '0', [1, 0, 1])])

	if indexLeafAtPileExcluder == ordinal([0,1], '0', [1,0,1]):
		listRemoveIndexLeaves.extend([ordinal([0,0,1], '0',[0]), ordinal([0,1,1], '0',[1]), ordinal([1,1,1], '0',[0])])

	if indexLeafAtPileExcluder == ordinal([0,1,1], '0', [1]):
		listRemoveIndexLeaves.extend([ordinal([0,1], '0',[1,1]), ordinal([1,0,1], '0',[1])])

	if indexLeafAtPileExcluder == ordinal([1,1], '0', [1]):
		listRemoveIndexLeaves.extend([ordinal([1,1,1], '0', 0)])

	if is_odd(indexLeafAtPileExcluder):
		listRemoveIndexLeaves.extend([indexLeafAtPileExcluder, productsOfDimensions[raiseIfNone(dimensionSecondNearest首(indexLeafAtPileExcluder))]])

		if indexLeafAtPileExcluder < 首零(state.dimensionsTotal):
			comebackOffset: int = sumsProductsOfDimensions[ptount(indexLeafAtPileExcluder) + 1]
			listRemoveIndexLeaves.extend([
				一
				, indexLeafAtPileExcluder + ordinal([0,1],'1',1)
				, indexLeafAtPileExcluder + ordinal([0,1],'1',1) - comebackOffset
			])
			if ptount(indexLeafAtPileExcluder) == 1:
				listRemoveIndexLeaves.extend([
					productsOfDimensions[dimensionNearest首(indexLeafAtPileExcluder)] + comebackOffset
					, 首零(state.dimensionsTotal) + comebackOffset
				])

		if 首零(state.dimensionsTotal) < indexLeafAtPileExcluder:
			listRemoveIndexLeaves.extend([ordinal([1,1],'0',1), productsOfDimensions[dimensionNearest首(indexLeafAtPileExcluder) - 1]])

	del pileExcluder

	"""pileExcluder = ordinal([1,1],'1',[0,1])
	(2,) * 5:
	22      [28]    False   [2, 4, 8, 19, 21, 22, 25, 26]   [2, 4, 8, 19, 21, 22, 25, 26, 28]

	38      [21]    False   [2, 4, 16, 35, 37, 38, 49, 50]  [2, 4, 16, 21, 35, 37, 38, 49, 50]
	42      [37]    False   [2, 4, 14, 16, 35, 38, 41, 42, 49, 50]  [2, 4, 14, 16, 35, 37, 38, 41, 42, 49, 50]
	"""
	pileExcluder = ordinal([1,1],'1',[0,1])
	indexLeafAtPileExcluder = state.pinnedLeaves[pileExcluder]
	if 首零(state.dimensionsTotal) < indexLeafAtPileExcluder:
		listRemoveIndexLeaves.extend([ordinal([1,1],'0',1), indexLeafAtPileExcluder])

		if is_even(indexLeafAtPileExcluder):
			listRemoveIndexLeaves.extend([ordinal([0,1],'0',0)])
			bit = 1
			if bit_test(indexLeafAtPileExcluder, bit):
				listRemoveIndexLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 0b000001])
				listRemoveIndexLeaves.extend([state.leavesTotal - sum(productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 2
			if bit_test(indexLeafAtPileExcluder, bit):
				listRemoveIndexLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 0b000001])
				if 1 < coordinatesOf0AtTail(indexLeafAtPileExcluder):
					listRemoveIndexLeaves.extend([state.leavesTotal - sum(productsOfDimensions[bit: state.dimensionsTotal - 2])])
			bit = 3
			if bit_test(indexLeafAtPileExcluder, bit):
				if 1 < coordinatesOf0AtTail(indexLeafAtPileExcluder):
					listRemoveIndexLeaves.extend([2**bit])
					listRemoveIndexLeaves.extend([state.leavesTotal - sum(productsOfDimensions[bit: state.dimensionsTotal - 2])])
				if coordinatesOf0AtTail(indexLeafAtPileExcluder) < bit:
					listRemoveIndexLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 0b000001])

			sheepOrGoat = 0

			shepherdOfDimensions: int = 2**(state.dimensionsTotal - 5)
			if (indexLeafAtPileExcluder//shepherdOfDimensions) & bit_mask(5) == 0b10101:
				listRemoveIndexLeaves.extend([0b000100])
				sheepOrGoat = ptount(indexLeafAtPileExcluder//shepherdOfDimensions)
				if 0 < sheepOrGoat < state.dimensionsTotal - 3:
					comebackOffset = 2**dimensionNearest首(indexLeafAtPileExcluder) - 0b100
					listRemoveIndexLeaves.extend([indexLeafAtPileExcluder - comebackOffset])
				if 0 < sheepOrGoat < state.dimensionsTotal - 4:
					comebackOffset = 2**raiseIfNone(dimensionSecondNearest首(indexLeafAtPileExcluder)) - 0b100
					listRemoveIndexLeaves.extend([indexLeafAtPileExcluder - comebackOffset])

		if is_odd(indexLeafAtPileExcluder):
			listRemoveIndexLeaves.extend([一])
			if indexLeafAtPileExcluder & bit_mask(4) == 0b001001:
				listRemoveIndexLeaves.extend([0b001011])
			sheepOrGoat = ptount(indexLeafAtPileExcluder)
			if 0 < sheepOrGoat < state.dimensionsTotal - 3:
				comebackOffset = 2**dimensionNearest首(indexLeafAtPileExcluder) - 0b10
				listRemoveIndexLeaves.extend([indexLeafAtPileExcluder - comebackOffset])
			if 0 < sheepOrGoat < state.dimensionsTotal - 4:
				comebackOffset = 2**raiseIfNone(dimensionSecondNearest首(indexLeafAtPileExcluder)) - 0b10
				listRemoveIndexLeaves.extend([indexLeafAtPileExcluder - comebackOffset])

	pileExcluder = 0b000100
	indexLeafAtPileExcluder = state.pinnedLeaves[pileExcluder]

	if is_even(indexLeafAtPileExcluder):
		listRemoveIndexLeaves.extend([一, indexLeafAtPileExcluder + 1, 首零(state.dimensionsTotal) + 一+零])
	if is_odd(indexLeafAtPileExcluder):
		listRemoveIndexLeaves.extend([indexLeafAtPileExcluder - 1])
		if ordinal([0,1],'0',0) < indexLeafAtPileExcluder < 首零(state.dimensionsTotal):
			listRemoveIndexLeaves.extend([ordinal([0,1],'0',0) + 一+零, ordinal([1,1],'0',1)])
		if 首零(state.dimensionsTotal) < indexLeafAtPileExcluder:
			listRemoveIndexLeaves.extend([ordinal([0,1],'0',0), ordinal([1,1],'0',1)])
			bit = 1
			if bit_test(indexLeafAtPileExcluder, bit):
				listRemoveIndexLeaves.extend([2**bit, 首零(state.dimensionsTotal) + 2**bit + 0b000001])
			bit = 2
			if bit_test(indexLeafAtPileExcluder, bit):
				listRemoveIndexLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 0b000001])
			bit = 3
			if bit_test(indexLeafAtPileExcluder, bit):
				listRemoveIndexLeaves.extend([首零(state.dimensionsTotal) + 2**bit + 0b000001])
			bit = 4
			if bit_test(indexLeafAtPileExcluder, bit) and (indexLeafAtPileExcluder.bit_length() > 5):
				listRemoveIndexLeaves.extend([ordinal([1,1,1],'0',0)])
	del pileExcluder

# ------- Tools for creating rules ------------------------
	"""
	listRemoveIndexLeaves: list[int] = []
	dictionaryExcludedIndexLeaves = getExcludedIndexLeaves(state, state.pile, (pileExcluder,))
	if print1Time < 1:
		print1Time += 1
		pprint(dictionaryExcludedIndexLeaves, width=140)

	listExcludedIndexLeavesGoal = dictionaryExcludedIndexLeaves[indexLeafAtPileExcluder]
	print(indexLeafAtPileExcluder, sorted(set(listExcludedIndexLeavesGoal).difference(set(listRemoveIndexLeaves)))
		, listExcludedIndexLeavesGoal == sorted(set(listRemoveIndexLeaves)), sorted(set(listRemoveIndexLeaves)), listExcludedIndexLeavesGoal, sep='\t')
	"""

	indexLeafAt__10 = state.pinnedLeaves[一]
	indexLeafAt11ones0 = state.pinnedLeaves[ordinal([1,1],'1',0)]
	indexLeafAt__11 = state.pinnedLeaves[一+零]
	indexLeafAt11ones01 = state.pinnedLeaves[ordinal([1,1],'1',[0,1])]

	if (indexLeafAt__11 != ordinal([1,1],'0',1)) and (indexLeafAt11ones0 == ordinal([1,1],'0',0)):
		listRemoveIndexLeaves.append(一)
	if (indexLeafAt11ones01 != ordinal([1,0],'0',1) + getDictionaryAddends4Prior(state)[ordinal([1,0],'0',1)][0]) and (indexLeafAt__10 == 一+零):
		listRemoveIndexLeaves.append(ordinal([0,1],'0',0))

	if (indexLeafAt__10 == ordinal([0,0,1],'0',1)) and (indexLeafAt11ones0 == ordinal([1,1],'0',0)):
		listRemoveIndexLeaves.extend([ordinal([0,0,1],'0',0), ordinal([1,1,1],'0',0)])
	if indexLeafAt__10 == ordinal([1,0],'0',1):
		listRemoveIndexLeaves.extend([ordinal([0,1],'0',0), indexLeafAt11ones0 + 0b000001])
	if indexLeafAt__10.bit_length() < state.dimensionsTotal - 2:
		listRemoveIndexLeaves.extend([一, indexLeafAt11ones0 + 一])

# ------- Tools for creating rules ------------------------
	"""
	tuplePilesExcluders = (一, ordinal([1,1],'1',0))
	(indexLeavesAtPilesExcluders) = (indexLeafAt__10, indexLeafAt11ones0)
	listRemoveIndexLeaves: list[int] = []
	dictionaryExcludedIndexLeaves = getExcludedIndexLeaves(state, state.pile, tuplePilesExcluders)
	if print1Time < 1:
		print1Time += 1
		pprint(dictionaryExcludedIndexLeaves, width=140)
	listExcludedIndexLeavesGoal = dictionaryExcludedIndexLeaves[((indexLeavesAtPilesExcluders))]
	print(indexLeavesAtPilesExcluders, sorted(set(listExcludedIndexLeavesGoal).difference(set(listRemoveIndexLeaves)))
		, listExcludedIndexLeavesGoal == sorted(set(listRemoveIndexLeaves)), sorted(set(listRemoveIndexLeaves)), listExcludedIndexLeavesGoal, sep='\t')

	"""
	return sorted(set(dictionaryPileToLeaves[state.pile]).difference(set(listRemoveIndexLeaves)))

def pinPile11ones0Addend(state: EliminationState) -> list[int]:
	listAddendIndicesExcluded: list[int] = []
	indexLeafAtPilePlus1: int = state.pinnedLeaves[state.pile + 1]
	indexLeafAt__10: int | None = state.pinnedLeaves.get(一)
	if indexLeafAt__10 and (indexLeafAt__10.bit_length() < state.dimensionsTotal):
		listAddendIndicesExcluded.extend([*range(0b000001, indexLeafAt__10.bit_length())])
	return addendsToListIndexLeavesAtPile(getDictionaryAddends4Prior(state)[indexLeafAtPilePlus1], indexLeafAtPilePlus1, listAddendIndicesExcluded)

def pinPile11ones01Addend(state: EliminationState) -> list[int]:
	ordinal: Callable[[int | list[int], str, int | list[int]], int] = numeralOfLengthInBase(positions=state.dimensionsTotal, base=state.mapShape[0])
	listAddendIndicesExcluded: list[int] = []
	indexLeafAtPilePlus1: int = state.pinnedLeaves[state.pile + 1]
	if indexLeafAtPilePlus1 < ordinal([1,1],'0',0):
		listAddendIndicesExcluded.append(-1)
	indexLeafAt__10 = state.pinnedLeaves[一]
	if (indexLeafAtPilePlus1 == ordinal([1,0],'0',1)) and (indexLeafAt__10 != 一+零):
		listAddendIndicesExcluded.extend([*range(indexLeafAt__10.bit_length() - 2)])
	return addendsToListIndexLeavesAtPile(getDictionaryAddends4Prior(state)[indexLeafAtPilePlus1], indexLeafAtPilePlus1, listAddendIndicesExcluded)

def pinPile11ones1Fixed(state: EliminationState) -> list[int]:
	listIndexLeavesAtPile: list[int] = [首零(state.dimensionsTotal)]
	return listIndexLeavesAtPile

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

